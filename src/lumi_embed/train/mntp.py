"""Estágio S0 do trilho bidirecional: masked next-token prediction (LLM2Vec [6], BidirLM [12]).

Com a bidirecionalidade ligada (Ba ou Bf), uma fração dos tokens é trocada pelo token de
máscara, e cada token mascarado na posição i é previsto pelo logit da posição i-1 (como
no LLM2Vec). Só os logits das posições mascaradas são calculados (o vocabulário tem 131k).
Treino curto com LoRA (MNTP longo causa esquecimento [12]); no fim o LoRA é fundido e o
backbone é exportado como embedder (pooling por média), ponto de partida do B1.
"""

from __future__ import annotations

import logging
import math
import random
import time

import torch
import torch.nn.functional as F

from ..modeling.bidir import apply_direction, build_masks
from ..modeling.embedder import EmbedSpec, LumiEmbedder, prepare_tokenizer
from ..modeling.export import save_embedder
from ..modeling.kernels import ensure_hub_kernels
from ..utils import EmbedError, ckpt_dir, iter_jsonl, run_metadata, seed_everything, write_json
from .contrastive import DEFAULT_LORA_TARGETS, lr_lambda, resolve_data, resolve_init

log = logging.getLogger("lumi_embed.mntp")


def mask_batch(input_ids, attention_mask, ratio: float, mask_id: int, rng: random.Random):
    """Escolhe posições válidas (1..len-2, sem o EOS) e troca pelo token de máscara."""
    ids = input_ids.clone()
    rows, cols, labels = [], [], []
    for b in range(ids.shape[0]):
        n = int(attention_mask[b].sum())
        cand = list(range(1, n - 1))
        k = max(1, int(round(len(cand) * ratio))) if cand else 0
        for c in rng.sample(cand, k) if k else []:
            rows.append(b)
            cols.append(c)
            labels.append(int(ids[b, c]))
            ids[b, c] = mask_id
    t = torch.tensor
    return ids, t(rows, dtype=torch.long), t(cols, dtype=torch.long), t(labels, dtype=torch.long)


def train_mntp(cfg: dict, ckpt_id: str, max_minutes: float | None = None):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = int(cfg.get("seed", 1234))
    seed_everything(seed)
    rng = random.Random(seed)
    out = ckpt_dir(ckpt_id)
    if (out / "model" / "meta.json").exists():
        log.info("%s já existe em %s", ckpt_id, out / "model")
        return out / "model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = EmbedSpec.from_dict(cfg.get("embedding", {}))
    if spec.direction == "C":
        raise EmbedError("MNTP só faz sentido no trilho bidirecional (direction: Ba ou Bf).")
    tcfg, mcfg = cfg["train"], cfg.get("mntp", {})
    path, rev = resolve_init(cfg["init"])
    kwargs = {"revision": rev, "dtype": torch.bfloat16,
              "attn_implementation": cfg.get("model", {}).get("attn_implementation", "sdpa")}
    lm = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device)
    kernel_info = None
    if cfg.get("model", {}).get("use_hub_kernels", True) and device.type == "cuda":
        kernel_info = ensure_hub_kernels(lm, lm.model)
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(path, revision=rev), spec)
    apply_direction(lm.model, spec.direction)
    mask_id = tok.convert_tokens_to_ids(mcfg.get("mask_token", "<unk>"))
    lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = tcfg.get("lora", {})
    lm = get_peft_model(lm, LoraConfig(r=int(lcfg.get("r", 16)), lora_alpha=int(lcfg.get("alpha", 32)),
                                       target_modules=lcfg.get("target_modules", DEFAULT_LORA_TARGETS),
                                       lora_dropout=0.0, bias="none"))
    for p in lm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    base = lm.get_base_model()
    texts = [r["text"] for r in iter_jsonl(resolve_data(cfg["data"]["train"]), cfg["data"].get("limit"))]
    if not texts:
        raise EmbedError("Sem texto para o MNTP.")
    batch, total = int(tcfg["batch_size"]), int(tcfg["max_steps"])
    micro = int(tcfg.get("micro_batch", batch))
    params = [p for p in lm.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=0.0, fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda(int(tcfg.get("warmup_steps", 0)), total))
    ratio = float(mcfg.get("mask_ratio", 0.3))
    log.info("MNTP %s: direção %s, %d textos, %d passos (lote %d, micro-lote %d), máscara %.0f%%", ckpt_id,
             spec.direction, len(texts), total, batch, micro, 100 * ratio)
    # Modo de treino: no transformers, o gradient checkpointing só age com o módulo em treino. Sem isso, o
    # lote inteiro guarda todas as ativações (o B0 estourou os ~89 GB da classe menor no primeiro passo).
    lm.train()
    t0 = time.time()
    order = list(range(len(texts)))
    rng.shuffle(order)
    cur = 0
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    history = []
    for step in range(1, total + 1):
        if cur + batch > len(order):
            rng.shuffle(order)
            cur = 0
        chunk = [texts[i] for i in order[cur : cur + batch]]
        cur += batch
        enc = tok(chunk, padding=True, truncation=True, max_length=spec.max_length, return_tensors="pt")
        # A máscara é sorteada no lote inteiro (mesmos sorteios com qualquer micro-lote); cada micro-lote
        # soma a sua parte da perda, dividida pelo total de tokens mascarados: o gradiente é o do lote inteiro.
        ids, rows, cols, labels = mask_batch(enc["input_ids"], enc["attention_mask"], ratio, mask_id, rng)
        optim.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for s in range(0, len(chunk), micro):
            sel = (rows >= s) & (rows < s + micro)
            if not bool(sel.any()):
                continue
            mids, mmask = ids[s : s + micro].to(device), enc["attention_mask"][s : s + micro].to(device)
            with autocast:
                embeds = base.model.get_input_embeddings()(mids)
                masks = build_masks(base.model, embeds, mmask)
                h = base.model(inputs_embeds=embeds, attention_mask=masks, use_cache=False).last_hidden_state
                logits = base.lm_head(h[(rows[sel] - s).to(device), (cols[sel] - 1).to(device)])  # i prevista por i-1
                part = F.cross_entropy(logits.float(), labels[sel].to(device), reduction="sum") / len(labels)
            part.backward()
            loss += part.detach()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        sched.step()
        history.append(float(loss.detach()))
        if step % 10 == 0 or step == 1:
            log.info("MNTP passo %d/%d | perda %.4f", step, total, float(loss.detach()))
        if max_minutes is not None and (time.time() - t0) / 60 > max_minutes:
            raise EmbedError("Tempo esgotado no MNTP (estágio curto, sem retomada): aumente max_time ou reduza max_steps.")
    lm.eval()
    merged = lm.merge_and_unload()
    model = LumiEmbedder(merged.model, tok, spec)
    meta = run_metadata(cfg, ckpt_id=ckpt_id, seed=seed, steps=total, hub_kernels=kernel_info,
                        final_loss=float(sum(history[-20:]) / max(1, len(history[-20:]))),
                        minutes=round((time.time() - t0) / 60, 1))
    save_embedder(model, out / "model", meta=meta)
    write_json(out / "summary.json", {k: v for k, v in meta.items() if k != "config"})
    return out / "model"


def perplexity_hint(losses: list[float]) -> float:
    return math.exp(sum(losses) / max(1, len(losses)))
