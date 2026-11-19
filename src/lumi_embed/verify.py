"""Verificações obrigatórias num checkpoint real (seção 3.6), na GPU, e exportação do R0.

T1 invariância a padding e a lote · T2 causalidade (trilho C) / bidirecionalidade (B)
· T3 posição do pooling · T4 ida e volta (lumi_embed e Sentence-Transformers).
Rodam em float32 para separar erro de lógica de ruído numérico do bf16; o resultado vai
para checkpoints/<ID>/verify.json. T5 (contaminação) e T6 (smoke vs R0) vêm do relatório.
"""

from __future__ import annotations

import json
import logging
import tempfile

import torch
import torch.nn.functional as F

from .modeling.embedder import LumiEmbedder
from .modeling.export import save_embedder
from .utils import EmbedError, ckpt_dir, run_metadata, write_json

log = logging.getLogger("lumi_embed.verify")

TEXTS = [
    "As despesas médicas podem ser deduzidas da base de cálculo do imposto de renda.",
    "Qual é a capital do Brasil?",
    "O Supremo Tribunal Federal julgou a ação direta de inconstitucionalidade proposta pelo partido.",
    "Recife.",
    "A fotossíntese converte energia luminosa em energia química armazenada nas moléculas de glicose.",
]


def export_base(cfg: dict) -> str:
    """R0: o lumi-4b sem treino, exportado no formato de embedder (também é a âncora dos merges)."""
    from transformers import AutoModel

    m = cfg["model"]
    out = ckpt_dir("R0") / "model"
    if (out / "meta.json").exists():
        return str(out)
    _, info = AutoModel.from_pretrained(m["hf_id"], revision=m["revision"], dtype=torch.bfloat16,
                                        output_loading_info=True)
    if info.get("missing_keys"):
        raise EmbedError(f"Chaves ausentes ao carregar o backbone: {info['missing_keys'][:10]}")
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith("lm_head")]
    if unexpected:
        raise EmbedError(f"Chaves inesperadas ao carregar o backbone: {unexpected[:10]}")
    model = LumiEmbedder.from_pretrained(m["hf_id"], revision=m["revision"], spec=cfg.get("embedding", {}),
                                         use_hub_kernels=False)
    meta = run_metadata(None, ckpt_id="R0", source=f"{m['hf_id']}@{m['revision']}", training="nenhum",
                        loading_info={"unexpected_keys": info.get("unexpected_keys", [])})
    save_embedder(model, out, meta=meta)
    log.info("R0 exportado em %s", out)
    return str(out)


def verify_checkpoint(ckpt_id: str) -> dict:
    path = str(ckpt_dir(ckpt_id) / "model")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = LumiEmbedder.from_pretrained(path, dtype=torch.float32, use_hub_kernels=False, device=dev).eval()
    res: dict = {"ckpt": ckpt_id, "direction": m.spec.direction, "pooling": m.spec.pooling}
    alone = torch.cat([m.encode([t]) for t in TEXTS])
    batched = m.encode(TEXTS, batch_size=len(TEXTS))
    shuffled = m.encode(TEXTS[::-1], batch_size=2).flip(0)
    cos = torch.minimum(F.cosine_similarity(alone, batched), F.cosine_similarity(alone, shuffled))
    res["T1"] = {"min_cos": float(cos.min()), "ok": bool(cos.min() >= 0.9999)}

    ids = m.tokenize([TEXTS[0]])["input_ids"].to(dev)
    ids2 = ids.clone()
    ids2[0, -2] = (ids2[0, -2] + 101) % len(m.tokenizer)
    with torch.no_grad():
        a = m.hidden_states(ids, torch.ones_like(ids), output_hidden_states=True).hidden_states
        b = m.hidden_states(ids2, torch.ones_like(ids2), output_hidden_states=True).hidden_states
    pre = slice(0, ids.shape[1] - 2)
    diffs = [float((x[0, pre] - y[0, pre]).abs().max()) for x, y in zip(a, b)]
    if m.spec.direction == "C":
        res["T2"] = {"max_prefix_change": max(diffs), "ok": max(diffs) <= 1e-4}
    else:
        res["T2"] = {"max_prefix_change": max(diffs), "ok": max(diffs) > 1e-3}

    enc = m.tokenize(TEXTS)
    with torch.no_grad():
        h = m.hidden_states(enc["input_ids"].to(dev), enc["attention_mask"].to(dev)).last_hidden_state
        got = m(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
    mask = enc["attention_mask"]
    manual = []
    for i in range(len(TEXTS)):
        n = int(mask[i].sum())
        manual.append(h[i, n - 1] if m.spec.pooling == "eos" else h[i, :n].mean(0))
    manual = F.normalize(torch.stack(manual).float(), dim=-1)
    res["T3"] = {"max_abs_diff": float((got - manual).abs().max()), "ok": bool(torch.allclose(got, manual, atol=1e-4))}

    with tempfile.TemporaryDirectory() as tmp:
        out = save_embedder(m, f"{tmp}/rt", dtype=torch.float32)
        again = LumiEmbedder.from_pretrained(str(out), dtype=torch.float32, use_hub_kernels=False, device=dev)
        c1 = float(F.cosine_similarity(again.encode(TEXTS), alone).min())
        t4 = {"lumi_embed_min_cos": c1}
        if m.spec.direction == "C":
            from sentence_transformers import SentenceTransformer

            st = SentenceTransformer(str(out), device=dev, model_kwargs={"dtype": torch.float32})
            st_emb = torch.tensor(st.encode(TEXTS, convert_to_numpy=True))
            t4["sentence_transformers_min_cos"] = float(F.cosine_similarity(st_emb, alone).min())
        t4["ok"] = all(v >= 0.9999 for k, v in t4.items() if k.endswith("min_cos"))
        res["T4"] = t4
    res["ok"] = all(res[k]["ok"] for k in ("T1", "T2", "T3", "T4"))
    write_json(ckpt_dir(ckpt_id) / "verify.json", {**res, **run_metadata(None)})
    log.info("Verificação %s: %s", ckpt_id, {k: res[k]["ok"] for k in ("T1", "T2", "T3", "T4")})
    if not res["ok"]:
        raise EmbedError(f"Verificação falhou para {ckpt_id}: {json.dumps(res)}")
    return res


def bench(cfg: dict, seq_len: int = 512, batch: int = 32, fallback_batch: int = 4) -> dict:
    """Vazão de forward/backward no backbone real, com e sem kernels do Hub.

    A fase com kernels verifica antes que o Mamba-2 não usa o caminho PyTorch puro (falha se usar).
    A fase sem kernels usa um lote menor: o scan em PyTorch puro guarda tensores de tamanho
    lote × cabeças × blocos × 128² por camada para o backward e não cabe com 32 × 512 tokens.
    A comparação é por token; um OOM vira resultado em vez de derrubar o job.
    """
    import gc
    import time

    m = cfg["model"]
    out = {}
    for name, kernels, b in (("hub_kernels", True, batch), ("torch_fallback", False, fallback_batch)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        res = {"batch": b, "seq_len": seq_len}
        model = None
        try:
            model = LumiEmbedder.from_pretrained(m["hf_id"], revision=m["revision"], use_hub_kernels=kernels,
                                                 device="cuda")
            if kernels:
                res["kernel_check"] = model.kernel_info
            ids = torch.randint(100, 50000, (b, seq_len), device="cuda")
            mask = torch.ones_like(ids)
            model.train()
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(2):
                    model(ids, mask).sum().backward()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    model(ids, mask).sum().backward()
                torch.cuda.synchronize()
            dt = (time.time() - t0) / 5
            res.update(tokens_per_s_fwd_bwd=round(b * seq_len / dt), sec_per_step=round(dt, 3),
                       peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        except torch.OutOfMemoryError as exc:
            res.update(error="OOM", peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1),
                       detail=str(exc).split(".")[0])
        finally:
            if model is not None:
                model.zero_grad(set_to_none=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        out[name] = res
        log.info("bench %s: %s", name, res)
    hk, tf = out["hub_kernels"], out["torch_fallback"]
    if "tokens_per_s_fwd_bwd" in hk and "tokens_per_s_fwd_bwd" in tf:
        out["speedup_hub_vs_torch"] = round(hk["tokens_per_s_fwd_bwd"] / tf["tokens_per_s_fwd_bwd"], 2)
    log.info("bench: %s", out)
    return out
