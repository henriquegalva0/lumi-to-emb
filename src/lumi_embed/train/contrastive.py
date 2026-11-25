"""Treino contrastivo (estágios S1, S2, S-long e a referência R1) — seções 3.4 e 3.6 do plano.

Uma configuração (configs/embed/train/<ID>.yaml) define um checkpoint. O treino:
- parte de `init` (o lumi-4b do Hub ou um checkpoint anterior, com pesos já fundidos);
- usa LoRA (peft) ou fine-tuning completo (pesos mestres fp32 + autocast bf16);
- com perda `hn` acumula gradiente por micro-lote; com `inbatch` usa GradCache,
  para o batch efetivo valer como conjunto de negativos;
- salva o estado a cada `save_every_min` minutos e ao receber SIGTERM, e retoma sozinho;
- avalia no dev a cada `eval_every` passos e no fim;
- no fim funde o LoRA e exporta o embedder (HF + Sentence-Transformers) com semente,
  commit e configuração (regra A.3).
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import signal
import time
from pathlib import Path

import torch

from ..modeling.embedder import EmbedSpec, LumiEmbedder, format_query
from ..modeling.export import save_embedder
from ..utils import (EmbedError, ckpt_dir, data_dir, ensure_free_space, iter_jsonl, run_metadata,
                     seed_everything, write_json)
from .gradcache import gradcache_step
from .losses import LossConfig, contrastive_loss

log = logging.getLogger("lumi_embed.train")

DEFAULT_LORA_TARGETS = ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
# `out_proj` dos mixers Mamba fica de fora: no treino, o kernel fundido do Mamba-2 recebe
# `out_proj.weight` diretamente, então um adaptador ali seria ignorado no treino e aplicado
# na inferência.


# --------------------------------------------------------------------------- dados


def resolve_data(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else data_dir(path)


def load_examples(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise EmbedError(f"Conjunto de treino não encontrado: {path}. Rode as etapas de dados antes.")
    rows = list(iter_jsonl(path, limit))
    if not rows:
        raise EmbedError(f"Conjunto vazio: {path}")
    return rows


def example_texts(ex: dict, spec: EmbedSpec, k_neg: int, neg_field: str = "negatives"):
    """(query, positivo, [negativos]) já com instrução aplicada (na query; nos dois lados se simétrico)."""
    inst = ex.get("instruction") or None
    q = format_query(ex["query"], inst, spec)
    symmetric = ex.get("task") in ("sts", "bitext", "clustering") or ex.get("symmetric", False)
    side = (lambda t: format_query(t, inst, spec)) if symmetric else (lambda t: t)
    pos = side(ex["positive"])
    negs = [side(n["text"] if isinstance(n, dict) else n) for n in (ex.get(neg_field) or [])[:k_neg]]
    return q, pos, negs


# --------------------------------------------------------------------------- modelo


def resolve_init(init: str) -> tuple[str, str | None]:
    """'hf:repo@rev' → (repo, rev); 'ckpt:ID' → (checkpoints/ID/model, None); caminho local."""
    if init.startswith("hf:"):
        repo, _, rev = init[3:].partition("@")
        return repo, rev or None
    if init.startswith("ckpt:"):
        path = ckpt_dir(init[5:]) / "model"
        if not (path / "config.json").exists():
            raise EmbedError(f"Checkpoint de partida não encontrado: {path} (treine {init[5:]} antes).")
        return str(path), None
    return init, None


def build_model(cfg: dict, device) -> LumiEmbedder:
    tcfg = cfg["train"]
    full = tcfg.get("method", "lora") == "full"
    path, rev = resolve_init(cfg["init"])
    model = LumiEmbedder.from_pretrained(
        path,
        revision=rev,
        spec=cfg.get("embedding", {}),
        dtype=torch.float32 if full else torch.bfloat16,
        attn_implementation=cfg.get("model", {}).get("attn_implementation", "sdpa"),
        use_hub_kernels=cfg.get("model", {}).get("use_hub_kernels", True),
        device=device,
    )
    if tcfg.get("gradient_checkpointing", True):
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if full:
        for p in model.backbone.parameters():
            p.requires_grad_(True)
    else:
        from peft import LoraConfig, get_peft_model

        lcfg = tcfg.get("lora", {})
        lora = LoraConfig(
            r=int(lcfg.get("r", 16)),
            lora_alpha=int(lcfg.get("alpha", 16)),
            lora_dropout=float(lcfg.get("dropout", 0.0)),
            target_modules=lcfg.get("target_modules", DEFAULT_LORA_TARGETS),
            bias="none",
        )
        model.backbone = get_peft_model(model.backbone, lora)
        for n, p in model.backbone.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()  # adaptadores em fp32; base congelada em bf16
    n_train = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.backbone.parameters())
    log.info("Parâmetros treináveis: %.1fM de %.2fB (%s)", n_train / 1e6, n_all / 1e9, tcfg.get("method", "lora"))
    return model


# --------------------------------------------------------------------------- estado


class TrainState:
    def __init__(self, out: Path):
        self.dir = out / "train_state"

    def exists(self) -> bool:
        return (self.dir / "state.json").exists()

    def save(self, model, optim, sched, info: dict) -> None:
        tmp = self.dir.with_name("train_state.tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        trainable = {n: p.detach().cpu() for n, p in model.backbone.named_parameters() if p.requires_grad}
        torch.save(trainable, tmp / "trainable.pt")
        torch.save({"optim": optim.state_dict(), "sched": sched.state_dict(),
                    "rng": {"py": random.getstate(), "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}},
                   tmp / "optim.pt")
        (tmp / "state.json").write_text(json.dumps(info, indent=2))
        if self.dir.exists():
            shutil.rmtree(self.dir)
        tmp.rename(self.dir)

    def load(self, model, optim, sched) -> dict:
        trainable = torch.load(self.dir / "trainable.pt", map_location="cpu")
        params = dict(model.backbone.named_parameters())
        with torch.no_grad():
            for n, t in trainable.items():
                params[n].copy_(t.to(params[n].device, params[n].dtype))
        st = torch.load(self.dir / "optim.pt", map_location="cpu", weights_only=False)
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        random.setstate(st["rng"]["py"])
        torch.set_rng_state(st["rng"]["torch"])
        if st["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        return json.loads((self.dir / "state.json").read_text())


# --------------------------------------------------------------------------- laço


def warmup_steps(tcfg: dict, total: int) -> int:
    """Passos de aquecimento: `warmup_steps` explícito ou `warmup_ratio` × total (padrão 10 %).

    A proporção acompanha a escala: com 195 passos no S1, 100 passos fixos seriam metade do treino,
    e numa varredura de LR curta a taxa nunca chegaria ao valor testado.
    """
    if tcfg.get("warmup_steps") is not None:
        return int(tcfg["warmup_steps"])
    return max(1, math.ceil(float(tcfg.get("warmup_ratio", 0.1)) * total))


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return f


def train(cfg: dict, ckpt_id: str, *, max_minutes: float | None = None, dev_eval_fn=None) -> Path:
    tcfg = cfg["train"]
    seed = int(cfg.get("seed", 1234))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ckpt_dir(ckpt_id)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model" / "meta.json").exists():
        log.info("%s já está treinado em %s (apague a pasta para refazer).", ckpt_id, out / "model")
        return out / "model"
    ensure_free_space(out, float(tcfg.get("min_free_gb", 20)) * 1e9, f"o treino {ckpt_id} (estado e modelo exportado)")

    data = load_examples(resolve_data(cfg["data"]["train"]), cfg["data"].get("limit"))
    dev = None
    if cfg["data"].get("dev") and dev_eval_fn is not None:
        dv = cfg["data"]["dev"]
        devp = [resolve_data(x) for x in (dv if isinstance(dv, list) else [dv])]
        if all(x.exists() for x in devp):
            from ..eval.dev import load_dev_set

            dev = load_dev_set(devp)
            log.info("dev: %d exemplos + distratores %s", len(dev.rows),
                     {t: len(v) for t, v in dev.distractors.items()})
    spec = EmbedSpec.from_dict(cfg.get("embedding", {}))
    loss_cfg = LossConfig.from_dict(cfg.get("loss", {}))
    k_neg = int(cfg["data"].get("negatives_per_example", 1))
    neg_field = cfg["data"].get("negatives_field", "negatives")
    batch = int(tcfg["batch_size"])
    micro = int(tcfg.get("micro_batch", 16))
    epochs = float(tcfg.get("epochs", 1))
    steps_per_epoch = max(1, len(data) // batch)
    total = int(tcfg.get("max_steps") or math.ceil(steps_per_epoch * epochs))
    log.info("%s: %d exemplos, batch %d, %d passos (%.2f épocas), perda %s, %d negativo(s)/exemplo",
             ckpt_id, len(data), batch, total, total / steps_per_epoch, loss_cfg.mode, k_neg)

    model = build_model(cfg, device)
    model.train()
    params = [p for p in model.backbone.parameters() if p.requires_grad]
    ocfg = tcfg.get("optimizer", {})
    optim = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(ocfg.get("weight_decay", 0.01)),
                              betas=tuple(ocfg.get("betas", (0.9, 0.999))), fused=device.type == "cuda")
    warmup = warmup_steps(tcfg, total)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda(warmup, total))
    log.info("%s: %d passos (aquecimento %d), lr %.2e, lote %d", ckpt_id, total, warmup, float(tcfg["lr"]),
             int(tcfg["batch_size"]))

    state = TrainState(out)
    step, epoch, cursor = 0, 0, 0
    if state.exists():
        info = state.load(model, optim, sched)
        step, epoch, cursor = info["step"], info["epoch"], info["cursor"]
        log.info("Retomando %s do passo %d (época %d)", ckpt_id, step, epoch)

    def order_for(ep: int) -> list[int]:
        idx = list(range(len(data)))
        random.Random(seed * 1000 + ep).shuffle(idx)
        return idx

    order = order_for(epoch)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    log_path = out / "train_log.jsonl"
    t_start = last_save = time.time()
    save_every = float(tcfg.get("save_every_min", 30)) * 60
    eval_every = int(tcfg.get("eval_every", 0) or 0)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    phase = (spec.max_length, micro)

    while step < total and not stop["flag"]:
        if cursor + batch > len(order):
            epoch, cursor = epoch + 1, 0
            order = order_for(epoch)
        idx = order[cursor : cursor + batch]
        cursor += batch
        t0 = time.time()
        micro_now = micro
        # currículo de comprimento (S-long): [fração, tokens] ou [fração, tokens, micro-lote]. O micro-lote só
        # muda a memória: com a perda hn cada linha usa só os próprios negativos, e o gradiente é o mesmo.
        for entry in tcfg.get("length_schedule", []):
            if step >= float(entry[0]) * total:
                spec.max_length = int(entry[1])
                micro_now = int(entry[2]) if len(entry) > 2 else micro
        if (spec.max_length, micro_now) != phase:
            phase = (spec.max_length, micro_now)
            log.info("Fase a partir do passo %d: até %d tokens, micro-lote %d", step + 1, *phase)
        rows = [example_texts(data[i], spec, k_neg, neg_field) for i in idx]
        optim.zero_grad(set_to_none=True)
        with autocast:
            loss = _step(model, rows, loss_cfg, micro_now, spec, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(tcfg.get("max_grad_norm", 1.0)))
        optim.step()
        sched.step()
        step += 1
        rec = {"step": step, "epoch": epoch, "loss": round(loss, 5), "lr": sched.get_last_lr()[0],
               "grad_norm": round(float(grad_norm), 4), "sec": round(time.time() - t0, 2)}
        if step % int(tcfg.get("log_every", 10)) == 0 or step == 1:
            log.info("passo %d/%d | perda %.4f | lr %.2e | norma %.3f | %.1fs", step, total, loss, rec["lr"],
                     rec["grad_norm"], rec["sec"])
        if dev is not None and eval_every and step % eval_every == 0:
            rec["dev"] = dev_eval_fn(model, dev)
            log.info("dev no passo %d: %s", step, rec["dev"])
            model.train()
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        elapsed_min = (time.time() - t_start) / 60
        out_of_time = max_minutes is not None and elapsed_min >= max_minutes
        if time.time() - last_save > save_every or out_of_time:
            state.save(model, optim, sched, {"step": step, "epoch": epoch, "cursor": cursor})
            last_save = time.time()
            if out_of_time and step < total:
                log.warning("Tempo limite de %.0f min atingido; estado salvo. Reenvie o job para continuar.", max_minutes)
                return out
    if stop["flag"]:
        state.save(model, optim, sched, {"step": step, "epoch": epoch, "cursor": cursor})
        log.warning("SIGTERM recebido: estado salvo no passo %d. Reenvie o job para continuar.", step)
        return out

    # ---- fim: funde LoRA, avalia e exporta
    if hasattr(model.backbone, "merge_and_unload"):
        model.backbone = model.backbone.merge_and_unload()
    model.eval()
    dev_metrics = dev_eval_fn(model, dev) if dev is not None else None
    meta = run_metadata(cfg, ckpt_id=ckpt_id, seed=seed, steps=step, epochs_done=step / steps_per_epoch,
                        hub_kernels=model.kernel_info,
                        n_train=len(data), dev=dev_metrics,
                        train_minutes=round((time.time() - t_start) / 60, 1))
    save_embedder(model, out / "model", meta=meta)
    write_json(out / "summary.json", {k: v for k, v in meta.items() if k != "config"})
    if state.dir.exists() and not tcfg.get("keep_train_state", False):
        shutil.rmtree(state.dir)
    log.info("%s pronto: %s | dev: %s", ckpt_id, out / "model", dev_metrics)
    return out / "model"


def _encode(model, texts: list[str], spec: EmbedSpec, device, grad: bool):
    enc = model.tokenize(texts, spec.max_length)
    with torch.set_grad_enabled(grad):
        return model(enc["input_ids"].to(device), enc["attention_mask"].to(device))


def _flatten(rows):
    qs = [r[0] for r in rows]
    ps = [r[1] for r in rows]
    negs, owner = [], []
    for i, r in enumerate(rows):
        negs.extend(r[2])
        owner.extend([i] * len(r[2]))
    return qs, ps, negs, owner


def _doc_ids(texts: list[str], device) -> torch.Tensor:
    return torch.tensor([hash(t) & 0x7FFFFFFFFFFF for t in texts], device=device)


def _step(model, rows, loss_cfg: LossConfig, micro: int, spec: EmbedSpec, device) -> float:
    if loss_cfg.mode == "inbatch":
        qs, ps, negs, owner = _flatten(rows)
        encs = [model.tokenize(qs, spec.max_length), model.tokenize(ps, spec.max_length),
                model.tokenize(negs, spec.max_length) if negs else None]
        encs = [None if e is None else {k: v.to(device) for k, v in e.items()} for e in encs]
        owner_t = torch.tensor(owner, device=device, dtype=torch.long)
        pos_ids, neg_ids = _doc_ids(ps, device), _doc_ids(negs, device) if negs else None

        def loss_fn(leaves):
            q, p, n = leaves
            return contrastive_loss(q, p, n, owner_t, loss_cfg, pos_ids, neg_ids)

        return gradcache_step(model, encs, loss_fn, chunk=micro)
    # hn: a perda é por linha; acumula gradiente por micro-lote de queries
    total = 0.0
    b = len(rows)
    for i in range(0, b, micro):
        part = rows[i : i + micro]
        qs, ps, negs, owner = _flatten(part)
        q = _encode(model, qs, spec, device, True)
        p = _encode(model, ps, spec, device, True)
        n = _encode(model, negs, spec, device, True) if negs else None
        owner_t = torch.tensor(owner, device=device, dtype=torch.long)
        loss = contrastive_loss(q, p, n, owner_t, loss_cfg) * (len(part) / b)
        loss.backward()
        total += float(loss.detach())
    return total
