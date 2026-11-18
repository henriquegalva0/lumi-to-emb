"""Merge de checkpoints no espaço de pesos (seção 4 do plano).

- linear:   θ = Σ w_i θ_i  (Σ w_i = 1; pesos iguais por padrão) [10]
- slerp:    interpolação esférica tensor a tensor entre dois checkpoints [9][22]
- base:     θ(λ) = (1 − λ) θ0 + λ θc, com θ0 = âncora (R0 = lumi-4b exportado, ou pós-S0) [12]

Só se mesclam checkpoints do mesmo trilho (mesma direção e pooling) e com os mesmos
nomes de tensor; o LoRA já vem fundido de cada treino. O processamento é feito tensor a
tensor (safetensors), sem carregar modelos inteiros na memória.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import torch

from ..modeling.embedder import SPEC_FILE
from ..utils import EmbedError

log = logging.getLogger("lumi_embed.merge")

COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", SPEC_FILE, "modules.json", "sentence_bert_config.json",
              "config_sentence_transformers.json")
COPY_DIRS = ("1_Pooling", "2_Normalize")


def slerp(t: float, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SLERP de um tensor (seção 4.5 do plano); cai na interpolação linear se quase colineares."""
    af, bf = a.flatten().float(), b.flatten().float()
    cos = (af @ bf) / (af.norm() * bf.norm() + eps)
    om = torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
    if om < 1e-4:
        return ((1 - t) * a.float() + t * b.float()).to(a.dtype)
    so = torch.sin(om)
    out = (torch.sin((1 - t) * om) / so) * a.float() + (torch.sin(t * om) / so) * b.float()
    return out.to(a.dtype)


class _Reader:
    """Acesso preguiçoso aos tensores de um checkpoint (arquivo único ou fragmentado)."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            wmap = json.loads(index.read_text())["weight_map"]
            files = sorted(set(wmap.values()))
        elif (self.path / "model.safetensors").exists():
            files, wmap = ["model.safetensors"], None
        else:
            raise EmbedError(f"Sem pesos safetensors em {self.path}")
        self.handles = {f: safe_open(str(self.path / f), "pt") for f in files}
        self.where = wmap or {k: files[0] for k in self.handles[files[0]].keys()}

    def keys(self) -> list[str]:
        return sorted(self.where)

    def get(self, key: str) -> torch.Tensor:
        return self.handles[self.where[key]].get_tensor(key)


def check_compatible(paths: list[Path]) -> None:
    specs = []
    for p in paths:
        f = Path(p) / SPEC_FILE
        specs.append(json.loads(f.read_text()) if f.exists() else {})
    sig = {(s.get("direction"), s.get("pooling"), s.get("eos_token")) for s in specs}
    if len(sig) > 1:
        raise EmbedError(f"Checkpoints de trilhos diferentes não podem ser mesclados: {sig}")


def merge_tensors(paths: list[Path], combine: Callable[[list[torch.Tensor]], torch.Tensor], out: Path,
                  recipe: dict, out_dtype: torch.dtype = torch.bfloat16, shard_bytes: int = 4 << 30) -> Path:
    from safetensors.torch import save_file

    paths = [Path(p) for p in paths]
    check_compatible(paths)
    readers = [_Reader(p) for p in paths]
    keys = readers[0].keys()
    for r, p in zip(readers[1:], paths[1:]):
        if r.keys() != keys:
            raise EmbedError(f"Nomes de tensor diferentes entre {paths[0]} e {p}")
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shard, size, shards, weight_map = {}, 0, [], {}

    def flush():
        nonlocal shard, size
        if not shard:
            return
        name = f"model-{len(shards) + 1:05d}.safetensors"
        save_file(shard, str(tmp / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        shards.append(name)
        shard, size = {}, 0

    for k in keys:
        ts = [r.get(k) for r in readers]
        if ts[0].is_floating_point():
            merged = combine([t.float() for t in ts]).to(out_dtype)
        else:
            merged = ts[0]
        shard[k] = merged.contiguous()
        size += merged.numel() * merged.element_size()
        if size >= shard_bytes:
            flush()
    flush()
    if len(shards) == 1:
        (tmp / shards[0]).rename(tmp / "model.safetensors")
    else:
        total = sum((tmp / s).stat().st_size for s in shards)
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    for f in COPY_FILES:
        if (paths[0] / f).exists():
            shutil.copy(paths[0] / f, tmp / f)
    for d in COPY_DIRS:
        if (paths[0] / d).exists():
            shutil.copytree(paths[0] / d, tmp / d)
    (tmp / "meta.json").write_text(json.dumps(recipe, indent=2, ensure_ascii=False, default=str) + "\n")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    log.info("Merge salvo em %s (%s)", out, recipe.get("method"))
    return out


def linear(paths: list[Path], out: Path, weights: list[float] | None = None, recipe: dict | None = None) -> Path:
    w = weights or [1.0 / len(paths)] * len(paths)
    if abs(sum(w) - 1.0) > 1e-6:
        raise EmbedError(f"Pesos do merge linear devem somar 1 (somam {sum(w)})")
    return merge_tensors(paths, lambda ts: sum(wi * t for wi, t in zip(w, ts)), out,
                         {**(recipe or {}), "method": "linear", "weights": w, "inputs": [str(p) for p in paths]})


def slerp_merge(a: Path, b: Path, t: float, out: Path, recipe: dict | None = None) -> Path:
    return merge_tensors([a, b], lambda ts: slerp(t, ts[0], ts[1]), out,
                         {**(recipe or {}), "method": "slerp", "t": t, "inputs": [str(a), str(b)]})


def base_interp(base: Path, ckpt: Path, lam: float, out: Path, recipe: dict | None = None) -> Path:
    return merge_tensors([base, ckpt], lambda ts: (1 - lam) * ts[0] + lam * ts[1], out,
                         {**(recipe or {}), "method": "base_interp", "lambda": lam,
                          "inputs": [str(base), str(ckpt)]})
