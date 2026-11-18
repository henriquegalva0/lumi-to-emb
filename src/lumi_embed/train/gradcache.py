"""GradCache: batch efetivo grande com memória de um micro-lote.

1. Forward sem gradiente de todos os textos, em micro-lotes → embeddings.
2. Perda sobre os embeddings (folhas com requires_grad) → gradiente de cada embedding.
3. Para cada micro-lote: forward com gradiente e backward de <emb, grad_emb>.

O resultado é idêntico ao backward sobre o batch inteiro (o modelo não tem dropout),
e é o que permite o batch efetivo de 2048 do estágio S1 numa GPU.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def encode_chunks(model, enc: dict, chunk: int, grad: bool) -> torch.Tensor:
    outs = []
    n = enc["input_ids"].shape[0]
    with torch.set_grad_enabled(grad):
        for i in range(0, n, chunk):
            ids = enc["input_ids"][i : i + chunk]
            mask = enc["attention_mask"][i : i + chunk]
            width = int(mask.sum(1).max())  # corta colunas só de padding (padding à direita)
            outs.append(model(ids[:, :width], mask[:, :width]))
    return torch.cat(outs, dim=0)


def gradcache_step(
    model,
    encs: list[dict],
    loss_fn: Callable[[list[torch.Tensor]], torch.Tensor],
    chunk: int,
    scale: float = 1.0,
) -> float:
    """Faz forward/backward de um passo com GradCache. `encs` = [queries, positivos, negativos?]."""
    with torch.no_grad():
        reps = [encode_chunks(model, e, chunk, grad=False) if e is not None else None for e in encs]
    leaves = [r.detach().requires_grad_(True) if r is not None else None for r in reps]
    loss = loss_fn(leaves) * scale
    loss.backward()
    for e, leaf in zip(encs, leaves):
        if e is None:
            continue
        g = leaf.grad
        n = e["input_ids"].shape[0]
        for i in range(0, n, chunk):
            ids = e["input_ids"][i : i + chunk]
            mask = e["attention_mask"][i : i + chunk]
            width = int(mask.sum(1).max())
            emb = model(ids[:, :width], mask[:, :width])
            torch.autograd.backward(emb, grad_tensors=g[i : i + chunk].to(emb.dtype))
    return float(loss.detach()) / scale
