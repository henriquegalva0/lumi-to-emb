"""Perdas contrastivas (seção 1.1 e eixo "Loss" da Tabela 4 do plano).

- `hn`: InfoNCE só com os negativos difíceis de cada query (melhor variante em [10]).
        O tamanho do batch não muda a perda.
- `inbatch`: InfoNCE com negativos difíceis + todos os documentos do batch, com a
        máscara de falso negativo do Qwen3-Embedding [9]:
        m_ij = 0 se s(q_i, d_j) > s(q_i, d_i+) + margem, ou se d_j é o próprio positivo.
        Opcional: termos query–query e documento–documento no denominador.
- Matryoshka (MRL, A12): média da perda em várias dimensões truncadas e renormalizadas.

Embeddings chegam normalizados (L2), então s = cosseno.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

NEG_INF = -1e4  # finito: evita NaN no softmax quando uma linha fica toda mascarada


@dataclass
class LossConfig:
    mode: str = "hn"                 # hn | inbatch
    temperature: float = 0.02
    fn_margin: float = 0.1           # máscara de falso negativo (inbatch)
    qq_dd: bool = False              # termos q–q e d–d no denominador (inbatch)
    mrl_dims: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LossConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def _hn_logits(q, p, n, n_owner):
    """[B, 1+Kmax]: coluna 0 = positivo; negativos de cada query; o resto mascarado."""
    b = q.shape[0]
    pos = (q * p).sum(-1, keepdim=True)
    if n is None or n.shape[0] == 0:
        return pos
    counts = torch.bincount(n_owner, minlength=b)
    kmax = int(counts.max().item())
    slot = torch.zeros_like(n_owner)
    # posição de cada negativo dentro da sua query (0..k-1), respeitando a ordem de entrada
    order = torch.argsort(n_owner, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    slot[order] = torch.arange(len(n_owner), device=q.device) - starts[n_owner[order]]
    neg = torch.full((b, kmax), NEG_INF, device=q.device, dtype=q.dtype)
    neg[n_owner, slot] = (q[n_owner] * n).sum(-1)
    return torch.cat([pos, neg], dim=1)


def _inbatch_logits(q, p, n, n_owner, pos_ids, neg_ids, cfg: LossConfig):
    """[B, B + M (+ B + B)]: todos os documentos do batch; alvo de i = coluna i."""
    b = q.shape[0]
    docs = p if n is None or n.shape[0] == 0 else torch.cat([p, n], dim=0)
    doc_ids = pos_ids if n is None or n.shape[0] == 0 else torch.cat([pos_ids, neg_ids])
    s = q @ docs.T
    s_pos = s[torch.arange(b), torch.arange(b)].unsqueeze(1)
    target = torch.arange(b, device=q.device)
    mask = torch.ones_like(s, dtype=torch.bool)
    with torch.no_grad():
        too_close = s > s_pos + cfg.fn_margin                     # provável falso negativo
        same_doc = doc_ids.unsqueeze(0) == pos_ids.unsqueeze(1)   # cópia do próprio positivo
        mask &= ~(too_close | same_doc)
        mask[target, target] = True
    s = s.masked_fill(~mask, NEG_INF)
    if cfg.qq_dd:
        qq = (q @ q.T).masked_fill(torch.eye(b, dtype=torch.bool, device=q.device), NEG_INF)
        qq = qq.masked_fill(qq > s_pos + cfg.fn_margin, NEG_INF)
        dd = (p @ p.T).masked_fill(torch.eye(b, dtype=torch.bool, device=q.device), NEG_INF)
        dd = dd.masked_fill(dd > s_pos + cfg.fn_margin, NEG_INF)
        s = torch.cat([s, qq, dd], dim=1)
    return s, target


def info_nce(q, p, n, n_owner, cfg: LossConfig, pos_ids=None, neg_ids=None) -> torch.Tensor:
    """q, p: [B, D]; n: [M, D] (negativos achatados) com n_owner: [M] (índice da query)."""
    tau = cfg.temperature
    if cfg.mode == "hn":
        logits = _hn_logits(q, p, n, n_owner) / tau
        target = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    elif cfg.mode == "inbatch":
        if pos_ids is None:
            pos_ids = torch.arange(q.shape[0], device=q.device)
        if neg_ids is None and n is not None:
            neg_ids = -1 - torch.arange(n.shape[0], device=q.device)
        logits, target = _inbatch_logits(q, p, n, n_owner, pos_ids, neg_ids, cfg)
        logits = logits / tau
    else:
        raise ValueError(f"modo de perda desconhecido: {cfg.mode}")
    return F.cross_entropy(logits.float(), target)


def contrastive_loss(q, p, n, n_owner, cfg: LossConfig, pos_ids=None, neg_ids=None) -> torch.Tensor:
    """Perda completa, com Matryoshka opcional (embeddings na dimensão cheia, normalizados)."""
    loss = info_nce(q, p, n, n_owner, cfg, pos_ids, neg_ids)
    dims = [d for d in cfg.mrl_dims if d < q.shape[-1]]
    if not dims:
        return loss
    total = loss
    for d in dims:
        def cut(x):
            return None if x is None else F.normalize(x[..., :d], dim=-1)

        total = total + info_nce(cut(q), cut(p), cut(n), n_owner, cfg, pos_ids, neg_ids)
    return total / (1 + len(dims))
