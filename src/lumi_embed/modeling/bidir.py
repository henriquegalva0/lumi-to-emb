"""Variantes bidirecionais do backbone NemotronH (trilho B, seção 3.5 do plano).

- Ba: só as 4 camadas de atenção ficam bidirecionais (máscara apenas de padding);
  as 21 camadas Mamba continuam causais.
- Bf: além disso, cada mixer Mamba-2 soma a varredura direta com uma varredura na
  ordem reversa, com os MESMOS pesos (sem parâmetros novos), e tira a média.
  A inversão é feita só sobre os tokens válidos de cada exemplo (padding à direita
  continua à direita), e a convolução causal curta roda na versão invertida porque
  ela faz parte do próprio mixer.

As modificações são feitas por *monkeypatch* do método `forward`, sem trocar módulos:
os nomes dos pesos (e, portanto, checkpoints, LoRA e merges) ficam idênticos aos do
modelo causal.
"""

from __future__ import annotations

import types

import torch
from torch import nn

DIRECTIONS = ("C", "Ba", "Bf")


def attention_modules(backbone: nn.Module) -> list[nn.Module]:
    return [layer.mixer for layer in backbone.layers if getattr(layer, "block_type", "") == "full_attention"]


def mamba_mixers(backbone: nn.Module) -> list[nn.Module]:
    return [layer.mixer for layer in backbone.layers if getattr(layer, "block_type", "") == "linear_attention"]


def reverse_index(attention_mask: torch.Tensor | None, batch: int, seq_len: int, device) -> torch.Tensor:
    """Índices que invertem só a parte válida de cada linha (padding à direita fica onde está).

    rev[b, t] = len_b - 1 - t  se t < len_b;  t  caso contrário. É uma involução.
    """
    t = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
    if attention_mask is None:
        lengths = torch.full((batch, 1), seq_len, device=device, dtype=torch.long)
    else:
        lengths = attention_mask.to(torch.long).sum(dim=1, keepdim=True)
    return torch.where(t < lengths, lengths - 1 - t, t)


def _bimamba_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
        # decodificação incremental (geração) não tem leitura reversa possível
        return self._lumi_unidirectional_forward(hidden_states, cache_params=cache_params,
                                                 attention_mask=attention_mask, **kwargs)
    fwd = self._lumi_unidirectional_forward(hidden_states, attention_mask=attention_mask, **kwargs)
    b, n, h = hidden_states.shape
    rev = reverse_index(attention_mask, b, n, hidden_states.device)
    gather = rev.unsqueeze(-1).expand(b, n, h)
    hs_rev = torch.gather(hidden_states, 1, gather)
    bwd = self._lumi_unidirectional_forward(hs_rev, attention_mask=attention_mask, **kwargs)
    bwd = torch.gather(bwd, 1, rev.unsqueeze(-1).expand(b, n, bwd.shape[-1]))
    return (fwd + bwd) / 2


def apply_direction(backbone: nn.Module, direction: str) -> None:
    """Aplica a direcionalidade ao backbone (idempotente)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direção desconhecida: {direction} (use {DIRECTIONS})")
    bidir_attention = direction in ("Ba", "Bf")
    for attn in attention_modules(backbone):
        # Necessário além da máscara: sem padding, a máscara bidirecional vira None e o SDPA
        # decidiria a causalidade só por `is_causal`.
        attn.is_causal = not bidir_attention
    for mixer in mamba_mixers(backbone):
        if not hasattr(mixer, "_lumi_unidirectional_forward"):
            mixer._lumi_unidirectional_forward = mixer.forward
        if direction == "Bf":
            mixer.forward = types.MethodType(_bimamba_forward, mixer)
        else:
            mixer.forward = mixer._lumi_unidirectional_forward
    backbone._lumi_direction = direction


def build_masks(backbone: nn.Module, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
    """Máscaras por tipo de camada, no formato aceito por NemotronHModel.forward (dict)."""
    from transformers.masking_utils import (
        create_bidirectional_mask,
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    cfg = backbone.config
    direction = getattr(backbone, "_lumi_direction", "C")
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    kw = {"config": cfg, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
    if direction == "C":
        full = create_causal_mask(**kw, past_key_values=None, position_ids=position_ids)
    else:
        full = create_bidirectional_mask(**kw)
    return {"full_attention": full, "linear_attention": create_recurrent_attention_mask(**kw)}
