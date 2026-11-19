"""Kernels Mamba-2 do Hub: ativação explícita no dispositivo final e prova de que estão em uso.

Por que não basta `from_pretrained(..., use_kernels=True)`:
- o transformers escolhe os kernels pelo dispositivo e pelo modo (treino/inferência) do modelo no
  momento do carregamento, e o modelo ainda muda de dispositivo e de modo depois;
- o `kernelize` usa fallback silencioso: um kernel que não carrega vira o caminho PyTorch puro, que
  é correto mas lento e, no treino, consome memória proporcional a lote × cabeças × bloco², o
  suficiente para estourar uma GPU de 178 GB com 32 × 512 tokens.

Aqui os kernels são ativados depois de o modelo estar na GPU, em modo de treino (que também serve à
inferência), e uma passada curta em cada modo conta as chamadas do scan em PyTorch puro
(`segment_sum`, usado só por esse caminho). Se ele aparecer, o carregamento falha com uma mensagem
clara; `LUMI_ALLOW_TORCH_MAMBA=1` permite seguir assim mesmo (por exemplo, para comparação).
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

from ..utils import EmbedError

log = logging.getLogger("lumi_embed.kernels")

ALLOW_ENV = "LUMI_ALLOW_TORCH_MAMBA"


def enable_hub_kernels(model: nn.Module) -> None:
    """Kernelize no dispositivo atual do modelo, em modo de treino (os kernels servem aos dois modos)."""
    from kernels import Mode

    model.set_use_kernels(True, mode=Mode.TRAINING)


def probe_mamba_path(backbone: nn.Module, seq_len: int = 64) -> dict:
    """Quantas vezes o scan Mamba-2 em PyTorch puro roda numa passada de inferência e numa de treino."""
    import transformers.models.nemotron_h.modeling_nemotron_h as mh

    calls = {"n": 0}
    original = mh.segment_sum

    def counting(x):
        calls["n"] += 1
        return original(x)

    device = next(backbone.parameters()).device
    vocab = int(getattr(backbone.config, "vocab_size", 1000))
    ids = torch.randint(min(10, vocab - 1), min(1000, vocab), (2, seq_len), device=device)
    mask = torch.ones_like(ids)
    was_training = backbone.training
    mh.segment_sum = counting
    try:
        with torch.no_grad():
            backbone.eval()
            backbone(input_ids=ids, attention_mask=mask, use_cache=False)
            n_eval = calls["n"]
            backbone.train()  # o mixer só usa o kernel fundido em modo de treino
            backbone(input_ids=ids, attention_mask=mask, use_cache=False)
            n_train = calls["n"] - n_eval
    finally:
        mh.segment_sum = original
        backbone.train(was_training)
    return {"torch_scan_calls_eval": n_eval, "torch_scan_calls_train": n_train}


def ensure_hub_kernels(model: nn.Module, backbone: nn.Module | None = None) -> dict:
    """Ativa os kernels do Hub e verifica que o Mamba-2 não usa o caminho PyTorch puro.

    `model` é o PreTrainedModel a kernelizar; `backbone`, o NemotronHModel usado na verificação
    (por padrão, o próprio `model`). Devolve o resultado da verificação, para os metadados.
    """
    backbone = backbone if backbone is not None else model
    problem = None
    try:
        enable_hub_kernels(model)
    except Exception as exc:  # noqa: BLE001 - registrado e tratado abaixo
        problem = f"falha ao ativar os kernels: {type(exc).__name__}: {exc}"
    probe = probe_mamba_path(backbone)
    active = problem is None and probe["torch_scan_calls_eval"] == 0 and probe["torch_scan_calls_train"] == 0
    result = {"active": active, **probe}
    if problem:
        result["problem"] = problem
    if active:
        log.info("Kernels Mamba-2 do Hub ativos (scan em PyTorch puro: 0 chamadas em inferência e em treino)")
        return result
    msg = (problem or "o scan em PyTorch puro ainda é usado") + (
        f" (chamadas em inferência: {probe['torch_scan_calls_eval']}, em treino: {probe['torch_scan_calls_train']}).")
    if os.environ.get(ALLOW_ENV) == "1":
        log.warning("Kernels Mamba-2 do Hub INATIVOS: %s Seguindo no PyTorch puro (%s=1).", msg, ALLOW_ENV)
        return result
    raise EmbedError(
        f"Kernels Mamba-2 do Hub inativos: {msg} Sem eles, o treino fica lento e pode estourar a memória. "
        f"Veja o log acima (download/validação de kernels-community/mamba-ssm). Para seguir mesmo assim, "
        f"defina {ALLOW_ENV}=1."
    )
