"""Embedder sobre o backbone NemotronH do Lumi-4B (seção 1.1 do plano).

e(x) = H_T^(L) / ||H_T^(L)||  com T = posição do token EOS (trilho C), ou média dos
estados válidos (trilho B). O EOS é anexado pelo pós-processador do tokenizer, então
o mesmo formato vale no treino, na avaliação, no Transformers e no Sentence-Transformers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .bidir import apply_direction, build_masks
from .kernels import ensure_hub_kernels

log = logging.getLogger("lumi_embed.model")

SPEC_FILE = "lumi_embed.json"
DEFAULT_TEMPLATE = "Instruct: {instruction}\nQuery: {query}"


@dataclass
class EmbedSpec:
    direction: str = "C"          # C | Ba | Bf
    pooling: str = "eos"          # eos | mean
    eos_token: str = "</s>"
    pad_token: str = "<unk>"
    normalize: bool = True
    max_length: int = 512
    instruction_template: str = DEFAULT_TEMPLATE
    echo: bool = False            # repete o texto (echo embeddings, R0e / A7)
    use_instructions: bool = True  # False = ablação A8 (sem instrução na query)
    mrl_dims: list[int] = field(default_factory=list)  # dimensões Matryoshka treinadas (A12)

    @classmethod
    def from_dict(cls, d: dict | None) -> "EmbedSpec":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


def prepare_tokenizer(tok, spec: EmbedSpec):
    """Anexa o EOS em todo texto (pós-processador), padding à direita e token de padding."""
    from tokenizers.processors import TemplateProcessing

    eos_id = tok.convert_tokens_to_ids(spec.eos_token)
    if eos_id is None or eos_id == tok.unk_token_id and spec.eos_token != tok.unk_token:
        raise ValueError(f"token EOS {spec.eos_token!r} não existe no vocabulário")
    tok.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"$A {spec.eos_token}",
        pair=f"$A {spec.eos_token} $B {spec.eos_token}",
        special_tokens=[(spec.eos_token, eos_id)],
    )
    if tok.pad_token is None:
        tok.pad_token = spec.pad_token
    tok.padding_side = "right"
    tok.truncation_side = "right"
    return tok


def format_query(query: str, instruction: str | None, spec: EmbedSpec) -> str:
    if not instruction or not spec.use_instructions:
        return query
    return spec.instruction_template.format(instruction=instruction, query=query)


def apply_echo(text: str) -> str:
    return f"{text}\n{text}"


class LumiEmbedder(nn.Module):
    def __init__(self, backbone: nn.Module, tokenizer, spec: EmbedSpec):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.spec = spec
        self.kernel_info: dict | None = None  # resultado de ensure_hub_kernels (metadados)
        apply_direction(self.backbone, spec.direction)

    # ------------------------------------------------------------------ carga
    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str,
        *,
        revision: str | None = None,
        spec: EmbedSpec | dict | None = None,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        use_hub_kernels: bool = True,
        device: str | torch.device | None = None,
    ) -> "LumiEmbedder":
        from transformers import AutoModel, AutoTokenizer

        saved_spec = None
        local = Path(path_or_id)
        if (local / SPEC_FILE).exists():
            saved_spec = json.loads((local / SPEC_FILE).read_text())
        if isinstance(spec, dict) or spec is None:
            spec = EmbedSpec.from_dict({**(saved_spec or {}), **(spec or {})})
        backbone = AutoModel.from_pretrained(path_or_id, revision=revision, dtype=dtype,
                                             attn_implementation=attn_implementation)
        if device is not None:
            backbone.to(device)
        # Kernels do Hub só depois de o modelo estar no dispositivo final, com verificação explícita
        # (ver modeling/kernels.py): sem eles, falha em vez de cair em silêncio no PyTorch puro.
        kernel_info = None
        if use_hub_kernels and next(backbone.parameters()).device.type == "cuda":
            kernel_info = ensure_hub_kernels(backbone)
        tok = AutoTokenizer.from_pretrained(path_or_id, revision=revision)
        prepare_tokenizer(tok, spec)
        model = cls(backbone, tok, spec)
        model.kernel_info = kernel_info
        return model

    # ------------------------------------------------------------------ propriedades
    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def hidden_size(self) -> int:
        return self.backbone.config.hidden_size

    # ------------------------------------------------------------------ forward
    def tokenize(self, texts: list[str], max_length: int | None = None) -> dict:
        if self.spec.echo:
            texts = [apply_echo(t) for t in texts]
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length or self.spec.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kw):
        # use_cache=False: o config do Lumi liga o cache (geração); num embedder ele só gasta memória
        # e, no trilho Bf, um cache presente desligaria a varredura reversa.
        kw.setdefault("use_cache", False)
        if self.spec.direction == "C":
            return self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kw)
        embeds = self.backbone.get_input_embeddings()(input_ids)
        masks = build_masks(self.backbone, embeds, attention_mask)
        return self.backbone(inputs_embeds=embeds, attention_mask=masks, **kw)

    def pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.spec.pooling == "eos":
            last = attention_mask.to(torch.long).sum(dim=1) - 1  # padding à direita
            return h[torch.arange(h.shape[0], device=h.device), last]
        if self.spec.pooling == "mean":
            m = attention_mask.to(h.dtype).unsqueeze(-1)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        raise ValueError(f"pooling desconhecido: {self.spec.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.hidden_states(input_ids, attention_mask)
        e = self.pool(out.last_hidden_state, attention_mask).float()
        return F.normalize(e, dim=-1) if self.spec.normalize else e

    # ------------------------------------------------------------------ inferência
    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        *,
        instruction: str | None = None,
        batch_size: int = 32,
        max_length: int | None = None,
        dim: int | None = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Embeddings (float32, CPU) na ordem de entrada. Ordena por tamanho para reduzir padding."""
        was_training = self.training
        self.eval()
        inputs = [format_query(t, instruction, self.spec) for t in texts]
        order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i]))
        out = torch.empty((len(inputs), self.hidden_size), dtype=torch.float32)
        rng = range(0, len(order), batch_size)
        if show_progress:
            from tqdm import tqdm

            rng = tqdm(rng, desc="encode", unit="lote")
        dev = self.device
        for b0 in rng:
            idx = order[b0 : b0 + batch_size]
            enc = self.tokenize([inputs[i] for i in idx], max_length)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                e = self(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            out[idx] = e.float().cpu()
        if dim is not None:
            out = F.normalize(out[:, :dim], dim=-1)
        if was_training:
            self.train()
        return out
