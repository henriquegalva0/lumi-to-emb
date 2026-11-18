"""Adaptador do LumiEmbedder para o mteb 2.x (protocolo da seção 5.1 do plano).

- Instrução só na query (PromptType.query); documentos sem instrução (PromptType.document);
  em tarefas simétricas (STS, classificação, clustering, pares: prompt_type None), a mesma
  instrução nos dois lados.
- Instruções por tarefa e por tipo de tarefa vêm de configs/embed/eval.yaml e são registradas
  (results/embed/instructions/<modelo>.json) para cada tarefa avaliada.
"""

from __future__ import annotations

import json
import logging

import torch

from ..modeling.embedder import EmbedSpec, LumiEmbedder
from ..utils import results_dir

log = logging.getLogger("lumi_embed.mteb")


def _mteb_imports():
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.models.model_meta import ModelMeta, ScoringFunction
    from mteb.types import PromptType

    return AbsEncoder, ModelMeta, ScoringFunction, PromptType


def resolve_instruction(instructions: dict, task_name: str, task_type: str, prompt_type) -> str | None:
    _, _, _, PromptType = _mteb_imports()
    if prompt_type == PromptType.document:
        return None
    by_task = instructions.get("by_task", {})
    if task_name in by_task:
        return by_task[task_name]
    return instructions.get("by_type", {}).get(task_type) or instructions.get("default")


def build_encoder(model_path: str, name: str, revision: str, instructions: dict, batch_size: int = 32,
                  max_length: int | None = None, spec_overrides: dict | None = None):
    AbsEncoder, ModelMeta, ScoringFunction, _ = _mteb_imports()

    class LumiMTEBEncoder(AbsEncoder):
        def __init__(self):
            self.model = LumiEmbedder.from_pretrained(model_path, spec=spec_overrides or {},
                                                      device="cuda" if torch.cuda.is_available() else "cpu")
            self.model.eval()
            self.batch_size = batch_size
            self.max_length = max_length or self.model.spec.max_length
            self.used: dict[str, dict] = {}
            n_params = sum(p.numel() for p in self.model.backbone.parameters())
            self.mteb_model_meta = ModelMeta(
                loader=None, name=name, revision=revision, release_date=None,
                languages=["por-Latn", "eng-Latn"], n_parameters=n_params, memory_usage_mb=None,
                max_tokens=self.max_length, embed_dim=self.model.hidden_size, license=None, open_weights=False,
                public_training_code="https://github.com/henriquegalva0/lumi-to-emb", public_training_data=None,
                framework=["PyTorch"], similarity_fn_name=ScoringFunction.COSINE, use_instructions=True,
                training_datasets=None, adapted_from="ceia-research/lumi-4b",
            )

        def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
            texts = [t for batch in inputs for t in batch["text"]]
            inst = resolve_instruction(instructions, task_metadata.name, task_metadata.type, prompt_type)
            key = getattr(prompt_type, "value", None) or "symmetric"
            self.used.setdefault(task_metadata.name, {})[key] = inst
            emb = self.model.encode(texts, instruction=inst, batch_size=self.batch_size, max_length=self.max_length)
            return emb.numpy()

        def save_instructions(self) -> None:
            path = results_dir("instructions", name.replace("/", "__") + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            old = json.loads(path.read_text()) if path.exists() else {}
            old.update(self.used)
            path.write_text(json.dumps(old, indent=2, ensure_ascii=False))

    return LumiMTEBEncoder()


def spec_override_for(ckpt_id: str) -> dict:
    """R0e = R0 com echo na inferência (A7)."""
    return {"echo": True} if ckpt_id.endswith("e") and ckpt_id.startswith("R0") else {}


def spec_from(path: str) -> EmbedSpec:
    return EmbedSpec.from_dict(json.loads((__import__("pathlib").Path(path) / "lumi_embed.json").read_text()))
