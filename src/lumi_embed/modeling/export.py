"""Salvamento de um embedder no formato Hugging Face + Sentence-Transformers.

Estrutura de um checkpoint exportado:
    config.json, model.safetensors   backbone NemotronHModel (transformers nativo, sem código remoto)
    tokenizer.json, ...              tokenizer com EOS anexado e padding à direita
    lumi_embed.json                  EmbedSpec (direção, pooling, EOS, template de instrução)
    modules.json, 1_Pooling/, ...    Sentence-Transformers (só trilho C: causal + último token)
    meta.json                        semente, commit, configuração (regra A.3 do plano)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .embedder import SPEC_FILE, EmbedSpec, LumiEmbedder

log = logging.getLogger("lumi_embed.export")

DEFAULT_QUERY_INSTRUCTION = "Dada uma pergunta, recupere passagens que a respondam"


def _clean_config(out_dir: Path) -> None:
    """Remove `auto_map` (código remoto) para o checkpoint carregar com o transformers nativo."""
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("auto_map", None)
    cfg["architectures"] = ["NemotronHModel"]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")


def write_sentence_transformers_files(out_dir: Path, spec: EmbedSpec, hidden_size: int,
                                      query_instruction: str = DEFAULT_QUERY_INSTRUCTION) -> None:
    """Arquivos do Sentence-Transformers (Transformer → Pooling → Normalize). Só para o trilho C."""
    pooling_mode = {"eos": "lasttoken", "mean": "mean"}[spec.pooling]
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ]
    if spec.normalize:
        modules.append({"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"})
    (out_dir / "modules.json").write_text(json.dumps(modules, indent=2) + "\n")
    (out_dir / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": spec.max_length, "do_lower_case": False}, indent=2) + "\n"
    )
    pool_dir = out_dir / "1_Pooling"
    pool_dir.mkdir(exist_ok=True)
    pool_cfg = {
        "word_embedding_dimension": hidden_size,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": pooling_mode == "mean",
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": pooling_mode == "lasttoken",
        "include_prompt": True,
    }
    (pool_dir / "config.json").write_text(json.dumps(pool_cfg, indent=2) + "\n")
    if spec.normalize:
        (out_dir / "2_Normalize").mkdir(exist_ok=True)
    query_prompt = spec.instruction_template.split("{query}")[0].format(instruction=query_instruction)
    st_cfg = {
        "prompts": {"query": query_prompt, "document": ""},
        "default_prompt_name": None,
        "similarity_fn_name": "cosine",
        "model_type": "SentenceTransformer",
    }
    (out_dir / "config_sentence_transformers.json").write_text(json.dumps(st_cfg, indent=2, ensure_ascii=False) + "\n")


def save_embedder(model: LumiEmbedder, out_dir: Path, meta: dict | None = None,
                  dtype: torch.dtype = torch.bfloat16) -> Path:
    """Salva backbone (com LoRA já fundido), tokenizer, spec e arquivos do ST."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = model.backbone
    if hasattr(backbone, "merge_and_unload"):  # PeftModel: o chamador deve fundir antes
        raise ValueError("Funda o LoRA (merge_and_unload) antes de salvar o embedder completo.")
    # Converte uma CÓPIA do state dict: converter o módulo in place estragaria pesos mestres fp32.
    state = {k: v.detach().to(dtype) if v.is_floating_point() else v for k, v in backbone.state_dict().items()}
    backbone.save_pretrained(out_dir, state_dict=state, safe_serialization=True)
    del state
    _clean_config(out_dir)
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["dtype"] = str(dtype).replace("torch.", "")
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    model.tokenizer.save_pretrained(out_dir)
    (out_dir / SPEC_FILE).write_text(json.dumps(model.spec.to_dict(), indent=2, ensure_ascii=False) + "\n")
    if model.spec.direction == "C":
        write_sentence_transformers_files(out_dir, model.spec, model.hidden_size)
    else:
        log.info("Trilho %s: sem arquivos do Sentence-Transformers (carregar com lumi_embed).", model.spec.direction)
    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n")
    return out_dir
