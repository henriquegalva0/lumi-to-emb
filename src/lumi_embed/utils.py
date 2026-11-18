"""Configuração, caminhos e metadados comuns às etapas do Lumi-Embed."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import yaml

from lumi_ctx.common import (  # noqa: F401  (reexportados)
    REPO_ROOT,
    Stage1Error,
    deep_merge,
    git_commit,
    read_json,
    read_jsonl,
    sha256_text,
    setup_logging,
    stable_hash,
    write_json,
    write_jsonl,
    write_yaml,
)

EMBED_CONFIG_DIR = REPO_ROOT / "configs" / "embed"


class EmbedError(Stage1Error):
    """Erro com mensagem pensada para o log (sem precisar do traceback)."""


def load_config(path: str | Path) -> dict:
    """YAML com `inherit:` (relativo ao arquivo; aceita lista). Resolve nomes curtos em configs/embed/."""
    p = Path(path)
    if not p.exists():
        for cand in (EMBED_CONFIG_DIR / p, EMBED_CONFIG_DIR / "train" / p, EMBED_CONFIG_DIR / "merge" / p,
                     EMBED_CONFIG_DIR / "train" / f"{p}.yaml", EMBED_CONFIG_DIR / "merge" / f"{p}.yaml",
                     EMBED_CONFIG_DIR / f"{p}.yaml"):
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise EmbedError(f"Configuração não encontrada: {path}")
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parents = cfg.pop("inherit", None)
    if parents:
        merged: dict = {}
        for parent in parents if isinstance(parents, list) else [parents]:
            merged = deep_merge(merged, load_config(p.parent / parent))
        cfg = deep_merge(merged, cfg)
    cfg.setdefault("_meta", {})["config_file"] = str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p)
    return cfg


def workdir() -> Path:
    return Path(os.environ.get("LUMI_WORKDIR", REPO_ROOT)).resolve()


def wpath(*parts: str | Path) -> Path:
    """Caminho dentro do diretório de trabalho (aceita absolutos)."""
    p = Path(*parts)
    return p if p.is_absolute() else workdir() / p


def data_dir(*parts) -> Path:
    return wpath("data", "embed", *parts)


def ckpt_dir(ckpt_id: str) -> Path:
    return wpath("checkpoints", ckpt_id)


def results_dir(*parts) -> Path:
    return wpath("results", "embed", *parts)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_metadata(cfg: dict | None = None, **extra: Any) -> dict:
    """Semente, commit e configuração registrados em cada artefato (regra A.3 do plano)."""
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_commit(),
        "lumi_embed_version": __import__("lumi_embed").__version__,
    }
    if cfg is not None:
        meta["config"] = copy.deepcopy(cfg)
    meta.update(extra)
    return meta


def iter_jsonl(path: Path, limit: int | None = None):
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


def append_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def dump_yaml(obj) -> str:
    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
