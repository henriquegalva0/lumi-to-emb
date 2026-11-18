"""Utilitários compartilhados: configuração, caminhos, logging, I/O e metadados."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

TASK_CODES = {"C0": 0, "S": 1, "MK": 2, "MV": 3}
TASKS = ("C0", "S", "MK", "MV")

log = logging.getLogger("lumi_ctx")


class Stage1Error(RuntimeError):
    """Erro com mensagem pensada para quem vai ler o log, sem precisar do traceback."""


# --------------------------------------------------------------------------- config


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """Lê um YAML de configuração, resolvendo `inherit:` (relativo ao próprio arquivo)."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        candidate = CONFIG_DIR / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise Stage1Error(f"Arquivo de configuração não encontrado: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = deep_merge(base, cfg)
    cfg.setdefault("_meta", {})["config_file"] = str(path.name)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    exp = cfg.get("experiment", {})
    missing = [k for k in ("seed", "lengths", "tasks", "length_tolerance") if k not in exp]
    if missing:
        raise Stage1Error(f"Configuração incompleta: faltam experiment.{missing}")
    lengths = exp["lengths"]
    if sorted(lengths) != list(lengths) or len(set(lengths)) != len(lengths):
        raise Stage1Error("experiment.lengths precisa ser crescente e sem repetições")
    max_len = cfg.get("vllm", {}).get("engine", {}).get("max_model_len")
    max_out = max(cfg["generation"]["max_tokens"], cfg["generation"]["max_tokens_mv"])
    if max_len and lengths[-1] + max_out > max_len:
        raise Stage1Error(
            f"O maior comprimento ({lengths[-1]}) + max_tokens ({max_out}) passa de "
            f"max_model_len ({max_len})."
        )
    for task in exp["tasks"]:
        if task not in TASK_CODES:
            raise Stage1Error(f"Tarefa desconhecida em experiment.tasks: {task}")
    keys = [m["key"] for m in cfg.get("models", [])]
    if not keys or len(set(keys)) != len(keys):
        raise Stage1Error("models: é preciso ao menos um modelo, com chaves únicas")


# --------------------------------------------------------------------------- caminhos


def workdir() -> Path:
    """Diretório onde tudo é gravado (resultados, logs, corpus, prompts).

    Padrão: a raiz do repositório (que, no servidor, é a pasta de trabalho do job).
    Pode ser trocado com a variável LUMI_WORKDIR.
    """
    return Path(os.environ.get("LUMI_WORKDIR", REPO_ROOT)).resolve()


def in_workdir(relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else workdir() / p


def results_dir(run_id: str) -> Path:
    return workdir() / "results" / run_id


def work_dir(run_id: str) -> Path:
    return workdir() / "work" / run_id


def new_run_id(prefix: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{prefix}"


# --------------------------------------------------------------------------- logging


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- I/O


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, allow_unicode=True, sort_keys=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))[:16]


# --------------------------------------------------------------------------- git


def git_commit(repo: Path = REPO_ROOT) -> dict:
    """Commit atual lido direto de .git (a imagem do vLLM não traz o binário git)."""
    info = {"commit": "desconhecido", "branch": None}
    git = repo / ".git"
    try:
        if git.is_file():  # worktree/submódulo: "gitdir: <caminho>"
            git = (repo / git.read_text().split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            info["branch"] = ref.rsplit("/", 1)[-1]
            ref_file = git / ref
            if ref_file.exists():
                info["commit"] = ref_file.read_text().strip()
            elif (git / "packed-refs").exists():
                for line in (git / "packed-refs").read_text().splitlines():
                    if line.endswith(" " + ref):
                        info["commit"] = line.split(" ", 1)[0]
                        break
            else:
                info["commit"] = "sem commits"
        else:
            info["commit"] = head
    except OSError:
        pass
    return info
