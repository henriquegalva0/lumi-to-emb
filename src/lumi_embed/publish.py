"""Publicação no Hugging Face Hub (repositório de modelo do projeto).

- Cada checkpoint vai para um branch com o seu ID (ex.: `C2`, `M-C`), com os pesos, o
  tokenizer, a spec, os arquivos do Sentence-Transformers e o meta.json (semente, commit,
  configuração).
- O modelo final (release) vai para o branch `main`, junto com o model card.
- O model card e a documentação (docs/huggingface/, docs/embed/, plano em PDF) podem ser
  enviados sozinhos com `card`.
O token vem da variável HF_TOKEN (arquivo .env), nunca da linha de comando.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from .utils import REPO_ROOT, EmbedError, ckpt_dir, git_commit

log = logging.getLogger("lumi_embed.publish")


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EmbedError("HF_TOKEN ausente: coloque-o no arquivo .env do repositório.")
    return HfApi(token=token)


def card_files() -> dict[str, Path]:
    """Arquivos do model card e da documentação enviados ao repositório do modelo."""
    files = {"README.md": REPO_ROOT / "docs" / "huggingface" / "README.md"}
    for p in sorted((REPO_ROOT / "docs" / "embed").glob("*")):
        if p.is_file():
            files[f"docs/{p.name}"] = p
    for p in sorted((REPO_ROOT / "docs" / "stage1").glob("*")):
        if p.is_file() and p.suffix in (".md", ".png", ".csv", ".json"):
            files[f"docs/stage1/{p.name}"] = p
    return {k: v for k, v in files.items() if v.exists()}


def push_card(repo: str, message: str | None = None) -> str:
    api = _api()
    files = card_files()
    if "README.md" not in files:
        raise EmbedError("Model card ausente: docs/huggingface/README.md")
    with tempfile.TemporaryDirectory() as tmp:
        for name, src in files.items():
            dst = Path(tmp) / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        info = api.upload_folder(repo_id=repo, folder_path=tmp, repo_type="model", revision="main",
                                 commit_message=message or f"Model card e documentação (código {git_commit()['commit'][:7]})")
    log.info("Model card enviado: %s", info)
    return str(info)


def publish_checkpoint(repo: str, ckpt_id: str, release: bool = False) -> str:
    api = _api()
    folder = ckpt_dir(ckpt_id) / "model"
    if not (folder / "config.json").exists():
        raise EmbedError(f"Checkpoint {ckpt_id} não encontrado em {folder}")
    revision = "main" if release else ckpt_id
    if not release:
        api.create_branch(repo_id=repo, branch=ckpt_id, repo_type="model", exist_ok=True)
    msg = f"{'Release' if release else 'Checkpoint'} {ckpt_id} (código {git_commit()['commit'][:7]})"
    info = api.upload_folder(repo_id=repo, folder_path=str(folder), repo_type="model", revision=revision,
                             commit_message=msg, ignore_patterns=["*.tmp", "train_state/*"])
    if release:
        push_card(repo, f"Model card da release {ckpt_id}")
    log.info("%s publicado em %s@%s", ckpt_id, repo, revision)
    return str(info)
