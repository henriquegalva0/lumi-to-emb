"""Leitura em streaming de parquets do Hugging Face Hub (revisão fixada, ordem determinística).

Mesma estratégia do palheiro da Etapa 1: os *row groups* são lidos sob demanda, numa ordem
embaralhada pela semente; não depende da biblioteca `datasets`, então funciona nos dois
ambientes (vLLM e treino).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np

log = logging.getLogger("lumi_embed.hub")


def list_files(repo: str, revision: str, pattern: str) -> tuple[object, list[str]]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{repo}@{revision}/{pattern}"))
    if not paths:
        raise FileNotFoundError(f"nenhum arquivo em datasets/{repo}@{revision}/{pattern}")
    return fs, paths


def iter_rows(repo: str, revision: str, pattern: str, columns: list[str] | None = None, seed: int = 0,
              shuffle: bool = True) -> Iterator[dict]:
    import pyarrow.parquet as pq

    fs, paths = list_files(repo, revision, pattern)
    files = {p: pq.ParquetFile(fs.open(p, "rb", block_size=16 * 1024 * 1024)) for p in paths}
    units = [(p, i) for p in paths for i in range(files[p].metadata.num_row_groups)]
    order = np.random.default_rng([seed, 7]).permutation(len(units)) if shuffle else range(len(units))
    log.info("%s@%s/%s: %d arquivo(s), %d row groups", repo, revision[:10], pattern, len(paths), len(units))
    for k in order:
        p, rg = units[int(k)]
        rows = files[p].read_row_group(rg, columns=columns).to_pylist()
        idx = np.random.default_rng([seed, 7, int(k)]).permutation(len(rows)) if shuffle else range(len(rows))
        for j in idx:
            yield rows[int(j)]


def download(repo: str, revision: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, revision=revision, filename=filename, repo_type="dataset")
