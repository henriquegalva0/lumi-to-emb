"""Estatística dos resultados (seção 5.1): IC por bootstrap sobre tarefas e bootstrap pareado.

Segue o MTEB-BR [16]: a unidade de reamostragem é a tarefa. O IC 95 % da média (ex.: Mean22)
vem de reamostrar tarefas com reposição; o p-valor pareado entre dois modelos é a fração de
reamostragens (das mesmas tarefas) em que o sinal da diferença média se inverte, bilateral.
"Vitória" só é declarada com p < 0,05.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def bootstrap_mean(values: list[float], n: int = 10_000, seed: int = 42, conf: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    a = (1 - conf) / 2
    return {"mean": float(v.mean()), "lo": float(np.quantile(means, a)), "hi": float(np.quantile(means, 1 - a)),
            "n": int(len(v))}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n: int = 10_000, seed: int = 42) -> dict:
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return {"delta": math.nan, "p": math.nan, "n": 0, "lo": math.nan, "hi": math.nan}
    d = np.array([a[t] - b[t] for t in tasks], dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"delta": float(d.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)),
            "p": float(min(1.0, p)), "n": len(tasks)}


def main_score(result_json: dict) -> float | None:
    """Score principal de um JSON de resultado do mteb (média dos splits/subsets avaliados)."""
    scores = result_json.get("scores") or {}
    vals = []
    for entries in scores.values():
        for e in entries if isinstance(entries, list) else []:
            if isinstance(e, dict) and "main_score" in e:
                vals.append(float(e["main_score"]))
    return float(np.mean(vals)) if vals else None


def load_scores(cache_root: Path, model_name: str) -> dict[str, float]:
    """{tarefa: score} a partir do cache de resultados do mteb (revisão mais recente)."""
    base = Path(cache_root) / "results" / model_name.replace("/", "__")
    out: dict[str, tuple[float, float]] = {}
    if not base.exists():
        return {}
    for f in base.rglob("*.json"):
        if f.name == "model_meta.json":
            continue
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        s = main_score(data)
        name = data.get("task_name") or f.stem
        if s is not None and (name not in out or f.stat().st_mtime > out[name][1]):
            out[name] = (s, f.stat().st_mtime)
    return {k: v[0] for k, v in out.items()}
