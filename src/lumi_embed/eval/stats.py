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


def _lang_ok(entry: dict, languages: tuple[str, ...] | None) -> bool:
    """O subconjunto é de um dos idiomas pedidos (código ISO 639-3, ex.: "por" em "por-Latn")?"""
    if not languages:
        return True
    langs = entry.get("languages")
    if not langs:  # sem a informação: mantém
        return True
    return any(str(lang).split("-")[0] in languages for lang in langs)


def load_scores(cache_root: Path, model_name: str, languages: tuple[str, ...] | None = None) -> dict[str, float]:
    """{tarefa: score} a partir do cache de resultados do mteb.

    Junta todos os arquivos da tarefa (o mteb grava uma pasta por revisão do modelo, e suítes
    diferentes podem avaliar a mesma tarefa em idiomas diferentes, como o MassiveIntent em português no
    MTEB-PT e em inglês no MTEB eng v2). Para cada (split, subconjunto) vale o arquivo mais recente; com
    `languages`, só entram os subconjuntos desses idiomas. A nota é a média sobre o que entrou.
    """
    base = Path(cache_root) / "results" / model_name.replace("/", "__")
    if not base.exists():
        return {}
    entries: dict[str, dict[tuple, tuple[float, float]]] = {}
    for f in base.rglob("*.json"):
        if f.name == "model_meta.json":
            continue
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name = data.get("task_name") or f.stem
        mtime = f.stat().st_mtime
        for split, es in (data.get("scores") or {}).items():
            for e in es if isinstance(es, list) else []:
                if isinstance(e, dict) and "main_score" in e and _lang_ok(e, languages):
                    key = (split, e.get("hf_subset"))
                    task = entries.setdefault(name, {})
                    if key not in task or mtime > task[key][1]:
                        task[key] = (float(e["main_score"]), mtime)
    return {t: float(np.mean([v for v, _ in d.values()])) for t, d in entries.items() if d}
