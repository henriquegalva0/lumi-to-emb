"""Schema único (JSONL) dos exemplos de treino e dev — seção 2.6 do plano.

Campos: id, task, lang, instruction, query, positive, negatives [{text, score}],
pos_score, source {...}, split. Campos extras opcionais: naive_negatives (top-k sem filtro,
ablação A3), bm25_negatives (receita R1), symmetric (instrução nos dois lados).
"""

from __future__ import annotations

import hashlib
import unicodedata

TASKS = ("retrieval", "sts", "classification", "bitext", "clustering", "reranking")
SYMMETRIC_TASKS = ("sts", "bitext", "clustering")
SPLITS = ("train", "dev")


def norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def example_id(task: str, query: str, positive: str) -> str:
    h = hashlib.sha1(f"{task}\x1f{norm_text(query)}\x1f{norm_text(positive)}".encode("utf-8")).hexdigest()
    return h[:16]


def make_example(task: str, lang: str, instruction: str, query: str, positive: str, source: dict,
                 negatives: list | None = None, pos_score: float | None = None, split: str = "train",
                 **extra) -> dict:
    ex = {
        "id": example_id(task, query, positive),
        "task": task,
        "lang": lang,
        "instruction": instruction,
        "query": norm_text(query),
        "positive": norm_text(positive),
        "negatives": [n if isinstance(n, dict) else {"text": norm_text(n), "score": None} for n in (negatives or [])],
        "pos_score": pos_score,
        "source": source,
        "split": split,
    }
    ex.update(extra)
    return ex


def validate(ex: dict) -> list[str]:
    problems = []
    for key in ("id", "task", "lang", "query", "positive", "source", "split"):
        if key not in ex or ex[key] in (None, ""):
            problems.append(f"campo ausente: {key}")
    if ex.get("task") not in TASKS:
        problems.append(f"tarefa inválida: {ex.get('task')}")
    if ex.get("split") not in SPLITS:
        problems.append(f"split inválido: {ex.get('split')}")
    for n in ex.get("negatives") or []:
        if not isinstance(n, dict) or "text" not in n:
            problems.append("negativo sem 'text'")
            break
    if norm_text(ex.get("query", "")) == norm_text(ex.get("positive", "")):
        problems.append("query igual ao positivo")
    return problems


def is_symmetric(ex: dict) -> bool:
    return ex.get("task") in SYMMETRIC_TASKS or bool(ex.get("symmetric"))
