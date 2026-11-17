"""Pontuação das respostas (seção 3.8 da especificação).

- Estrita (principal): o PRIMEIRO número de 7 dígitos da saída é igual ao esperado.
  No MV: fração dos 4 valores esperados presentes entre os números de 7 dígitos da saída.
- Leniente: o número esperado aparece em qualquer parte da saída (também aceita
  separadores de milhar, como "1.234.567"). No MV: fração dos valores que aparecem.
- thinking_leak: a saída contém <think> ou </think> (erro de configuração).
"""

from __future__ import annotations

import re

NUM7 = re.compile(r"(?<!\d)\d{7}(?!\d)")
_DIGIT_SEP = re.compile(r"(?<=\d)[.,\s  ](?=\d)")
THINK_MARKERS = ("<think>", "</think>")


def extract_numbers(text: str) -> list[str]:
    return NUM7.findall(text or "")


def _squash_separators(text: str) -> str:
    return _DIGIT_SEP.sub("", text or "")


def lenient_hit(expected: str, text: str) -> bool:
    return expected in (text or "") or expected in _squash_separators(text)


def has_think_leak(text: str, token_ids: list[int] | None = None, think_ids: set[int] | None = None) -> bool:
    if any(marker in (text or "") for marker in THINK_MARKERS):
        return True
    if token_ids and think_ids:
        return any(t in think_ids for t in token_ids)
    return False


def score(task: str, text: str, answers: list[str], distractors: list[str] | None = None) -> dict:
    """Pontua uma saída. `answers` é a lista de valores esperados (1, ou 4 no MV)."""
    numbers = extract_numbers(text)
    first = numbers[0] if numbers else None
    out: dict = {"pred_first": first, "n_numbers": len(numbers), "answered_number": first is not None}
    if task == "MV":
        expected = set(answers)
        found = set(numbers)
        out["strict"] = len(expected & found) / len(expected)
        out["lenient"] = sum(lenient_hit(a, text) for a in answers) / len(answers)
        out["mv_all"] = float(expected <= found)
        out["mv_extra"] = len(found - expected)  # números que não eram valores da cidade
    else:
        expected = answers[0]
        out["strict"] = float(first == expected)
        out["lenient"] = float(lenient_hit(expected, text))
        out["hit_distractor"] = float(bool(distractors) and first in set(distractors))
    return out
