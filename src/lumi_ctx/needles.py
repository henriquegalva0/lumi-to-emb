"""Agulhas, perguntas e texto do prompt (seção 3.2 e 3.6 da especificação)."""

from __future__ import annotations

import numpy as np

from .common import TASK_CODES

# 60 cidades brasileiras. Nenhum nome é prefixo/substring de outro, para que a
# pergunta "Qual é o número mágico de X?" nunca seja ambígua.
CITIES: tuple[str, ...] = (
    "São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Fortaleza",
    "Recife", "Manaus", "Curitiba", "Porto Alegre", "Belém",
    "Goiânia", "Brasília", "São Luís", "Maceió", "Natal",
    "Teresina", "Campo Grande", "João Pessoa", "Aracaju", "Cuiabá",
    "Florianópolis", "Vitória", "Palmas", "Porto Velho", "Macapá",
    "Rio Branco", "Boa Vista", "Campinas", "Guarulhos", "São Gonçalo",
    "Duque de Caxias", "Nova Iguaçu", "São Bernardo do Campo", "Santo André", "Osasco",
    "Sorocaba", "Ribeirão Preto", "Uberlândia", "Contagem", "Juiz de Fora",
    "Londrina", "Joinville", "Feira de Santana", "Niterói", "Campos dos Goytacazes",
    "Caxias do Sul", "Pelotas", "Santos", "Maringá", "Blumenau",
    "Petrolina", "Caruaru", "Mossoró", "Campina Grande", "Anápolis",
    "Montes Claros", "Ilhéus", "Juazeiro do Norte", "Foz do Iguaçu", "Ponta Grossa",
)

INSTRUCTION = (
    "Abaixo há um texto longo. Em algum ponto dele há informações sobre números mágicos.\n"
    "Leia o texto e responda à pergunta no final."
)

PROMPT_TEMPLATE = "{instruction}\n\n<texto>\n{haystack}\n</texto>\n\n{question}"

QUESTION_SINGLE = "Pergunta: Qual é o número mágico de {city}? Responda apenas com o número."
QUESTION_MULTI = (
    "Pergunta: Quais são todos os números mágicos de {city}? "
    "Responda apenas com os números, separados por vírgula."
)


def needle_sentence(city: str, value: str) -> str:
    return f"O número mágico de {city} é {value}."


def question_for(task: str, city: str) -> str:
    return (QUESTION_MULTI if task == "MV" else QUESTION_SINGLE).format(city=city)


def render_user_message(haystack: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=INSTRUCTION, haystack=haystack, question=question)


def prompt_rng(base_seed: int, task: str, length: int, depth: float | None, rep: int) -> np.random.Generator:
    """Gerador determinístico por prompt.

    Não depende do modelo nem da ordem de execução: o mesmo (tarefa, L, profundidade,
    repetição) gera o mesmo prompt para os dois modelos (comparação pareada) e no smoke.
    """
    depth_pm = -1 if depth is None else int(round(depth * 1000))
    return np.random.default_rng([int(base_seed), TASK_CODES[task], int(length), depth_pm + 1, int(rep)])


def draw_value(rng: np.random.Generator, used: set[str]) -> str:
    """Inteiro aleatório de 7 dígitos (1000000–9999999), distinto dos já usados."""
    while True:
        value = str(int(rng.integers(1_000_000, 10_000_000)))
        if value not in used:
            used.add(value)
            return value


def number_variants(value: str) -> tuple[str, ...]:
    """Formas em que o número poderia aparecer por acaso no palheiro."""
    v = value
    dotted = f"{v[0]}.{v[1:4]}.{v[4:]}"
    spaced = f"{v[0]} {v[1:4]} {v[4:]}"
    return (v, dotted, spaced)


def draw_needles(task: str, rng: np.random.Generator, n_distractors: int = 3, n_values: int = 4) -> dict:
    """Sorteia cidade(s) e número(s) da tarefa.

    Retorna {"city": cidade perguntada, "needles": [{"role", "city", "value"}...]}.
    A primeira agulha é sempre o alvo (a que vai na profundidade definida).
    """
    used: set[str] = set()
    if task in ("C0", "S"):
        city = str(rng.choice(CITIES))
        needles = [{"role": "target", "city": city, "value": draw_value(rng, used)}]
    elif task == "MK":
        idx = rng.choice(len(CITIES), size=1 + n_distractors, replace=False)
        city = CITIES[int(idx[0])]
        needles = [{"role": "target", "city": city, "value": draw_value(rng, used)}]
        for i in idx[1:]:
            needles.append({"role": "distractor", "city": CITIES[int(i)], "value": draw_value(rng, used)})
    elif task == "MV":
        city = str(rng.choice(CITIES))
        needles = [{"role": "target", "city": city, "value": draw_value(rng, used)}]
        for _ in range(n_values - 1):
            needles.append({"role": "value", "city": city, "value": draw_value(rng, used)})
    else:  # pragma: no cover - validado na config
        raise ValueError(task)
    return {"city": city, "needles": needles}


def expected_answers(task: str, needles: list[dict]) -> list[str]:
    if task == "MV":
        return [n["value"] for n in needles]
    return [needles[0]["value"]]
