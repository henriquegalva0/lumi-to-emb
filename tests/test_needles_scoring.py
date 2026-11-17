import numpy as np
import pytest

from lumi_ctx.needles import CITIES, draw_needles, number_variants, prompt_rng, question_for, render_user_message
from lumi_ctx.scoring import extract_numbers, has_think_leak, lenient_hit, score


def test_cities_unique_and_unambiguous():
    assert len(CITIES) == 60
    assert len(set(CITIES)) == 60
    for a in CITIES:
        for b in CITIES:
            if a != b:
                assert a not in b, f"{a!r} está contido em {b!r}"


def test_rng_is_deterministic_and_model_independent():
    a = draw_needles("MK", prompt_rng(1, "MK", 4096, 0.5, 3))
    b = draw_needles("MK", prompt_rng(1, "MK", 4096, 0.5, 3))
    c = draw_needles("MK", prompt_rng(1, "MK", 4096, 0.5, 4))
    assert a == b
    assert a != c


@pytest.mark.parametrize("task,n", [("C0", 1), ("S", 1), ("MK", 4), ("MV", 4)])
def test_needle_shapes(task, n):
    d = draw_needles(task, np.random.default_rng(0))
    values = [x["value"] for x in d["needles"]]
    assert len(d["needles"]) == n
    assert len(set(values)) == n
    assert all(len(v) == 7 and v.isdigit() and v[0] != "0" for v in values)
    assert d["needles"][0]["role"] == "target" and d["needles"][0]["city"] == d["city"]
    if task == "MK":
        assert len({x["city"] for x in d["needles"]}) == 4
    if task == "MV":
        assert {x["city"] for x in d["needles"]} == {d["city"]}


def test_prompt_text_matches_spec():
    msg = render_user_message("PALHEIRO", question_for("S", "Recife"))
    assert msg.startswith("Abaixo há um texto longo. Em algum ponto dele há informações sobre números mágicos.\n")
    assert "<texto>\nPALHEIRO\n</texto>\n\nPergunta: Qual é o número mágico de Recife? Responda apenas com o número." in msg
    assert "Quais são todos os números mágicos de Natal?" in question_for("MV", "Natal")
    assert number_variants("1234567") == ("1234567", "1.234.567", "1 234 567")


def test_strict_uses_first_number():
    assert score("S", "1234567", ["1234567"])["strict"] == 1.0
    assert score("S", "O número é 1234567.", ["1234567"])["strict"] == 1.0
    out = score("S", "7654321 ou talvez 1234567", ["1234567"])
    assert out["strict"] == 0.0 and out["lenient"] == 1.0


def test_lenient_accepts_separators_but_strict_does_not():
    out = score("S", "1.234.567", ["1234567"])
    assert out["strict"] == 0.0 and out["lenient"] == 1.0
    assert lenient_hit("1234567", "1 234 567")
    assert extract_numbers("12345678 1234567 123456") == ["1234567"]


def test_mk_distractor_and_mv_fraction():
    mk = score("MK", "5555555", ["1111111"], distractors=["5555555", "6666666"])
    assert mk["strict"] == 0.0 and mk["hit_distractor"] == 1.0
    mv = score("MV", "1111111, 2222222, 9999999", ["1111111", "2222222", "3333333", "4444444"])
    assert mv["strict"] == 0.5 and mv["mv_all"] == 0.0 and mv["mv_extra"] == 1
    mv_all = score("MV", "4444444, 3333333, 2222222, 1111111", ["1111111", "2222222", "3333333", "4444444"])
    assert mv_all["strict"] == 1.0 and mv_all["mv_all"] == 1.0
    none = score("S", "Não sei.", ["1234567"])
    assert none["strict"] == 0.0 and not none["answered_number"]


def test_think_leak():
    assert has_think_leak("<think>pensando</think>1234567")
    assert has_think_leak("1234567", token_ids=[5, 12], think_ids={12, 13})
    assert not has_think_leak("1234567", token_ids=[5, 6], think_ids={12, 13})
