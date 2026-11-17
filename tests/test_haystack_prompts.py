import re

import numpy as np
import pytest

from conftest import synthetic_articles
from lumi_ctx.common import load_config
from lumi_ctx.haystack import (
    article_sentence_starts,
    build_pool,
    clean_article,
    load_pool,
    sentence_starts,
    token_index,
)
from lumi_ctx.needles import number_variants
from lumi_ctx.prompts import (
    PromptBuilder,
    build_prompt_set,
    check_template,
    enumerate_specs,
    render_ids,
    stop_token_ids,
    think_token_ids,
)


def test_clean_article_keeps_only_prose():
    raw = (
        "Título\n* item de lista curto\n{| class=wikitable\n| a | b |\n"
        "Uma frase de prosa longa o bastante para passar no filtro de tamanho de linha mínima. Outra frase ( ) aqui.\n"
        "Linha sem ponto final mas comprida o suficiente para passar no filtro de tamanho mínimo"
    )
    out = clean_article(raw, min_line_chars=40)
    assert out.startswith("Uma frase de prosa")
    assert "( )" not in out and "  " not in out
    assert "\n" not in out  # só um parágrafo sobreviveu


def test_sentence_split_respects_abbreviations():
    p = "O Sr. Silva chegou em 1990. J. R. R. Tolkien escreveu livros. Viveu no séc. XX e morreu em 1973 d.C. Fim."
    starts = sentence_starts(p)
    sents = [p[a:b].strip() for a, b in zip(starts, starts[1:] + [len(p)])]
    assert sents[0] == "O Sr. Silva chegou em 1990."
    assert sents[1] == "J. R. R. Tolkien escreveu livros."
    # "séc." e "d.C." são tratados como abreviação (conservador: nunca corta no meio de "séc. XX")
    assert sents[-1] == "Viveu no séc. XX e morreu em 1973 d.C. Fim."
    assert article_sentence_starts("Ab cd. Ef gh.\nIj kl.") == [0, 7, 14]


@pytest.fixture()
def pool_and_index(tiny_cfg, tiny_tok, tmp_path):
    path = tmp_path / "pool.parquet"
    meta = build_pool(tiny_cfg, path, tiny_tok, seed=1, articles_iter_factory=synthetic_articles)
    pool = load_pool(path)
    return pool, token_index(pool, tiny_tok, tmp_path), meta, path


def test_pool_build_filters_and_metadata(pool_and_index, tiny_cfg):
    pool, tok_cum, meta, _ = pool_and_index
    assert meta["n_tokens"] >= tiny_cfg["haystack"]["target_tokens"]
    assert meta["filter_stats"]["magic"] > 0 and meta["filter_stats"]["too_short"] > 0
    assert "mágic" not in pool.text
    assert len(tok_cum) == pool.n_sentences + 1
    assert np.all(np.diff(tok_cum) > 0)
    assert pool.piece(0) and pool.text.startswith(pool.piece(0))


def test_token_index_matches_real_tokenization(pool_and_index, tiny_tok):
    pool, tok_cum, _, _ = pool_and_index
    for s in (0, 100, 500):
        e = s + 60
        text = pool.text[pool.sent_char[s]:pool.sent_char[e]]
        real = len(tiny_tok(text, add_special_tokens=False)["input_ids"])
        assert abs((tok_cum[e] - tok_cum[s]) - real) <= 3


def test_template_check_and_special_ids(tiny_tok):
    chk = check_template(tiny_tok)
    assert chk["ok"] and chk["generation_suffix_off"].endswith("<think></think>")
    assert len(think_token_ids(tiny_tok)) == 2
    assert tiny_tok.convert_tokens_to_ids("<|im_end|>") in stop_token_ids(tiny_tok)


def test_builder_hits_length_and_places_needles(pool_and_index, tiny_cfg, tiny_tok):
    pool, tok_cum, _, _ = pool_and_index
    b = PromptBuilder(tiny_tok, pool, tok_cum, tiny_cfg)
    tol = tiny_cfg["experiment"]["length_tolerance"]
    for L in tiny_cfg["experiment"]["lengths"]:
        for task, depth in [("S", 0.0), ("S", 0.5), ("S", 1.0), ("MK", 0.25), ("MV", 0.75)]:
            meta, ids = b.build(task, L, depth, 0)
            assert meta["n_tokens"] == len(ids) <= L
            assert len(ids) >= L * (1 - tol)
            text = tiny_tok.decode(ids)
            hay = text.split("<texto>\n", 1)[1].split("\n</texto>", 1)[0]
            for n in meta["needles"]:
                sent = f"O número mágico de {n['city']} é {n['value']}."
                assert hay.count(sent) == 1
                i = hay.index(sent)
                assert i == 0 or hay[i - 1] in " \n"  # sempre em fronteira de frase
                after = hay[i + len(sent):]
                assert after == "" or after[0] in " \n"
                rest = hay.replace(sent, "")
                assert not any(v in rest for v in number_variants(n["value"]))
            if task == "S" and depth == 0.0:
                assert hay.startswith("O número mágico de")
            if task == "S" and depth == 1.0:
                assert hay.endswith(f"é {meta['answers'][0]}.")
            assert text.endswith("<think></think>")
            assert ids == render_ids(tiny_tok, text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>")[0])


def test_builder_is_deterministic_across_instances(pool_and_index, tiny_cfg, tiny_tok):
    pool, tok_cum, _, _ = pool_and_index
    b1 = PromptBuilder(tiny_tok, pool, tok_cum, tiny_cfg)
    b2 = PromptBuilder(tiny_tok, pool, tok_cum, tiny_cfg)
    m1, i1 = b1.build("MV", 1024, 0.5, 1)
    m2, i2 = b2.build("MV", 1024, 0.5, 1)
    assert i1 == i2 and m1["answers"] == m2["answers"]
    m3, i3 = b1.build("MV", 1024, 0.5, 2)
    assert i3 != i1


def test_spec_counts_match_design():
    full = enumerate_specs(load_config("stage1.yaml"))
    counts = {t: sum(1 for s in full if s["task"] == t) for t in ("C0", "S", "MK", "MV")}
    assert counts == {"C0": 50, "S": 1760, "MK": 400, "MV": 400}
    assert len(full) == 2610
    smoke = enumerate_specs(load_config("smoke.yaml"))
    assert len(smoke) == 45
    assert {s["uid"] for s in smoke} <= {s["uid"] for s in full}  # smoke ⊂ completo


def test_prompt_set_parallel_build_and_reuse(pool_and_index, tiny_cfg, tiny_tok, tiny_tok_dir, tmp_path):
    pool, _, _, path = pool_and_index
    ps = build_prompt_set(tiny_cfg, tiny_tok, str(tiny_tok_dir), pool, path, tmp_path / "prompts", n_workers=2)
    assert len(ps.index) == len(enumerate_specs(tiny_cfg))
    row = next(r for r in ps.index if r["task"] == "S")
    ids = ps.ids(row)
    assert len(ids) == row["n_tokens"]
    again = build_prompt_set(tiny_cfg, tiny_tok, str(tiny_tok_dir), pool, path, tmp_path / "prompts", n_workers=1)
    assert again.key == ps.key and again.index == ps.index
    uids = [r["uid"] for r in ps.index]
    assert len(uids) == len(set(uids))
    assert all(re.match(r"^(C0-r\d\d|(S|MK|MV)-L\d{6}-d\d{4}-r\d\d)$", u) for u in uids)
