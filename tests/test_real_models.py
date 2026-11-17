"""Testes com os tokenizers reais e a Wikipédia real (precisam de rede).

Rode com:  LUMI_TEST_NETWORK=1 ./run_stage1.sh selftest -k real
Baixam só tokenizer/config dos modelos (sem pesos) e ~2 row groups da Wikipédia.
"""

import os

import pytest

from lumi_ctx.common import load_config
from lumi_ctx.haystack import build_pool, load_pool, token_index
from lumi_ctx.prompts import PromptBuilder, check_template, load_tokenizer, render_ids, tokenizer_dir, tokenizer_fingerprint

pytestmark = pytest.mark.skipif(os.environ.get("LUMI_TEST_NETWORK") != "1", reason="defina LUMI_TEST_NETWORK=1")


@pytest.fixture(scope="module")
def real():
    cfg = load_config("stage1.yaml")
    toks = {m["key"]: load_tokenizer(tokenizer_dir(m)) for m in cfg["models"]}
    return cfg, toks


def test_templates_disable_thinking_and_models_share_tokenizer(real):
    cfg, toks = real
    for key, tok in toks.items():
        chk = check_template(tok)
        assert chk["ok"], (key, chk)
        assert chk["generation_suffix_off"].endswith("<|im_start|>assistant\n<think></think>")
    fps = {tokenizer_fingerprint(t) for t in toks.values()}
    assert len(fps) == 1, "Lumi e controle deveriam ter o mesmo tokenizer (prompts pareados)"


def test_real_prompts_at_every_length(real, tmp_path):
    cfg, toks = real
    cfg["haystack"]["target_tokens"] = 700_000  # suficiente para um prompt de 261k
    tok = toks["lumi"]
    path = tmp_path / "pool.parquet"
    build_pool(cfg, path, tok, seed=int(cfg["experiment"]["seed"]))
    pool = load_pool(path)
    b = PromptBuilder(tok, pool, token_index(pool, tok), cfg)
    for L in cfg["experiment"]["lengths"]:
        for task, depth in (("S", 0.0), ("S", 1.0), ("MK", 0.5), ("MV", 0.25)):
            meta, ids = b.build(task, L, depth, 0)
            assert L * 0.98 <= len(ids) <= L
            assert ids == render_ids(tok, tok.decode(ids).split("<|im_start|>user\n", 1)[1].split("<|im_end|>")[0])
