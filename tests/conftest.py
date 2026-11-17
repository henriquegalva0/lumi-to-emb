"""Fixtures dos testes locais (sem GPU e sem rede).

- `tiny_tok`: tokenizer BPE pequeno, treinado na hora sobre texto sintético em PT,
  com os tokens especiais e um chat template no formato do Nemotron/Lumi
  (enable_thinking=False -> "<think></think>" no fim do prompt).
- `synthetic_articles`: fonte de artigos determinística no lugar do streaming do Hub.
- `tiny_cfg`: configuração reduzida (comprimentos 512–2048) para rodar em segundos.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumi_ctx.common import load_config  # noqa: E402

CHAT_TEMPLATE = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
    "{{- '<|im_start|>system\\n<|im_end|>\\n' }}"
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{%- if enable_thinking %}{{- '<|im_start|>assistant\\n<think>\\n' }}"
    "{%- else %}{{- '<|im_start|>assistant\\n<think></think>' }}{%- endif %}"
    "{%- endif %}"
)

WORDS = (
    "a cidade de rio porto campo grande serra vale história povo rei guerra paz ciência arte música "
    "foi é era são tem teve havia segundo durante após antes entre sobre para com sem pelo pela "
    "governo estado nação império república século ano década região província município capital "
    "economia agricultura indústria comércio rede estrada ferrovia escola universidade igreja museu "
    "primeiro segundo terceiro grande pequeno antigo novo importante principal conhecido famoso"
).split()
SPECIALS = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def _sentence(rng: np.random.Generator) -> str:
    n = int(rng.integers(8, 22))
    words = [WORDS[int(i)] for i in rng.integers(0, len(WORDS), size=n)]
    words[0] = words[0].capitalize()
    if rng.random() < 0.3:
        words.insert(int(rng.integers(1, n)), str(int(rng.integers(1500, 2024))))
    return " ".join(words) + "."


def make_article(rng: np.random.Generator, n_par: int = 6) -> str:
    lines = ["Título do Artigo", "* item curto de lista", "{| class=wikitable", "| célula | célula |"]
    for _ in range(n_par):
        lines.append(" ".join(_sentence(rng) for _ in range(int(rng.integers(3, 7)))))
        lines.append("Seção")
    return "\n".join(lines)


def build_tiny_tokenizer(out_dir: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    rng = np.random.default_rng(0)
    corpus = [make_article(rng) for _ in range(200)]
    corpus += ["O número mágico de São Paulo é 1234567.", "Pergunta: Qual é o número mágico de Recife?"]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=800, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="<|im_end|>", unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_dir)
    (out_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    (out_dir / "config.json").write_text(json.dumps({
        "hybrid_override_pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
        "num_key_value_heads": 8, "head_dim": 128, "mamba_num_heads": 96, "mamba_head_dim": 80,
        "ssm_state_size": 128, "max_position_embeddings": 262144,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
    }))
    return tok


@pytest.fixture(scope="session")
def tiny_tok_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tiny_tok")
    build_tiny_tokenizer(d)
    return d


@pytest.fixture(scope="session")
def tiny_tok(tiny_tok_dir):
    from lumi_ctx.prompts import load_tokenizer

    return load_tokenizer(str(tiny_tok_dir))


def synthetic_articles(source: dict, seed: int):
    rng = np.random.default_rng([seed, 5])
    i = 0
    while True:
        text = make_article(rng, n_par=int(rng.integers(5, 12)))
        if i % 17 == 3:
            text = "Curto."  # reprovado pelo filtro de tamanho
        if i % 29 == 5:
            text = text + "\nEm física nuclear, os números mágicos indicam camadas completas de núcleons."
        yield {"source_id": str(i), "title": f"Artigo {i}", "text": text, "file": "sintetico", "row_group": 0}
        i += 1


@pytest.fixture()
def tiny_cfg() -> dict:
    cfg = copy.deepcopy(load_config("stage1.yaml"))
    cfg["experiment"]["lengths"] = [512, 1024, 2048]
    cfg["experiment"]["tasks"] = {
        "C0": {"reps": 4},
        "S": {"depths": [0.0, 0.5, 1.0], "reps": 2},
        "MK": {"depths": [0.0, 1.0], "reps": 2, "n_distractors": 3},
        "MV": {"depths": [0.5], "reps": 2, "n_values": 4},
    }
    cfg["experiment"]["bootstrap"]["n_resamples"] = 200
    cfg["haystack"]["target_tokens"] = 60000
    cfg["haystack"]["sources"] = [cfg["haystack"]["sources"][0]]
    cfg["vllm"]["chunk_tokens"] = 6000
    cfg["perf"]["reps"] = 1
    cfg["build"]["workers"] = 2
    return cfg
