"""Pipeline de dados sem rede nem GPU: schema, Wikipédia, síntese (parsing/filtros), mineração
(seleção de negativos), descontaminação e composição dos conjuntos."""

import json
import random

import numpy as np
import pytest

from lumi_embed.data import mix as mixmod
from lumi_embed.data.decontaminate import Hasher, is_contaminated, overlap_ratio, render_report
from lumi_embed.data.mine import near_duplicate, select_negatives
from lumi_embed.data.schema import example_id, make_example, validate
from lumi_embed.data.sources import wiki_sections
from lumi_embed.data.synth import build_requests, copies_document, extract_json, looks_portuguese, to_example
from lumi_embed.utils import load_config

WIKI = """Gates é um empresário americano.

William Henry Gates III é um magnata, empresário, diretor executivo, investidor, filantropo e autor americano, que ficou conhecido por fundar, junto com Paul Allen, a Microsoft.

Biografia

Infância e juventude
Gates nasceu em uma família de classe média de Seattle. Seu pai era advogado de grandes empresas, e sua mãe foi professora da Universidade de Washington e diretora de bancos.

Carreira
| tabela | ruim |
Trabalhou na Taito com o desenvolvimento de software básico para máquinas de jogos eletrônicos até seus 16 anos, e depois fundou a Microsoft com Paul Allen."""


def test_schema_ids_and_validation():
    ex = make_example("retrieval", "pt-BR", "inst", "  o que  é? ", "texto  positivo", {"dataset": "x"},
                      negatives=["neg"])
    assert ex["query"] == "o que é?" and ex["negatives"] == [{"text": "neg", "score": None}]
    assert ex["id"] == example_id("retrieval", "o que é?", "texto positivo")
    assert validate(ex) == []
    bad = dict(ex, task="xyz", split="test", query="texto positivo")
    assert len(validate(bad)) == 3


def test_wiki_sections_extracts_lead_and_sections():
    lead, sections = wiki_sections(WIKI)
    assert lead and lead[0].startswith("William Henry Gates III")
    heads = [h for h, _ in sections]
    assert "Infância e juventude" in heads and "Carreira" in heads
    assert all("|" not in p for _, ps in sections for p in ps)


def test_extract_json_tolerates_reasoning_and_noise():
    assert extract_json('analysis blah {"x":1} assistantfinal {"query": "imposto de renda"}', "query")["query"] \
        == "imposto de renda"
    assert extract_json('texto {"query": ""} fim', "query") is None
    assert extract_json("sem json", "query") is None
    assert extract_json('{"parafrase": "a", "contraste": "b"}', "parafrase")["contraste"] == "b"


def test_copy_and_language_filters():
    doc = "As despesas médicas podem ser deduzidas da base de cálculo do imposto de renda devido pelo contribuinte"
    assert copies_document("podem ser deduzidas da base de cálculo do imposto de renda", doc)
    assert not copies_document("quais gastos com saúde abatem o IR?", doc)
    assert looks_portuguese("quais são as regras para dedução de despesas médicas no imposto de renda?")
    assert not looks_portuguese("which medical expenses can be deducted from the income tax base this year?")


@pytest.fixture()
def data_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    cfg = load_config("base.yaml")
    from lumi_embed.utils import deep_merge

    cfg = deep_merge(cfg, load_config("data.yaml"))
    seeds = [{"id": f"w-{i}", "text": f"A cidade número {i} foi fundada no século XIX. Ela tem uma universidade "
                                       f"antiga e um museu de arte muito visitado por turistas do país inteiro.",
              "source": "wikipedia"} for i in range(60)]
    (tmp_path / "data/embed/seeds").mkdir(parents=True)
    (tmp_path / "data/embed/seeds/seeds.jsonl").write_text("\n".join(json.dumps(s) for s in seeds))
    longs = [{"id": f"wl-{i}", "text": " ".join(["palavra"] * 900) + f" fim {i}.", "source": "wikipedia"}
             for i in range(9)]
    (tmp_path / "data/embed/seeds/long_seeds.jsonl").write_text("\n".join(json.dumps(s) for s in longs))
    for g in cfg["synth"]["generators"].values():
        g.update(n_retrieval=10, n_sts=5, n_long=2)
    return cfg


def test_synth_requests_are_disjoint_across_generators_and_parse(data_cfg):
    keys = list(data_cfg["synth"]["generators"])
    reqs = {k: build_requests(data_cfg, k, i, len(keys), seed=1) for i, k in enumerate(keys)}
    seeds_by_gen = [{r["seed"]["id"] for r in rs} for rs in reqs.values()]
    assert all(not (a & b) for i, a in enumerate(seeds_by_gen) for b in seeds_by_gen[i + 1 :])
    kinds = {r["kind"] for rs in reqs.values() for r in rs}
    assert kinds == {"retrieval", "sts", "long"}
    r = next(x for x in reqs[keys[0]] if x["kind"] == "retrieval")
    assert "Responda apenas em JSON" in r["prompt"] and r["seed"]["text"] in r["prompt"]
    ex = to_example(r, '{"query": "qual universidade antiga existe na cidade?"}', data_cfg, keys[0])
    assert ex and ex["task"] == "retrieval" and ex["source"]["gen_key"] == keys[0] and ex["positive"] == r["seed"]["text"]
    s = next(x for x in reqs[keys[0]] if x["kind"] == "sts")
    ex = to_example(s, '{"parafrase": "Uma frase reescrita com outras palavras para o teste", "contraste": "outra"}',
                    data_cfg, keys[0])
    assert ex["task"] == "sts" and ex["negatives"][0]["kind"] == "contraste"
    lg = next(x for x in reqs[keys[0]] if x["kind"] == "long")
    ex = to_example(lg, '{"query": "o que aparece no fim do texto?"}', data_cfg, keys[0])
    assert ex["source"]["long"] and ex["positive"] == lg["seed"]["text"]


def test_select_negatives_percpos_and_naive():
    cands = [("positivo quase igual", 0.95), ("muito parecido", 0.92), ("bom negativo", 0.70), ("outro", 0.60),
             ("fraco", 0.30)]
    percpos, naive = select_negatives(cands, "positivo quase igual", pos_score=0.9, k=2, perc=0.95)
    assert [n["text"] for n in percpos] == ["bom negativo", "outro"]  # 0,92 ≥ 0,95·0,9 = 0,855 sai
    assert [n["text"] for n in naive] == ["muito parecido", "bom negativo"]
    assert near_duplicate("A casa é azul e grande e bonita.", "a casa é azul e grande e bonita")


def test_decontamination_ngrams_and_exact():
    h = Hasher()
    eval_text = "o supremo tribunal federal julgou a ação direta de inconstitucionalidade proposta pelo partido " \
                "político em março de dois mil e vinte"
    idx = {"ngrams": np.unique(h.ngrams(eval_text, 13, 1)), "texts": np.array([Hasher.text_hash(eval_text)],
                                                                                dtype=np.uint64)}
    assert is_contaminated(h, eval_text.upper(), idx, 13, 0.3) == "exact"
    partial = "segundo a notícia, " + eval_text + " e outras coisas"
    assert overlap_ratio(h, partial, idx["ngrams"], 13) > 0.3
    assert is_contaminated(h, partial, idx, 13, 0.3) == "ngram"
    assert is_contaminated(h, "um texto totalmente diferente sobre culinária baiana e moqueca de peixe", idx, 13,
                           0.3) is None
    # hash estável entre processos (não depende de PYTHONHASHSEED)
    assert Hasher.text_hash("Texto") == Hasher.text_hash("texto")
    md = render_report({"benchmarks": ["mteb_br"], "ngram": 13, "stride": 2, "threshold": 0.3,
                        "eval_index": {"n_texts": 1, "n_ngrams": 5, "tasks": {"X": 1}},
                        "files": {"raw/a.jsonl": {"in": 3, "kept": 2, "dropped": {"ngram": 1}, "negatives_dropped": 0}}})
    assert "| `raw/a.jsonl` | 3 | 2 | 0 | 1 | 0 |" in md


def _mined(i, comp, k=8):
    src = {"retrieval": {"dataset": "wikimedia/wikipedia"}, "synth": {"gen": "g", "corpus": "w"},
           "sts": {"gen": "g"}, "mmarco": {"dataset": "unicamp-dl/mmarco"}, "en": {"dataset": "st/gooaq"},
           "long": {"gen": "g", "long": True}}[comp]
    task = "sts" if comp == "sts" else "retrieval"
    lang = "en" if comp == "en" else "pt-BR"
    ex = make_example(task, lang, "i", f"q {comp} {i}", f"p {comp} {i}", src,
                      negatives=[{"text": f"n{j} {comp} {i}", "score": 0.5} for j in range(k)])
    ex["naive_negatives"] = [{"text": f"nv{j} {comp} {i}", "score": 0.6} for j in range(k)]
    if comp == "mmarco" and i < 5:
        ex.update(r1=True, bm25_negatives=[f"bm{j}" for j in range(70)])
    return ex


def test_mix_builds_disjoint_sets_and_ablations(data_cfg, tmp_path):
    rows = [_mined(i, c) for c in ("retrieval", "synth", "sts", "mmarco", "en", "long") for i in range(120)]
    random.Random(0).shuffle(rows)
    (tmp_path / "data/embed/mined").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/embed/mined/all.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "data/embed/clean").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/embed/clean/mmarco_pt.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r.get("r1")))
    cfg = data_cfg
    cfg["mix"].update(inputs=["mined/all.jsonl"], dev_per_component=10, dev_long=5)
    cfg["mix"]["s1"]["size"], cfg["mix"]["s2"]["size"] = 200, 100
    man = mixmod.run_mix(cfg)

    def load(name, split="train"):
        p = tmp_path / f"data/embed/sets/{name}/{split}.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()]

    s1, s2, dev = load("S1"), load("S2"), load("S1", "dev")
    ids1, ids2, idsd = {r["id"] for r in s1}, {r["id"] for r in s2}, {r["id"] for r in dev}
    assert not (ids1 & ids2) and not (ids1 & idsd) and not (ids2 & idsd)
    assert all(len(r["negatives"]) == 1 for r in s1) and all(len(r["negatives"]) == 4 for r in s2)
    assert "labeled_pt" in man["sets"]["S2"]["redistributed_from"]
    naive, nohn = load("S2-naivehn"), load("S2-nohn")
    assert [r["id"] for r in naive] == [r["id"] for r in s2]
    assert all(n["text"].startswith("nv") for r in naive for n in r["negatives"])
    assert all(r["negatives"] == [] for r in nohn)
    real, syn = load("S2-real"), load("S2-syn")
    assert all(mixmod.component(r) in mixmod.REAL for r in real)
    assert all(mixmod.component(r) in mixmod.SYNTH for r in syn)
    assert not ({r["id"] for r in real} & (ids1 | idsd))
    r1 = load("R1")
    assert len(r1) == 5 and all(len(r["negatives"]) == 63 for r in r1)
    longs = load("long")
    assert all(r["source"].get("long") for r in longs)


def test_mmarco_samples_only_queries_with_triples(data_cfg, tmp_path, monkeypatch):
    # Regressão: as queries eram sorteadas entre todas as de treino, mas só as que estão nos trios
    # oficiais têm positivo; metade das escolhidas sumia.
    from lumi_embed.data import sources

    d = tmp_path / "mm"
    d.mkdir()
    (d / "q.tsv").write_text("".join(f"{i}\tpergunta número {i}?\n" for i in range(20)))
    (d / "t.tsv").write_text("".join(f"{i}\t{100 + i}\t{200 + i * 10 + k}\n" for i in range(0, 20, 2) for k in range(5)))
    pids = [100 + i for i in range(0, 20, 2)] + [200 + i * 10 + k for i in range(0, 20, 2) for k in range(5)]
    (d / "c.tsv").write_text("".join(f"{p}\tpassagem {p} sobre o assunto.\n" for p in pids))
    files = {"Q": d / "q.tsv", "T": d / "t.tsv", "C": d / "c.tsv"}
    monkeypatch.setattr(sources, "download", lambda repo, rev, path: str(files[path]))
    data_cfg["sources"]["mmarco"].update(queries="Q", triples="T", collection="C", n_queries=6, r1_queries=2,
                                         r1_negatives=4, other_negatives=1)
    stats = sources.mmarco_pt(data_cfg, seed=1)
    assert stats == {"mmarco_pt": 6, "r1_queries": 2}
    rows = [json.loads(x) for x in (tmp_path / "data/embed/raw/mmarco_pt.jsonl").read_text().splitlines()]
    assert all(int(r["source"]["qid"]) % 2 == 0 for r in rows)
    assert sorted(len(r["bm25_negatives"]) for r in rows) == [1, 1, 1, 1, 4, 4]
