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


def test_run_synth_end_to_end_with_fake_vllm(data_cfg, tmp_path, monkeypatch):
    # O laço completo da síntese (geração em lotes, filtros, gravação, retomada) com um vLLM falso.
    # Regressão: a pasta data/embed/synth/ não existia e o primeiro lote se perdia ao gravar.
    import sys
    import types

    calls = {"llm": 0, "convs": 0}

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["llm"] += 1

        def chat(self, convs, sp, use_tqdm=False, chat_template_kwargs=None):
            calls["convs"] += len(convs)
            outs = []
            for conv in convs:
                prompt = conv[-1]["content"]
                if prompt.startswith("Reescreva a frase"):
                    text = ('{"parafrase": "Uma frase reescrita com outras palavras para o teste", '
                            '"contraste": "Uma frase parecida com sentido diferente"}')
                else:
                    text = 'assistantfinal{"query": "qual universidade antiga existe na cidade?"}'
                outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]))
            return outs

    fake = types.ModuleType("vllm")
    fake.LLM = FakeLLM
    fake.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", fake)
    from lumi_embed.data.synth import run_synth

    data_cfg["synth"]["chunk"] = 7
    key = list(data_cfg["synth"]["generators"])[0]
    assert not (tmp_path / "data/embed/synth").exists()
    res = run_synth(data_cfg, key)
    rows = [json.loads(x) for x in (tmp_path / "data/embed/synth" / f"{key}.jsonl").read_text().splitlines()]
    assert res["new"] == len(rows) == calls["convs"] > 7  # mais de um lote, nada rejeitado
    assert {r["task"] for r in rows} == {"retrieval", "sts"} and any(r["source"].get("long") for r in rows)
    # retomada: tudo já feito, o modelo nem é carregado de novo
    res2 = run_synth(data_cfg, key)
    assert res2["new"] == 0 and calls["llm"] == 1


def test_run_mine_end_to_end_with_fake_teacher(data_cfg, tmp_path, monkeypatch):
    # O laço completo da mineração com um "teacher" falso (saco de palavras com hash), na CPU.
    pytest.importorskip("torch")
    import hashlib
    import sys
    import types

    def bow(text: str, dim: int = 128) -> list[float]:
        v = [0.0] * dim
        for w in text.lower().replace("?", " ").replace(".", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        return v

    engines = []

    class FakeLLM:
        def __init__(self, **kwargs):
            assert kwargs.get("runner") == "pooling"
            engines.append(1)

        def embed(self, texts, use_tqdm=False, tokenization_kwargs=None):
            assert tokenization_kwargs["truncation"] is True
            return [types.SimpleNamespace(outputs=types.SimpleNamespace(embedding=bow(t))) for t in texts]

    fake = types.ModuleType("vllm")
    fake.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake)
    from lumi_embed.data.mine import run_mine

    rows = []
    for i in range(24):
        rows.append(make_example("retrieval", "pt-BR", "inst", f"quando a cidade{i} foi fundada?",
                                 f"A cidade{i} foi fundada no século XIX e tem um museu de arte.", {"dataset": "x"}))
    for i in range(3):
        rows.append(make_example("retrieval", "pt-BR", "inst", f"o que diz o trecho {i} do documento?",
                                 f"documento longo {i} " + "palavra " * 50, {"gen": "g", "long": True}))
    for i in range(6):
        rows.append(make_example("sts", "pt-BR", "inst", f"o rio{i} corta a cidade de norte a sul",
                                 f"a cidade é atravessada pelo rio{i} de norte a sul", {"gen": "g"},
                                 negatives=[{"text": f"o rio{i} corta a cidade de leste a oeste", "kind": "contraste"}]))
    clean = tmp_path / "data/embed/clean"
    clean.mkdir(parents=True)
    (clean / "a.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    data_cfg["mining"].update(inputs=["clean/a.jsonl"], embed_chunk=10, synthetic_min_cos=0.0, positive_rank_check=64,
                              defer_families=[])

    stats = run_mine(data_cfg)
    assert set(stats) == {"retrieval", "retrieval_long", "sts"}
    for fam, st in stats.items():
        assert st["in"] == st["out"] + st["drop_low_cos"] + st["drop_rank"], fam
        out = [json.loads(x) for x in (tmp_path / "data/embed/mined" / f"{fam}.jsonl").read_text().splitlines()]
        assert len(out) == st["out"]
        for r in out:
            assert r["positive"] not in {n["text"] for n in r["negatives"]}
            assert isinstance(r["pos_score"], float) and r["teacher"]["rank_pos"] >= 1
    sts = [json.loads(x) for x in (tmp_path / "data/embed/mined/sts.jsonl").read_text().splitlines()]
    assert all(r["negatives"][0]["kind"] == "contraste" for r in sts)
    ret = [json.loads(x) for x in (tmp_path / "data/embed/mined/retrieval.jsonl").read_text().splitlines()]
    assert sum(len(r["negatives"]) for r in ret) > 0  # outras cidades viram negativos
    assert (tmp_path / "data/embed/mined/stats.json").exists()
    # retomada: famílias prontas (arquivo + stats_<família>.json) não são refeitas nem carregam o teacher
    assert all((tmp_path / f"data/embed/mined/stats_{fam}.json").exists() for fam in stats)
    (tmp_path / "data/embed/mined/stats_sts.json").unlink()   # simula um job que morreu antes do sts
    stats2 = run_mine(data_cfg)
    assert stats2 == stats and len(engines) == 2               # só o sts foi refeito
    # família adiada: não é minerada nem gera arquivo, e o resto continua pulado
    (tmp_path / "data/embed/mined/stats_retrieval_long.json").unlink()
    (tmp_path / "data/embed/mined/retrieval_long.jsonl").unlink()
    data_cfg["mining"]["defer_families"] = ["retrieval_long"]
    stats3 = run_mine(data_cfg)
    assert stats3["retrieval_long"] == {"in": 3, "deferred": True} and len(engines) == 2
    assert not (tmp_path / "data/embed/mined/retrieval_long.jsonl").exists()


def test_spun_text_filter():
    # Texto "spinado" da web (troca automática de "o" por "este", inclusive no fim de palavras)
    from lumi_embed.data.decontaminate import is_spun

    assert is_spun("Vejam a situaçãeste do Estado do Rio, recordista em terceirizações.")
    assert is_spun("Com a plateia lotada de modo a assistir à sua atuaçãeste, Myrthes surpreendeu este juiz")
    for ok in ("A mãe e as irmãs estão nesta situação.", "Os alemães e os cães, este e aquele.",
               "As lições desta questão são essas.", "Irmã, esta é a capital."):
        assert not is_spun(ok), ok


def test_run_decontaminate_drops_spun_examples_and_negatives(data_cfg, tmp_path, monkeypatch):
    from lumi_embed.data import decontaminate as dc

    empty = {"ngrams": np.array([], dtype=np.uint64), "texts": np.array([], dtype=np.uint64),
             "meta": {"n_texts": 0, "n_ngrams": 0, "tasks": {}}}
    monkeypatch.setattr(dc, "build_index", lambda cfg: dict(empty))
    rows = [
        make_example("retrieval", "pt-BR", "i", "qual a situação do estado?", "Vejam a situaçãeste do Estado.", {"dataset": "x"}),
        make_example("retrieval", "pt-BR", "i", "qual a capital do Brasil?", "Brasília é a capital do Brasil.",
                     {"dataset": "x"}, negatives=["Com a sua atuaçãeste, ele venceu.", "Rio de Janeiro foi capital."]),
    ]
    raw = tmp_path / "data/embed/raw"
    raw.mkdir(parents=True)
    (raw / "t.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    data_cfg["decontamination"]["inputs"] = ["raw/t.jsonl"]
    rep = dc.run_decontaminate(data_cfg)
    s = rep["files"]["raw/t.jsonl"]
    assert s == {"in": 2, "kept": 1, "dropped": {"spun": 1}, "negatives_dropped": 1}
    kept = [json.loads(x) for x in (tmp_path / "data/embed/clean/t.jsonl").read_text().splitlines()]
    assert [n["text"] for n in kept[0]["negatives"]] == ["Rio de Janeiro foi capital."]
    assert "spinado" in (tmp_path / "results/embed/contamination/report.md").read_text()


def test_eval_index_is_saved_per_task_and_resumes(data_cfg, tmp_path, monkeypatch):
    # O índice de avaliação é salvo tarefa a tarefa: um job que morre no meio continua de onde parou.
    import types

    from lumi_embed.data import decontaminate as dc

    loads = []

    def task(name, texts):
        t = types.SimpleNamespace(metadata=types.SimpleNamespace(name=name), dataset=None)

        def load_data():
            loads.append(name)
            t.dataset = {"test": {"corpus": texts}}

        t.load_data = load_data
        return t

    long_txt = " ".join(f"palavra{i}" for i in range(40))
    tasks = [task("A", [long_txt, "curto"]), task("B", [long_txt + " extra"]), task("Grande", [long_txt + " x"])]
    monkeypatch.setattr(dc, "eval_tasks", lambda benchmarks: tasks)
    data_cfg["decontamination"]["skip_tasks"] = ["Grande"]
    idx = dc.build_index(data_cfg)
    assert loads == ["A", "B"] and idx["meta"]["tasks"]["Grande"].startswith("excluída")
    assert idx["meta"]["tasks"]["A"] == 1 and len(idx["texts"]) == 2 and len(idx["ngrams"]) > 0
    # simula um job que morreu depois de indexar A e B: o índice final some, as tarefas ficam
    (tmp_path / "data/embed/decontam/eval_index.npz").unlink()
    loads.clear()
    idx2 = dc.build_index(data_cfg)
    assert loads == [] and (idx2["ngrams"] == idx["ngrams"]).all()


def test_eval_index_from_before_version_2_is_rebuilt(data_cfg, tmp_path, monkeypatch):
    # Um índice gravado antes da correção do extrator (chave sem index_version) não pode ser reaproveitado.
    from lumi_embed.data import decontaminate as dc

    d = tmp_path / "data/embed/decontam"
    d.mkdir(parents=True)
    dc_cfg = data_cfg["decontamination"]
    np.savez(d / "eval_index.npz", ngrams=np.array([1, 2], dtype=np.uint64), texts=np.array([3], dtype=np.uint64))
    (d / "eval_index.json").write_text(json.dumps({"key": {k: dc_cfg[k] for k in ("benchmarks", "ngram", "stride")},
                                                   "tasks": {}, "n_ngrams": 2, "n_texts": 1}))
    called = []
    monkeypatch.setattr(dc, "eval_tasks", lambda b: called.append(b) or [])
    dc_cfg["skip_tasks"] = []
    dc.build_index(data_cfg)
    assert called, "o índice antigo devia ser refeito"


def _legacy_task(name, corpus, queries, loads=None):
    # Formato antigo do mteb (retrieval do mteb_pt, LongEmbed): textos em task.corpus / task.queries.
    import types

    t = types.SimpleNamespace(metadata=types.SimpleNamespace(name=name), dataset=None)

    def load_data():
        if loads is not None:
            loads.append(name)
        t.corpus = {"test": {f"d{i}": {"title": "", "text": c} for i, c in enumerate(corpus)}}
        t.queries = {"test": {f"q{i}": q for i, q in enumerate(queries)}}
        t.relevant_docs = {"test": {"q0": {"d0": 1}}}

    t.load_data = load_data
    return t


def test_eval_index_reads_legacy_corpus_and_queries(data_cfg, tmp_path, monkeypatch):
    # Regressão: as tarefas de retrieval do MTEB-BR entravam com 0 textos (só task.dataset era lido).
    from lumi_embed.data import decontaminate as dc

    doc = " ".join(f"documento{i}" for i in range(40))
    query = "qual é a regra para declarar despesas médicas no imposto de renda deste ano"
    monkeypatch.setattr(dc, "eval_tasks", lambda b: [_legacy_task("FaqBacenRetrieval", [doc], [query])])
    idx = dc.build_index(data_cfg)
    assert idx["meta"]["tasks"]["FaqBacenRetrieval"] == 2
    h = dc.Hasher()
    assert dc.is_contaminated(h, doc, idx, 13, 0.3) == "exact"
    assert dc.is_contaminated(h, query, idx, 13, 0.3) == "exact"


def test_eval_index_fails_on_task_without_texts(data_cfg, monkeypatch):
    from lumi_embed.data import decontaminate as dc
    from lumi_embed.utils import EmbedError

    monkeypatch.setattr(dc, "eval_tasks", lambda b: [_legacy_task("Vazia", [], [])])
    with pytest.raises(EmbedError, match="sem nenhum texto"):
        dc.build_index(data_cfg)


def test_audit_redoes_empty_cached_tasks_and_finds_hits(data_cfg, tmp_path, monkeypatch):
    # Cache com a tarefa vazia (extrator antigo) é refeito; a auditoria acha o texto e atribui à tarefa.
    from lumi_embed.data import decontaminate as dc

    n, stride = data_cfg["decontamination"]["ngram"], data_cfg["decontamination"]["stride"]
    task_dir = tmp_path / f"data/embed/decontam/tasks_n{n}_s{stride}"
    task_dir.mkdir(parents=True)
    empty = np.array([], dtype=np.uint64)
    np.savez(task_dir / "Quati.npz", texts=empty, ngrams=empty, count=np.array(0))
    doc = " ".join(f"passagem{i}" for i in range(40))
    loads = []
    monkeypatch.setattr(dc, "eval_tasks", lambda b: [_legacy_task("Quati", [doc], ["pergunta curta demais"], loads)])
    sets = tmp_path / "data/embed/sets/S1"
    sets.mkdir(parents=True)
    rows = [{"id": "a", "query": "consulta qualquer sobre a passagem", "positive": doc, "negatives": []},
            {"id": "b", "query": "outra consulta", "positive": "texto limpo sem relação nenhuma com a avaliação",
             "negatives": [{"text": doc, "score": 0.5}]},
            {"id": "c", "query": "nada", "positive": "também limpo", "negatives": []}]
    (sets / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rep = dc.run_audit(data_cfg)
    assert loads == ["Quati"] and rep["focus_tasks"] == ["Quati"]
    s = rep["sets"]["S1/train"]
    assert s["examples"] == 3 and s["examples_hit"] == 2
    assert s["by_field"] == {"positive": 1, "negative": 1} and s["by_task"] == {"Quati": 2}
    assert "Quati: 2" in (tmp_path / "results/embed/contamination/audit.md").read_text()
    # reexecutar não reindexa e mantém a lista de tarefas conferidas
    loads.clear()
    assert dc.run_audit(data_cfg)["focus_tasks"] == ["Quati"] and loads == []


def test_cached_near_dup_matches_reference_and_selection_is_unchanged():
    # A versão com cache e parada antecipada tem de dar exatamente o mesmo resultado da original.
    from lumi_embed.data.mine import NearDup

    rng = random.Random(0)
    vocab = [f"p{i}" for i in range(30)]
    base = [" ".join(rng.choice(vocab) for _ in range(rng.randint(2, 60))) for _ in range(40)]
    variants = []
    for b in base:
        w = b.split()
        for _ in range(3):
            v = list(w)
            for _ in range(rng.randint(0, 4)):
                if v:
                    v[rng.randrange(len(v))] = rng.choice(vocab)
            variants.append(" ".join(v) + rng.choice(["", ".", "!"]))
    texts = base + variants + ["A casa é azul.", "a casa É azul", "", "um dois"]
    dup = NearDup()
    for a in texts[::3]:
        for b in texts[::5]:
            assert dup(a, b) == near_duplicate(a, b), (a, b)
    for _ in range(50):
        pos = rng.choice(texts)
        cands = sorted(((rng.choice(texts), rng.random()) for _ in range(20)), key=lambda x: -x[1])
        ps = rng.random()
        ref_clean = [(t, s) for t, s in cands if not near_duplicate(t, pos)]
        ref_perc = [{"text": t, "score": round(s, 4)} for t, s in ref_clean if s < 0.95 * ps][:4]
        ref_naive = [{"text": t, "score": round(s, 4)} for t, s in ref_clean][:4]
        assert select_negatives(cands, pos, ps, 4, 0.95, dup=dup) == (ref_perc, ref_naive)
        assert select_negatives(cands, pos, ps, 4, 0.95) == (ref_perc, ref_naive)


def test_exit_now_flushes_logs_and_output(tmp_path):
    # Etapas com vLLM saem por os._exit (a desmontagem do motor aborta): nada pode se perder.
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    code = ("import logging, sys; from lumi_embed.cli import _exit_now; from lumi_embed.utils import setup_logging; "
            "from pathlib import Path; setup_logging(Path(sys.argv[1])); "
            "logging.getLogger('lumi_embed.t').info('gravado no log'); "
            "print('saida padrao'); _exit_now(0)")
    log = tmp_path / "x.log"
    r = subprocess.run([sys.executable, "-c", code, str(log)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": src})
    assert r.returncode == 0, r.stderr
    assert "saida padrao" in r.stdout and "gravado no log" in log.read_text()
