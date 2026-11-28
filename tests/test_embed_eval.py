"""Estatística, relatório (só a partir de arquivos de resultado) e adaptador do mteb."""

import importlib.util
import json

import pytest

from lumi_embed.eval import report as rep
from lumi_embed.eval.stats import bootstrap_mean, load_scores, main_score, paired_bootstrap


def test_bootstrap_and_paired():
    b = bootstrap_mean([0.5] * 10)
    assert b["mean"] == b["lo"] == b["hi"] == 0.5
    a = {f"t{i}": 0.6 + 0.01 * i for i in range(20)}
    c = {f"t{i}": 0.5 + 0.01 * i for i in range(20)}
    r = paired_bootstrap(a, c)
    assert abs(r["delta"] - 0.1) < 1e-9 and r["p"] < 0.01 and r["n"] == 20
    tie = paired_bootstrap({"x": 0.5, "y": 0.6}, {"x": 0.6, "y": 0.5})
    assert tie["p"] > 0.5
    assert paired_bootstrap({}, {})["n"] == 0


def _write_result(root, model, task, score):
    d = root / "results" / model.replace("/", "__") / "rev1"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task}.json").write_text(json.dumps({"task_name": task, "scores": {"test": [{"main_score": score}]}}))


def test_main_score_and_load(tmp_path):
    assert main_score({"scores": {"test": [{"main_score": 0.4}], "dev": [{"main_score": 0.6}]}}) == 0.5
    _write_result(tmp_path, "lumi-embed/C2", "HateBR", 0.7)
    assert load_scores(tmp_path, "lumi-embed/C2") == {"HateBR": 0.7}


def test_report_only_uses_result_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    card = tmp_path / "card.md"
    card.write_text("antes\n<!-- RESULTS:START -->\nvelho\n<!-- RESULTS:END -->\ndepois\n")
    monkeypatch.setattr(rep, "REPO_ROOT", tmp_path)
    (tmp_path / "docs" / "huggingface").mkdir(parents=True)
    (tmp_path / "docs" / "huggingface" / "README.md").write_text(card.read_text())
    cache = tmp_path / "results" / "embed" / "mteb"
    for i, t in enumerate(rep.MTEB_BR_TASKS):
        _write_result(cache, "lumi-embed/C2", t, 0.60 + 0.001 * i)
        _write_result(cache, "lumi-embed/C2-direct", t, 0.55 + 0.001 * i)
    _write_result(cache, "lumi-embed/R0", "HateBR", 0.3)  # incompleto: sem Mean22
    md = rep.build_report({})
    t12 = md.split("## Tabela 12")[1].split("## Tabela 13")[0]
    rows = {line.split("|")[1].strip(): line for line in t12.splitlines() if line.startswith("| ")}
    assert "61.05 [" in rows["C2"]            # média das 22 tarefas (× 100), com IC
    assert rows["R0"].split("|")[2].strip() == "—"   # sem as 22 tarefas, não há Mean22
    assert rows["C1"].count("—") >= 8          # sem resultados: tudo "—"
    a1 = next(line for line in md.splitlines() if line.startswith("| A1 "))
    assert "+5.00" in a1 and "| 22 |" in a1
    assert "66.21 [16]" in md                  # literatura com referência
    out = (tmp_path / "docs" / "huggingface" / "README.md").read_text()
    assert "velho" not in out and "MTEB-BR (22 tarefas" in out and out.startswith("antes") and out.endswith("depois\n")
    assert "| C2-nohn |" not in out            # no card, só linhas com resultado
    assert (tmp_path / "results/embed/tables/table13b_mteb_br_tasks.csv").exists()


def test_report_release_seeds_and_hypotheses(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    monkeypatch.setattr(rep, "REPO_ROOT", tmp_path)
    (tmp_path / "docs" / "huggingface").mkdir(parents=True)
    (tmp_path / "docs" / "huggingface" / "README.md").write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n")
    cache = tmp_path / "results" / "embed" / "mteb"
    base = {"C2": 0.60, "C2-s2": 0.602, "C2-s3": 0.598, "C2-real": 0.60, "C2-syn": 0.59, "lumi-emb": 0.61}
    for i, t in enumerate(rep.MTEB_BR_TASKS):
        for m, v in base.items():
            _write_result(cache, f"lumi-embed/{m}", t, v + 0.001 * i)
        _write_result(cache, rep.QWEN4B, t, 0.65 + 0.001 * i)
    rel = tmp_path / "checkpoints" / "lumi-emb"
    rel.mkdir(parents=True)
    (rel / "choice.json").write_text(json.dumps({"source": "M-C", "merge_selected_variant": "M-C-base0.75",
                                                 "rule": "regra"}))
    mc = tmp_path / "checkpoints" / "M-C"
    mc.mkdir(parents=True)
    (mc / "merge_report.json").write_text(json.dumps({
        "selected": "M-C-base0.75", "best_individual": "C2-real",
        "dev": {"M-C-base0.75": {"mean_ndcg@10": 0.9779}, "C2-real": {"mean_ndcg@10": 0.9771}}}))
    md = rep.build_report({})
    t12 = md.split("## Tabela 12")[1].split("## Tabela 13")[0]
    rows = {line.split("|")[1].strip(): line for line in t12.splitlines() if line.startswith("| ")}
    assert "62.05 [" in rows["M-C"]            # a origem do lumi-emb herda as notas (mesmos pesos)
    seeds = md.split("## Variação entre sementes")[1].split("##")[0]
    assert "| MTEB-BR Mean22 | 61.05 | 61.25 | 60.85 | 61.05 | 0.20 |" in seeds
    hyp = {line.split("|")[1].strip(): line for line in md.split("## Hipóteses")[1].splitlines()
           if line.startswith("| H")}
    assert "Refutada" in hyp["H1"] and "-5.00" in hyp["H1"]   # C2 abaixo do Qwen3-4B com IC todo negativo
    assert "Não refutada" in hyp["H3"]                          # reais > sintéticos
    assert "Não refutada" in hyp["H6"] and "+0.08" in hyp["H6"]  # dev: merge > melhor individual
    assert hyp["H2"].rstrip().endswith("| — |")                 # C2-direct não rodou
    pair = next(line for line in md.splitlines() if line.startswith("| lumi-emb − Qwen3-Embedding-4B"))
    assert "-4.00" in pair and "0/22" in pair
    assert "## Sensibilidade à contaminação" not in md     # sem auditoria, sem a seção
    audit = {"sets": {"S1/train": {"hits": [{"field": "positive", "tasks": ["Quati"]},
                                            {"field": "negative", "tasks": ["HateBR"]},
                                            {"field": "query", "tasks": ["outra"]}]},
                      "S1/dev": {"hits": [{"field": "query", "tasks": ["JurisTCU"]}]}}}
    d = tmp_path / "results/embed/contamination"
    d.mkdir(parents=True)
    (d / "audit.json").write_text(json.dumps(audit))
    assert rep.contaminated_tasks() == ["Quati"]          # só query/positivo dos conjuntos de treino
    md = rep.build_report({})
    sec = md.split("## Sensibilidade à contaminação")[1].split("\n## ")[0]
    row = next(line for line in sec.splitlines() if line.startswith("| Qwen3-Embedding-4B"))
    assert "| 66.05 | 66.03 |" in row and "-4.00" in row and "Mean21" in sec


def test_highlights_and_figures_come_from_result_files(tmp_path, monkeypatch):
    # O bloco de destaques (README e card) e as figuras saem só das notas: o modelo publicado contra o
    # lumi-emb e os baselines, com o MLDR-pt em 8k vindo do cache separado (sufixo L8192).
    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    monkeypatch.setattr(rep, "REPO_ROOT", tmp_path)
    block = "intro\n<!-- HIGHLIGHTS:START -->\nvelho\n<!-- HIGHLIGHTS:END -->\nfim\n"
    (tmp_path / "README.md").write_text(block)
    (tmp_path / "docs" / "huggingface").mkdir(parents=True)
    (tmp_path / "docs" / "huggingface" / "README.md").write_text(block + "<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n")
    cache = tmp_path / "results" / "embed" / "mteb"
    for i, t in enumerate(rep.MTEB_BR_TASKS):
        _write_result(cache, "lumi-embed/lumi-emb-long", t, 0.62 + 0.001 * i)
        _write_result(cache, "lumi-embed/lumi-emb", t, 0.62 + 0.001 * i - (0.002 if i % 2 else -0.002))
        _write_result(cache, "lumi-embed/R0", t, 0.30)
        _write_result(cache, rep.QWEN4B, t, 0.50 if t == "HateBR" else 0.65 + 0.001 * i)
        _write_result(cache, rep.QWEN8B, t, 0.55 if t == "HateBR" else 0.67 + 0.001 * i)
    for m, v512, v8k in (("lumi-embed/lumi-emb", 0.62, 0.4788), ("lumi-embed/lumi-emb-long", 0.64, 0.7394),
                         (rep.QWEN4B, 0.67, 0.80)):
        _write_result(cache, m, "MultiLongDocRetrieval", v512)
        _write_result(cache, f"{m}-{rep.LONG_TAG}", "MultiLongDocRetrieval", v8k)
    rel = tmp_path / "checkpoints" / "lumi-emb-long"
    rel.mkdir(parents=True)
    (rel / "choice.json").write_text(json.dumps({"source": "M-L", "merge_selected_variant": "M-L-x"}))
    md = rep.build_report({})
    pair = next(line for line in md.splitlines() if line.startswith("| lumi-emb-long − Qwen3-Embedding-4B"))
    assert "1/22" in pair                                   # só o HateBR
    assert "| lumi-emb-long − lumi-emb |" in md
    out = (tmp_path / "README.md").read_text()
    assert "velho" not in out and out.startswith("intro") and out.endswith("fim\n")
    assert "| **73.94** | 47.88 sem o estágio longo (+26.06); 64.00 lendo só os primeiros 512 tokens |" in out
    assert "| MTEB-BR (22 tarefas nativas em PT-BR) | **" in out and "30.00 do Lumi-4B sem treino" in out
    assert "| HateBR | Classificação | **62.00** | 50.00 | +12.00 | ✓ |" in out   # > 4B e > 8B
    assert "| Quati |" not in out                           # perde para o 4B: fica fora da lista
    assert "O contexto longo não custou nada" in out         # lumi-emb-long − lumi-emb sem diferença
    card = (tmp_path / "docs" / "huggingface" / "README.md").read_text()
    assert "HateBR" in card and "<picture>" not in card     # no card, imagens simples (docs/figures/)
    figs = tmp_path / "docs" / "embed" / "figures"
    if importlib.util.find_spec("matplotlib"):
        assert {p.name for p in figs.glob("*.png")} >= {"mldr-pt-light.png", "mldr-pt-dark.png",
                                                       "mteb-br-categories-light.png", "mteb-br-ranking-dark.png"}
        assert 'srcset="docs/embed/figures/mldr-pt-dark.png"' in out
        assert "](docs/figures/mteb-br-ranking-light.png)" in card


def test_lessons_read_each_ablation_as_an_effect():
    # Bloco ABLATIONS do README e do card: Δ = com o componente − sem ele; o efeito sai do IC, e as
    # ablações sem as 22 tarefas não aparecem.
    base = {t: 0.60 + 0.001 * i for i, t in enumerate(rep.MTEB_BR_TASKS)}
    S = {"C2": base,
         "C2-direct": {t: v - 0.012 for t, v in base.items()},
         "C2-nohn": {t: v + (0.004 if i % 2 else -0.004) for i, (t, v) in enumerate(base.items())},
         "B2f": {t: v - 0.01 for t, v in base.items()},
         "C2-s2": {t: v - 0.002 for t, v in base.items()}}
    md = rep.lessons(S)
    row = lambda k: next(line for line in md.splitlines() if f"| {k} |" in line)  # noqa: E731
    assert row("C2 − C2-direct").startswith("| Estágio S1") and "+1.20" in row("C2 − C2-direct")
    assert row("C2 − C2-direct").endswith("| **ajuda** |")
    assert row("C2 − C2-nohn").endswith("| empate |")
    assert row("B2f − C2").endswith("| **piora** |")
    assert "C2-syn" not in md and "¹" not in md
    assert "2 sementes varia 0.14" in md
    assert rep.lessons({}).startswith("_As lições")


def test_figures_render_all_kinds(tmp_path):
    pytest.importorskip("matplotlib")
    from lumi_embed.eval.figures import render

    spec = {
        "a": ("mldr_slope", {"data": [("x", "muted", 62.0, 47.9), ("y", "s1", 64.0, 73.9), ("z", "s2", 66.9, 80.2)]}),
        "b": ("length_curves", {"data": {"Passkey": {"y": {256: 90, 8192: 100, 16384: 92}, "z": {256: 100, 8192: 96}}},
                                "roles": {"y": "s1", "z": "s2"}, "trained": 8192}),
        "c": ("category_dots", {"cats": ["Cls", "Ret"], "series": [("y", "s1", [67.7, 62.6]), ("z", "s2", [63.7, 62.3]),
                                                                   ("w", "s3", [67.1, 64.8])]}),
        "d": ("ranking_bars", {"rows": [("y", "s1", 62.5), ("z", "muted", 65.5), ("r", "muted", 30.6)],
                               "title": "t", "subtitle": "s"}),
    }
    out = render(spec, tmp_path)
    assert len(out) == 8 and all(p.stat().st_size > 5000 for p in out)
    assert {p.name for p in out} >= {"a-light.png", "a-dark.png", "d-light.png"}


def test_instruction_resolution():
    pytest.importorskip("torch")
    pytest.importorskip("mteb")
    from mteb.types import PromptType

    from lumi_embed.eval.mteb_encoder import resolve_instruction

    inst = {"default": "D", "by_type": {"Retrieval": "R", "STS": "S"}, "by_task": {"HateBR": "H"}}
    assert resolve_instruction(inst, "Quati", "Retrieval", PromptType.query) == "R"
    assert resolve_instruction(inst, "Quati", "Retrieval", PromptType.document) is None
    assert resolve_instruction(inst, "AssinSTS", "STS", None) == "S"
    assert resolve_instruction(inst, "HateBR", "Classification", None) == "H"
    assert resolve_instruction(inst, "X", "Clustering", None) == "D"


def test_same_max_length_for_every_model():
    # Protocolo: o comprimento vem da suíte ou de eval.max_length, nunca do modelo (um baseline de 32k
    # codificava documentos inteiros enquanto o nosso corta em 512, e estourava a memória).
    import types

    from lumi_embed.eval.run import cap_length, suite_params
    from lumi_embed.utils import load_config

    ecfg = load_config("eval.yaml")["eval"]
    assert suite_params(ecfg, "mteb_pt") == (512, ecfg["batch_size"])
    assert suite_params(ecfg, "mteb_pt", model_batch=64) == (512, 64)
    assert suite_params(ecfg, "longembed", model_batch=64) == (32768, 1)  # lote da suíte limita

    st = types.SimpleNamespace(max_seq_length=32768)
    assert cap_length(types.SimpleNamespace(model=st), 512) and st.max_seq_length == 512
    assert not cap_length(types.SimpleNamespace(model=object()), 512)


def test_publish_creates_private_repo_branch_and_card(tmp_path, monkeypatch):
    import sys
    import types
    from pathlib import Path

    from lumi_embed import publish as pub

    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("token", bool(token)))

        def create_repo(self, **kw):
            calls.append(("create_repo", kw["repo_id"], kw["private"], kw["exist_ok"]))

        def create_branch(self, **kw):
            calls.append(("branch", kw["branch"]))

        def upload_folder(self, **kw):
            names = sorted(p.name for p in Path(kw["folder_path"]).iterdir())
            calls.append(("upload", kw["revision"], names))
            if "README.md" not in names:   # pesos: apaga os antigos do branch no mesmo commit
                assert "*.safetensors" in kw["delete_patterns"]
            return "ok"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    m = tmp_path / "checkpoints" / "C2" / "model"
    m.mkdir(parents=True)
    (m / "config.json").write_text("{}")
    pub.publish_checkpoint("org/repo", "C2")
    assert ("create_repo", "org/repo", True, True) in calls and ("branch", "C2") in calls
    assert ("upload", "C2", ["config.json"]) in calls
    calls.clear()
    pub.publish_checkpoint("org/repo", "C2", release=True)
    ups = [c for c in calls if c[0] == "upload"]
    assert ups[0][1] == "main" and ups[1][1] == "main" and "README.md" in ups[1][2]   # pesos, depois o card
    assert not any(c[0] == "branch" for c in calls)


def test_longembed_by_length_and_run_summary_accumulates(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    d = tmp_path / "results/embed/mteb/results/lumi-embed__lumi-emb/rev1"
    d.mkdir(parents=True)
    scores = {f"test_{n}": [{"main_score": v}] for n, v in [(256, 0.96), (8192, 1.0), (32768, 0.48)]}
    (d / "LEMBPasskeyRetrieval.json").write_text(json.dumps({"task_name": "LEMBPasskeyRetrieval", "scores": scores}))
    h, rows = rep.lemb_length_rows([("lumi-emb", "lumi-embed/lumi-emb"), ("Outro", "org/outro")])
    assert h == ["Modelo", "Tarefa", "256", "8192", "32768"]
    assert rows == [["lumi-emb", "Passkey", "96", "100", "48"]]

    from lumi_embed.eval import run as runmod

    f = tmp_path / "runs.json"
    runmod._write_run(f, {"mteb_br": {"minutes": 90}})
    runmod._write_run(f, {"longembed": {"minutes": 108}})
    assert set(json.loads(f.read_text())) == {"mteb_br", "longembed"}


def test_scores_filtered_by_language_and_merged_across_revisions(tmp_path):
    # O MTEB eng v2 grava o MassiveIntent em inglês numa pasta de revisão nova: a família PT não pode
    # passar a usar essa nota, e o MTEB eng v2 não pode usar a de português.
    import os

    base = tmp_path / "results" / "lumi-embed__lumi-emb"
    pt = {"task_name": "MassiveIntentClassification", "scores": {"test": [
        {"main_score": 0.68, "hf_subset": "pt", "languages": ["por-Latn"]}]}}
    en = {"task_name": "MassiveIntentClassification", "scores": {"test": [
        {"main_score": 0.72, "hf_subset": "en", "languages": ["eng-Latn"]}]}}
    for rev, data in (("old", pt), ("new", en)):
        (base / rev).mkdir(parents=True)
        (base / rev / "MassiveIntentClassification.json").write_text(json.dumps(data))
    os.utime(base / "new" / "MassiveIntentClassification.json", (2e9, 2e9))
    assert load_scores(tmp_path, "lumi-embed/lumi-emb", ("por",)) == {"MassiveIntentClassification": 0.68}
    assert load_scores(tmp_path, "lumi-embed/lumi-emb", ("eng",)) == {"MassiveIntentClassification": 0.72}
    assert abs(load_scores(tmp_path, "lumi-embed/lumi-emb")["MassiveIntentClassification"] - 0.70) < 1e-9


def test_long_suite_cache_name_and_mldr_table(tmp_path, monkeypatch):
    # O MLDR-pt em 8k tem cache próprio (sufixo L8192): não sobrescreve o MLDR em 512 do MTEB-PT, e o
    # relatório mostra os dois lado a lado; o lumi-emb-long herda as notas da origem (choice.json).
    from lumi_embed.eval.run import cache_name
    from lumi_embed.utils import load_config

    ecfg = load_config("eval.yaml")["eval"]
    assert cache_name("lumi-embed/lumi-emb", ecfg, "mldr_pt_long") == "lumi-embed/lumi-emb-L8192"
    assert cache_name("lumi-embed/lumi-emb", ecfg, "mteb_br") == "lumi-embed/lumi-emb"
    s = ecfg["suites"]["mldr_pt_long"]
    assert s["max_length"] == 8192 and s["languages"] == ["por"]

    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    cache = tmp_path / "results" / "embed" / "mteb"

    def put(model, score):
        d = cache / "results" / model.replace("/", "__") / "rev1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "MultiLongDocRetrieval.json").write_text(json.dumps({"task_name": "MultiLongDocRetrieval", "scores": {
            "test": [{"main_score": score, "hf_subset": "pt", "languages": ["por-Latn"]}]}}))

    put("lumi-embed/lumi-emb", 0.30)
    put("lumi-embed/lumi-emb-L8192", 0.20)
    put("lumi-embed/M-L-L8192", 0.45)
    put(rep.QWEN4B, 0.40)
    put(rep.QWEN4B + "-L8192", 0.60)
    rel = tmp_path / "checkpoints" / "lumi-emb-long"
    rel.mkdir(parents=True)
    (rel / "choice.json").write_text(json.dumps({"source": "M-L", "merge_selected_variant": "M-L-avg"}))
    S = rep.collect([rep.QWEN4B])
    rows = {r[0]: r for r in rep.mldr_long_rows(S, [rep.QWEN4B])}
    assert rows["lumi-emb"][1:] == ["30.00", "20.00"]
    assert rows["Qwen3-Embedding-4B"][1:] == ["40.00", "60.00"]
    assert rows["M-L"][2] == "45.00" and "C3-long" not in rows
