"""Estatística, relatório (só a partir de arquivos de resultado) e adaptador do mteb."""

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
