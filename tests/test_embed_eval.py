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
    assert "0.6105 [" in rows["C2"]           # média das 22 tarefas, com IC
    assert rows["R0"].split("|")[2].strip() == "—"   # sem as 22 tarefas, não há Mean22
    assert rows["C1"].count("—") >= 8          # sem resultados: tudo "—"
    a1 = next(line for line in md.splitlines() if line.startswith("| A1 "))
    assert "+0.0500" in a1 and "| 22 |" in a1
    assert "0,6621 [16]" in md                 # literatura com referência
    out = (tmp_path / "docs" / "huggingface" / "README.md").read_text()
    assert "velho" not in out and "Tabela 12" in out and out.startswith("antes") and out.endswith("depois\n")
    assert (tmp_path / "results/embed/tables/table13b_mteb_br_tasks.csv").exists()


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
