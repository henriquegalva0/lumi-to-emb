"""Ponta a ponta sem GPU: preparação, fila com 2 workers (subprocessos), pontuação,
análise, relatório, retomada e pacote de diagnóstico. Usa o backend `fake`."""

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import ROOT, synthetic_articles


@pytest.fixture()
def e2e_env(tmp_path, monkeypatch, tiny_cfg, tiny_tok_dir):
    work = tmp_path / "work"
    monkeypatch.setenv("LUMI_WORKDIR", str(work))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_cfg, allow_unicode=True))
    import lumi_ctx.haystack as hs
    import lumi_ctx.runner as rn

    monkeypatch.setattr(rn, "tokenizer_dir", lambda m: str(tiny_tok_dir))
    monkeypatch.setattr(hs, "iter_articles", synthetic_articles)
    return work, cfg_path, rn


def test_full_pipeline_with_fake_backend(e2e_env):
    work, cfg_path, rn = e2e_env
    assert rn.cmd_run("smoke", str(cfg_path), "t_e2e", "fake", 2) == 0
    rdir = work / "results" / "t_e2e"
    for f in ("REPORT.md", "DESVIOS.md", "summary.csv", "perf.csv", "acc_vs_len.png", "heatmap_S_lumi.png",
              "heatmap_S_nemotron.png", "config_used.yaml", "env.json", "lstar.json", "raw/results.jsonl",
              "template_lumi.txt", "paired_tests.csv", "gates.json"):
        assert (rdir / f).exists(), f
    summary = pd.read_csv(rdir / "summary.csv")
    assert set(summary.task) == {"C0", "S", "MK", "MV"}
    assert (summary.n_missing == 0).all()
    results = [json.loads(x) for x in (rdir / "raw/results.jsonl").read_text().splitlines()]
    from lumi_ctx.prompts import enumerate_specs

    assert len(results) == 2 * len(enumerate_specs(yaml.safe_load(cfg_path.read_text())))
    assert {r["worker"] for r in results} <= {"gpuf0", "gpuf1"}
    gates = json.loads((rdir / "gates.json").read_text())
    for model, gs in gates.items():
        assert all(g["ok"] for g in gs), (model, gs)
    report = (rdir / "REPORT.md").read_text()
    assert "RESULTADOS SIMULADOS" in report and "L\\* por modelo" in report and "Recomendação" in report
    info = json.loads((rdir / "run_info.json").read_text())
    assert info["paired_identical"] is True

    # retomar a mesma execução não refaz nada nem duplica resultados
    assert rn.cmd_run("smoke", str(cfg_path), "t_e2e", "fake", 2) == 0
    results2 = (rdir / "raw/results.jsonl").read_text().splitlines()
    assert len(results2) == len(results)


def test_worker_crash_is_retried(e2e_env, monkeypatch):
    work, cfg_path, rn = e2e_env
    # Um worker que cai na primeira vez: a fila devolve o lote e o worker é relançado.
    flag = work.parent / "crash_once"
    flag.write_text("1")
    monkeypatch.setenv("LUMI_TEST_CRASH_ONCE", str(flag))
    assert rn.cmd_run("smoke", str(cfg_path), "t_crash", "fake", 2) == 0
    rdir = work / "results" / "t_crash"
    summary = pd.read_csv(rdir / "summary.csv")
    assert (summary.n_missing == 0).all()
    assert "caiu" in (rdir / "logs" / "main.log").read_text()


def test_resume_latest_and_diagnose(e2e_env):
    work, cfg_path, rn = e2e_env
    rdir = work / "results" / "20990101-000000_smoke"
    rdir.mkdir(parents=True)
    (rdir / "run_info.json").write_text("{}")
    assert rn.latest_incomplete_run("smoke") == "20990101-000000_smoke"
    (rdir / "REPORT.md").write_text("ok")
    assert rn.latest_incomplete_run("smoke") is None

    env = dict(os.environ)
    out = subprocess.run(["bash", str(ROOT / "run_stage1.sh"), "diagnose"], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    tars = list((work / "diagnostics").glob("diag_*.tar.gz"))
    assert len(tars) == 1
    with tarfile.open(tars[0]) as t:
        names = t.getnames()
    assert any(n.endswith("env.txt") for n in names) and any(n.endswith("system.txt") for n in names)
    with tarfile.open(tars[0]) as t:
        env_txt = t.extractfile(next(m for m in t.getmembers() if m.name.endswith("/env.txt"))).read().decode("utf-8", "replace")
    assert "HF_HOME=" in env_txt
