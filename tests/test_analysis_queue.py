import math

import numpy as np
import pandas as pd

from lumi_ctx import workqueue
from lumi_ctx.analysis import (
    bootstrap_ci,
    build_summary,
    l_star,
    lstar_table,
    model_memory,
    recommendation,
    sign_test_p,
)
from lumi_ctx.common import load_config

LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 261120]


def test_l_star_requires_all_shorter_lengths():
    acc = dict(zip(LENGTHS, [1.0, 0.95, 0.9, 0.85, 0.95, 0.99, 0.2, 0.1]))
    assert l_star(acc, LENGTHS, 0.9) == 8192  # 32k volta acima, mas 16k já falhou
    assert l_star({2048: 0.5}, LENGTHS, 0.9) is None
    assert l_star(dict.fromkeys(LENGTHS, 1.0), LENGTHS, 0.9) == 261120
    assert l_star({2048: 1.0, 4096: math.nan}, LENGTHS, 0.9) == 2048


def test_bootstrap_ci_properties():
    lo, hi = bootstrap_ci([1.0] * 20, [0.5] * 20, 500, 1)
    assert lo == hi == 1.0
    v = [1.0] * 10 + [0.0] * 10
    lo, hi = bootstrap_ci(v, [0.0] * 10 + [1.0] * 10, 500, 1)
    assert lo == hi == 0.5  # estratificado: cada estrato é constante
    lo, hi = bootstrap_ci(v, [0.0] * 20, 2000, 1)
    assert 0.25 < lo < 0.5 < hi < 0.75
    assert bootstrap_ci(v, [-1.0] * 20, 100, 1) == bootstrap_ci(v, [-1.0] * 20, 100, 1)


def test_sign_test():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(10, 0) < 0.01
    assert abs(sign_test_p(3, 3) - 1.0) < 1e-12
    assert abs(sign_test_p(1, 0) - 1.0) < 1e-12


def _synthetic_df(cfg):
    rows = []
    rng = np.random.default_rng(0)
    for model, knee in (("lumi", 16384), ("nemotron", 131072)):
        label = next(m["label"] for m in cfg["models"] if m["key"] == model)
        for r in range(50):
            rows.append(dict(model=model, model_label=label, task="C0", length=0, depth=None, status="ok", strict=1.0,
                             lenient=1.0, think_leak=False, answered_number=True, hit_distractor=0.0, mv_all=None,
                             mv_extra=None, n_out=8, finish_reason="stop", n_tokens=95, len_dev=None,
                             target_depth_actual=None, uid=f"C0-r{r:02d}"))
        for L in LENGTHS:
            for d in (0.0, 0.5, 1.0):
                for r in range(10):
                    ok = float(L <= knee or rng.random() < 0.3)
                    rows.append(dict(model=model, model_label=label, task="S", length=L, depth=d, status="ok",
                                     strict=ok, lenient=ok, think_leak=False, answered_number=True,
                                     hit_distractor=0.0, mv_all=None, mv_extra=None, n_out=8, finish_reason="stop",
                                     n_tokens=L - 10, len_dev=-10 / L, target_depth_actual=d,
                                     uid=f"S-L{L:06d}-d{int(d * 1000):04d}-r{r:02d}"))
    return pd.DataFrame(rows)


def test_summary_lstar_and_recommendation():
    cfg = load_config("stage1.yaml")
    cfg["experiment"]["bootstrap"]["n_resamples"] = 200
    summary = build_summary(_synthetic_df(cfg), cfg)
    c0 = summary[(summary.model == "lumi") & (summary.task == "C0")].iloc[0]
    assert c0.acc_strict == 1.0 and c0.ci95_low == 1.0
    ls = lstar_table(summary, cfg)
    assert ls["lumi"]["L_S"] == 16384 and ls["nemotron"]["L_S"] == 131072
    rec = " ".join(recommendation(ls, cfg))
    assert "Chunks de 4k a 8k" in rec and "pré-treino contínuo curto" in rec
    ls["lumi"]["L_S"] = 65536
    assert "Documento inteiro" in " ".join(recommendation(ls, cfg))
    ls["lumi"]["L_S"] = 4096
    assert "Reavaliar" in " ".join(recommendation(ls, cfg))


def test_model_memory_matches_spec(tiny_tok_dir):
    mem = model_memory(tiny_tok_dir / "config.json")
    assert mem["attention_layers"] == 4 and mem["mamba_layers"] == 21 and mem["mlp_layers"] == 17
    assert mem["kv_bytes_per_token"] == 16 * 1024
    assert abs(mem["mamba_state_mb_per_seq"] - 82.6) < 0.1
    assert mem["time_step_limit"][1] == float("inf")
    assert not mem["has_rope_keys"]


def test_queue_claims_are_exclusive_and_resettable(tmp_path):
    index = [{"uid": f"p{i}", "n_tokens": n} for i, n in enumerate([100, 50, 400, 300, 10, 10])]
    q = tmp_path / "q"
    n = workqueue.create_chunks(q, index, chunk_tokens=400, max_prompts=3)
    assert n == 3  # [400] [300+100] [50+10+10]
    assert workqueue.create_chunks(q, index, 400, 3) == n  # não recria
    first = workqueue.claim(q, "gpu0")
    second = workqueue.claim(q, "gpu1")
    assert first[1]["chunk"] != second[1]["chunk"]
    assert first[1]["items"] == [2]  # o mais longo primeiro
    workqueue.complete(first[0])
    assert workqueue.reset_running(q, "gpu1") == 1
    st = workqueue.status(q)
    assert st["done"] == 1 and st["todo"] == 2 and st["run"] == 0 and st["prompts_total"] == 6
    claimed = set()
    while (c := workqueue.claim(q, "gpu2")) is not None:
        claimed.add(c[1]["chunk"])
        workqueue.complete(c[0])
    assert len(claimed) == 2 and workqueue.status(q)["done"] == 3


def test_g3_scope_and_failure_table():
    """<think> em contexto longo (colapso) não reprova o G3; no C0/curto, reprova."""
    from lumi_ctx.analysis import compute_gates, failure_table

    cfg = load_config("stage1.yaml")
    cfg["experiment"]["bootstrap"]["n_resamples"] = 50
    df = _synthetic_df(cfg)
    df["finish_reason"] = "stop"
    long_ = (df.model == "lumi") & (df.length == 261120)
    df.loc[long_, ["think_leak", "strict", "answered_number"]] = [True, 0.0, False]
    df.loc[long_, "finish_reason"] = "length"
    info = {"template_checks": {"lumi": {"ok": True}, "nemotron": {"ok": True}}}
    summary = build_summary(df, cfg)
    g3 = {m: next(g for g in gs if g["id"] == "G3") for m, gs in compute_gates(df, summary, info, cfg).items()}
    assert g3["lumi"]["ok"] and "30 acima" in g3["lumi"]["detalhe"]
    ft = failure_table(df, cfg)
    row = ft[(ft.model == "lumi") & (ft.length == 261120)].iloc[0]
    assert row.com_think == 1.0 and row.cortada_no_limite == 1.0 and row.sem_numero == 1.0
    df.loc[(df.model == "lumi") & (df.task == "C0") & (df.uid == "C0-r00"), "think_leak"] = True
    g3 = next(g for g in compute_gates(df, build_summary(df, cfg), info, cfg)["lumi"] if g["id"] == "G3")
    assert not g3["ok"]


def test_perf_table_discards_incomplete_decode():
    from lumi_ctx.analysis import perf_table

    cfg = load_config("stage1.yaml")
    rows = [
        {"model": "lumi", "length": 2048, "n_tokens": 2020, "rep": 0, "prefill_s": 0.04, "gen_tokens": 8, "decode_tok_s": 7e9},
        {"model": "lumi", "length": 4096, "n_tokens": 4080, "rep": 0, "prefill_s": 0.05, "gen_tokens": 129, "decode_tok_s": 400.0},
    ]
    t = perf_table(rows, {}, cfg)
    assert math.isnan(t.loc[t.length == 2048, "decode_tok_s"].iloc[0])
    assert t.loc[t.length == 4096, "decode_tok_s"].iloc[0] == 400.0


def test_gpu_description_tolerates_unavailable_fields():
    # nvidia-smi devolve "[N/A]" na memória de GPUs particionadas (MIG): o relatório não pode quebrar
    from lumi_ctx.analysis import gpu_description

    assert gpu_description([]) == "–"
    assert gpu_description([{"name": "NVIDIA B200", "memory_mib": "183359"}]) == "1× NVIDIA B200 (179 GiB)"
    assert gpu_description([{"name": "NVIDIA B200", "memory_mib": "[N/A]"}] * 2) == "2× NVIDIA B200"


def test_pipeline_with_mig_like_gpu(tmp_path, monkeypatch, tiny_cfg, tiny_tok_dir):
    # Regressão: o setup no servidor falhava na análise com memory_mib = "[N/A]"
    import yaml

    import lumi_ctx.envinfo as ei
    import lumi_ctx.haystack as hs
    import lumi_ctx.runner as rn
    from conftest import ROOT, synthetic_articles

    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_cfg, allow_unicode=True))
    monkeypatch.setattr(rn, "tokenizer_dir", lambda m: str(tiny_tok_dir))
    monkeypatch.setattr(hs, "iter_articles", synthetic_articles)
    monkeypatch.setattr(ei, "gpu_inventory",
                        lambda: [{"index": "0", "name": "NVIDIA B200", "memory_mib": "[N/A]", "driver": "[N/A]"}])
    assert rn.cmd_run("smoke", str(cfg_path), "t_mig", "fake", 2) == 0
    report = (tmp_path / "work" / "results" / "t_mig" / "REPORT.md").read_text()
    assert "1× NVIDIA B200" in report
