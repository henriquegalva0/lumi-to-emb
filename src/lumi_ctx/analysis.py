"""Análise de uma execução: summary.csv, L*, comparação pareada, figuras, perf.csv,
REPORT.md e DESVIOS.md. Roda sem GPU (dá para refazer localmente com `analyze`)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import plots
from .common import CONFIG_DIR, load_config, read_json, read_jsonl, results_dir, stable_hash, write_json
from .envinfo import Deviations
from .prompts import enumerate_specs

log = logging.getLogger("lumi_ctx.analysis")

TASK_ORDER = {"C0": 0, "S": 1, "MK": 2, "MV": 3}


# --------------------------------------------------------------------------- modelo


def _decode_float(x):
    if isinstance(x, dict) and "__float__" in x:
        return float(x["__float__"])
    return x


def model_memory(config_path: Path) -> dict:
    """Memória teórica a partir do config.json (KV cache das camadas de atenção e estado Mamba)."""
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    pattern = raw.get("hybrid_override_pattern", "")
    n_attn, n_mamba, n_mlp = pattern.count("*"), pattern.count("M"), pattern.count("-")
    head_dim = raw.get("head_dim") or raw.get("attention_head_dim")
    kv = n_attn * 2 * raw["num_key_value_heads"] * head_dim * 2  # K e V em bf16
    ssm = n_mamba * raw["mamba_num_heads"] * raw["mamba_head_dim"] * raw["ssm_state_size"] * 4  # fp32
    max_pos = raw.get("max_position_embeddings")
    tsl = raw.get("time_step_limit")
    return {
        "layers": len(pattern),
        "attention_layers": n_attn,
        "mamba_layers": n_mamba,
        "mlp_layers": n_mlp,
        "kv_heads": raw["num_key_value_heads"],
        "head_dim": head_dim,
        "kv_bytes_per_token": kv,
        "kv_gib_at_max_pos": round(kv * max_pos / 2**30, 2) if max_pos else None,
        "mamba_state_mb_per_seq": round(ssm / 1e6, 1),
        "max_position_embeddings": max_pos,
        "has_rope_keys": any(k.startswith("rope") for k in raw),
        "time_step_limit": [_decode_float(v) for v in tsl] if isinstance(tsl, list) else tsl,
    }


# --------------------------------------------------------------------------- estatística


def bootstrap_ci(values, strata, n_resamples: int, seed: int) -> tuple[float, float]:
    """IC 95 % percentil, com reamostragem estratificada (por profundidade)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return math.nan, math.nan
    s = np.asarray(strata, dtype=float)  # código numérico do estrato (sem NaN)
    if np.isnan(s).any():
        raise ValueError("estratos com NaN")
    rng = np.random.default_rng(seed)
    total = np.zeros(n_resamples)
    for u in np.unique(s):
        g = v[s == u]
        idx = rng.integers(0, len(g), size=(n_resamples, len(g)))
        total += g[idx].sum(axis=1)
    means = total / len(v)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(pos: int, neg: int) -> float:
    """Teste exato bilateral (McNemar exato quando as notas são 0/1)."""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def l_star(acc_by_len: dict, lengths: list[int], thr: float) -> int | None:
    """Maior L com acurácia >= thr nele e em todos os comprimentos menores."""
    best = None
    for L in lengths:
        a = acc_by_len.get(L)
        if a is None or not (a >= thr):
            break
        best = L
    return best


def _seed(*parts) -> int:
    return int(stable_hash(list(parts))[:8], 16)


# --------------------------------------------------------------------------- tabelas


def load_results(rdir: Path) -> pd.DataFrame:
    df = pd.DataFrame(list(read_jsonl(rdir / "raw" / "results.jsonl")))
    for col in ("strict", "lenient", "think_leak", "answered_number", "hit_distractor", "mv_all", "mv_extra", "n_out"):
        if col not in df:
            df[col] = np.nan
    return df


def build_summary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    model_order = {m["key"]: i for i, m in enumerate(cfg["models"])}
    n_res = int(cfg["experiment"]["bootstrap"]["n_resamples"])
    bseed = int(cfg["experiment"]["bootstrap"]["seed"])
    rows = []
    for (model, task, length), g in df.groupby(["model", "task", "length"], sort=False):
        ok = g[g["status"] == "ok"]
        strata = ok["depth"].astype(float).fillna(-1.0).to_numpy()  # C0 não tem profundidade
        lo, hi = bootstrap_ci(ok["strict"], strata, n_res, _seed(bseed, model, task, length, "s"))
        llo, lhi = bootstrap_ci(ok["lenient"], strata, n_res, _seed(bseed, model, task, length, "l"))
        row = {
            "model": model,
            "model_label": g["model_label"].iloc[0],
            "task": task,
            "length": int(length),
            "n": len(g),
            "n_missing": int((g["status"] != "ok").sum()),
            "mean_tokens": round(float(g["n_tokens"].mean()), 1),
            "max_abs_len_dev": float(g["len_dev"].abs().max()) if task != "C0" else math.nan,
            "acc_strict": float(ok["strict"].mean()) if len(ok) else math.nan,
            "ci95_low": lo,
            "ci95_high": hi,
            "acc_lenient": float(ok["lenient"].mean()) if len(ok) else math.nan,
            "lenient_ci95_low": llo,
            "lenient_ci95_high": lhi,
            "min_depth_acc_strict": math.nan,
            "worst_depth": math.nan,
            "think_leak_rate": float(ok["think_leak"].astype(float).mean()) if len(ok) else math.nan,
            "no_number_rate": float(1 - ok["answered_number"].astype(float).mean()) if len(ok) else math.nan,
            "distractor_rate": float(ok["hit_distractor"].astype(float).mean()) if task == "MK" and len(ok) else math.nan,
            "mv_all_values_rate": float(ok["mv_all"].astype(float).mean()) if task == "MV" and len(ok) else math.nan,
            "mv_extra_numbers_mean": float(ok["mv_extra"].astype(float).mean()) if task == "MV" and len(ok) else math.nan,
            "finish_length_rate": float((ok["finish_reason"] == "length").mean()) if len(ok) else math.nan,
            "mean_out_tokens": float(ok["n_out"].astype(float).mean()) if len(ok) else math.nan,
        }
        if task != "C0" and len(ok):
            by_depth = ok.groupby("depth")["strict"].mean()
            row["min_depth_acc_strict"] = float(by_depth.min())
            row["worst_depth"] = float(by_depth.idxmin())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_m"] = out["model"].map(model_order)
    out["_t"] = out["task"].map(TASK_ORDER)
    return out.sort_values(["_m", "_t", "length"]).drop(columns=["_m", "_t"]).reset_index(drop=True)


def depth_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df[(df["task"] == "S") & (df["status"] == "ok")]
    if s.empty:
        return pd.DataFrame(columns=["model", "length", "depth", "n", "acc_strict", "acc_lenient"])
    t = s.groupby(["model", "length", "depth"]).agg(
        n=("strict", "size"), acc_strict=("strict", "mean"), acc_lenient=("lenient", "mean"),
        depth_actual_mean=("target_depth_actual", "mean"),
    )
    return t.reset_index()


def lstar_table(summary: pd.DataFrame, cfg: dict) -> dict:
    lengths = list(cfg["experiment"]["lengths"])
    thr = float(cfg["experiment"]["threshold"])
    out = {}
    for m in cfg["models"]:
        res = {"label": m["label"], "role": m["role"]}
        sm = summary[summary["model"] == m["key"]]
        for task in ("S", "MK", "MV"):
            t = sm[sm["task"] == task]
            if t.empty:
                continue
            acc = dict(zip(t["length"], t["acc_strict"]))
            res[f"L_{task}"] = l_star(acc, lengths, thr)
            res[f"L_{task}_ci"] = l_star(dict(zip(t["length"], t["ci95_low"])), lengths, thr)
            res[f"L_{task}_lenient"] = l_star(dict(zip(t["length"], t["acc_lenient"])), lengths, thr)
            if task == "S":
                res["L_S_strict_depth"] = l_star(dict(zip(t["length"], t["min_depth_acc_strict"])), lengths, thr)
            res[f"acc_{task}"] = {int(k): float(v) for k, v in acc.items()}
        c0 = sm[sm["task"] == "C0"]
        res["C0"] = float(c0["acc_strict"].iloc[0]) if len(c0) else None
        out[m["key"]] = res
    return out


def paired_tests(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    roles = {m["role"]: m["key"] for m in cfg["models"]}
    a, b = roles.get("alvo"), roles.get("controle")
    if not a or not b:
        return pd.DataFrame()
    ok = df[df["status"] == "ok"]
    pa = ok[ok["model"] == a].set_index("uid")
    pb = ok[ok["model"] == b].set_index("uid")
    common = pa.index.intersection(pb.index)
    rows = []
    for (task, length), uids in pa.loc[common].groupby(["task", "length"]).groups.items():
        x = pa.loc[uids, "strict"].astype(float).to_numpy()
        y = pb.loc[uids, "strict"].astype(float).to_numpy()
        d = x - y
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        rows.append({
            "task": task, "length": int(length), "n_pairs": len(uids),
            f"acc_{a}": float(x.mean()), f"acc_{b}": float(y.mean()), "delta": float(x.mean() - y.mean()),
            f"melhor_{a}": pos, f"melhor_{b}": neg, "p_value": sign_test_p(pos, neg),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_t"] = out["task"].map(TASK_ORDER)
    return out.sort_values(["_t", "length"]).drop(columns="_t").reset_index(drop=True)


def failure_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Como o modelo erra, por comprimento (S, MK e MV juntos)."""
    ok = df[(df["status"] == "ok") & (df["task"] != "C0")].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["wrong"] = ok["strict"].astype(float) < 1.0
    ok["cut"] = ok["finish_reason"] == "length"
    ok["no_num"] = ~ok["answered_number"].astype(bool)
    ok["think"] = ok["think_leak"].astype(bool)
    ok["distr"] = ok["hit_distractor"].fillna(0).astype(float) > 0
    t = ok.groupby(["model", "length"]).agg(
        n=("uid", "size"), erro=("wrong", "mean"), cortada_no_limite=("cut", "mean"),
        sem_numero=("no_num", "mean"), com_think=("think", "mean"), numero_distrator=("distr", "mean"),
    ).reset_index()
    order = {m["key"]: i for i, m in enumerate(cfg["models"])}
    t["_m"] = t["model"].map(order)
    return t.sort_values(["_m", "length"]).drop(columns="_m").reset_index(drop=True)


def perf_table(perf_rows: list[dict], info: dict, cfg: dict) -> pd.DataFrame:
    if not perf_rows:
        return pd.DataFrame()
    p = pd.DataFrame(perf_rows)
    need = int(cfg.get("perf", {}).get("decode_tokens", 128)) + 1
    bad = (p["gen_tokens"] < need) | ~np.isfinite(p["decode_tok_s"].astype(float)) | (p["decode_tok_s"] > 1e6)
    p.loc[bad, "decode_tok_s"] = np.nan  # medida incompleta (parou antes) ou degenerada
    t = p.groupby(["model", "length"]).agg(
        n_tokens=("n_tokens", "first"), reps=("rep", "size"),
        prefill_s=("prefill_s", "median"), prefill_s_min=("prefill_s", "min"), prefill_s_max=("prefill_s", "max"),
        decode_tok_s=("decode_tok_s", "median"),
    ).reset_index()
    t["prefill_tok_s"] = t["n_tokens"] / t["prefill_s"]
    mem = info.get("model_memory", {})
    t["kv_cache_gib_theoretical"] = [
        round(mem.get(m, {}).get("kv_bytes_per_token", 16384) * n / 2**30, 3) for m, n in zip(t["model"], t["n_tokens"])
    ]
    t["mamba_state_mb_theoretical"] = [mem.get(m, {}).get("mamba_state_mb_per_seq") for m in t["model"]]
    order = {m["key"]: i for i, m in enumerate(cfg["models"])}
    t["_m"] = t["model"].map(order)
    return t.sort_values(["_m", "length"]).drop(columns="_m").reset_index(drop=True)


# --------------------------------------------------------------------------- gates


def compute_gates(df: pd.DataFrame, summary: pd.DataFrame, info: dict, cfg: dict) -> dict:
    tol = cfg["experiment"]["length_tolerance"]
    c0_gate = cfg["experiment"]["c0_gate"]
    gates: dict[str, list[dict]] = {}
    for m in cfg["models"]:
        k = m["key"]
        g = df[df["model"] == k]
        ok = g[g["status"] == "ok"]
        tmpl = info.get("template_checks", {}).get(k, {})
        stats = info.get("prompt_sets", {}).get(k, {}).get("stats", {})
        c0 = summary[(summary["model"] == k) & (summary["task"] == "C0")]
        c0_acc = float(c0["acc_strict"].iloc[0]) if len(c0) else math.nan
        think_max = int(cfg["experiment"].get("think_gate_max_length", 8192))
        leak = ok["think_leak"].astype(bool) if len(ok) else pd.Series(dtype=bool)
        in_scope = (ok["task"] == "C0") | (ok["length"] <= think_max) if len(ok) else pd.Series(dtype=bool)
        leaks = int((leak & in_scope).sum()) if len(ok) else 0
        leaks_out = int((leak & ~in_scope).sum()) if len(ok) else 0
        dev = g.loc[g["task"] != "C0", "len_dev"].abs()
        over = int((g.loc[g["task"] != "C0", "n_tokens"] > g.loc[g["task"] != "C0", "length"]).sum())
        missing = int((g["status"] != "ok").sum())
        gates[k] = [
            {"id": "G1", "nome": "Template com thinking desligado", "ok": bool(tmpl.get("ok")),
             "detalhe": f"prompt termina em {tmpl.get('generation_suffix_off', '?')[-30:]!r}"},
            {"id": "G2", "nome": f"C0 (sem palheiro) ≥ {c0_gate:.0%}", "ok": bool(c0_acc >= c0_gate),
             "detalhe": f"acurácia estrita {c0_acc:.3f}" if not math.isnan(c0_acc) else "sem C0"},
            {"id": "G3", "nome": f"Nenhum <think> no C0 nem em L ≤ {plots.fmt_len(think_max)}", "ok": leaks == 0,
             "detalhe": f"{leaks} no escopo; {leaks_out} acima de {plots.fmt_len(think_max)} (modo de falha, seção 2)"},
            {"id": "G4", "nome": f"Comprimento dentro de ±{tol:.0%} e ≤ L", "ok": bool((dev <= tol).all()) and over == 0,
             "detalhe": f"desvio máx. {dev.max():.4f}; {over} acima do alvo" if len(dev) else "-"},
            {"id": "G5", "nome": "Número da agulha ausente do palheiro", "ok": True,
             "detalhe": f"verificado na construção; {stats.get('n_value_redraws', 0)} número(s) re-sorteado(s)"},
            {"id": "G6", "nome": "Todas as gerações concluídas", "ok": missing == 0, "detalhe": f"{missing} faltando de {len(g)}"},
        ]
    return gates


def model_valid(gates: list[dict]) -> bool:
    return all(x["ok"] for x in gates if x["id"] in ("G1", "G2", "G3", "G6"))


# --------------------------------------------------------------------------- recomendação


def _int(n) -> str:
    """Inteiro com espaço fino como separador de milhar (não confunde com o ponto decimal)."""
    return f"{int(n):,}".replace(",", "\u202f")


def fmt_L(L) -> str:
    if L is None:
        return "< 2k (não atinge o limiar)"
    return f"{plots.fmt_len(L)} ({_int(L)})"


def _decision(L: int | None) -> tuple[str, str]:
    """Regra de decisão da Etapa 2 (tabela da seção 1 do README)."""
    if L is not None and L >= 32768:
        return "Documento inteiro", f"codificar documentos inteiros até {fmt_L(L)}, com currículo de comprimento até lá"
    if L is not None and L >= 8192:
        return "Chunks de 4k a 8k tokens", "usar *chunking* com janelas de 4k–8k e treinar o embedder nesse regime"
    return "Reavaliar", "nem *chunks* de 8k estão garantidos; rever o desenho (outra base, fine-tuning de contexto ou chunks menores)"


def recommendation(lstars: dict, cfg: dict) -> list[str]:
    roles = {m["role"]: m["key"] for m in cfg["models"]}
    a, b = roles.get("alvo"), roles.get("controle")
    lengths = list(cfg["experiment"]["lengths"])
    lines: list[str] = []
    if not a or a not in lstars:
        return ["Sem resultados do modelo-alvo."]
    la_ = lstars[a]
    L = la_.get("L_S")
    name, what = _decision(L)
    lines.append(f"**Pela regra da especificação (L\\* de S = {fmt_L(L)}): {name}.** Isto é, {what}.")

    # L* efetivo: o menor entre as tarefas (um embedder precisa separar muitos fatos parecidos)
    per_task = {t: la_.get(f"L_{t}") for t in ("S", "MK", "MV") if f"L_{t}" in la_}
    L_eff = min(per_task.values(), key=lambda x: -1 if x is None else x) if per_task else L
    if per_task and (L_eff or 0) < (L or 0):
        worst = [t for t, v in per_task.items() if v == L_eff]
        accs = ", ".join(f"{t} {la_.get(f'acc_{t}', {}).get(L, float('nan')):.2f}" for t in ("MK", "MV") if t in per_task)
        name_eff, what_eff = _decision(L_eff)
        lines.append(
            f"**Ressalva — várias agulhas: L\\* efetivo = {fmt_L(L_eff)} ({'/'.join(worst)}) → {name_eff}.** "
            f"Em {fmt_L(L)} a acurácia já cai ({accs}). Um embedder precisa distinguir muitos fatos parecidos "
            f"no mesmo documento, então o critério mais seguro é o menor L\\* entre as tarefas: {what_eff}."
        )
    # margem: o que acontece no comprimento seguinte ao L*
    if L is not None and L in lengths and lengths.index(L) + 1 < len(lengths):
        nxt = lengths[lengths.index(L) + 1]
        acc_next = la_.get("acc_S", {}).get(nxt)
        if acc_next is not None and acc_next < 0.10:
            lines.append(
                f"**Sem margem acima de L\\*.** Em {plots.fmt_len(nxt)} a acurácia de S cai para {acc_next:.2f}: "
                "é um colapso (saídas degeneradas, ver 'Modos de falha'), não uma degradação gradual. "
                "Documentos pouco acima de L\\* falham por completo."
            )
    if b and b in lstars:
        Lb = lstars[b].get("L_S")
        la, lb = (L or 0), (Lb or 0)
        if lb >= 131072 and la <= 16384:
            lines.append(
                f"**A perda vem do pré-treino contínuo curto.** O controle mantém L\\* = {fmt_L(Lb)}, "
                f"e o Lumi cai logo depois de 8k–16k. A Etapa 4 deve incluir um fine-tuning de contexto longo "
                "antes do treino contrastivo."
            )
        elif lb >= 4 * max(la, 1):
            lines.append(
                f"**Perda grande, atribuível ao pré-treino contínuo.** Controle: L\\* = {fmt_L(Lb)}; Lumi: {fmt_L(L)} "
                f"(~{round(lb / max(la, 1))}× menor), com os mesmos prompts. Como a base mantém o contexto longo, ele deve ser "
                "recuperável: para documentos acima de L\\*, a Etapa 4 precisa de um fine-tuning de contexto longo "
                "antes do treino contrastivo."
            )
        elif lb > la:
            lines.append(
                f"**Degradação parcial em relação à base.** Controle: L\\* = {fmt_L(Lb)}; Lumi: {fmt_L(L)}. "
                "Parte da janela foi perdida no pré-treino contínuo; considerar um fine-tuning de contexto "
                "longo leve se a Etapa 2 escolher documentos inteiros."
            )
        else:
            lines.append(
                f"**O Lumi preserva a janela da base.** Controle: L\\* = {fmt_L(Lb)}; Lumi: {fmt_L(L)}. "
                "Não há indício de perda de contexto longo causada pelo pré-treino contínuo."
            )
    return lines


# --------------------------------------------------------------------------- estimativa (smoke)


def estimate_full(perf: pd.DataFrame, throughput: list[dict], cfg_full: dict, n_models: int) -> pd.DataFrame | None:
    specs = enumerate_specs(cfg_full)
    counts: dict[int, int] = {}
    for s in specs:
        if s["task"] != "C0":
            counts[s["length"]] = counts.get(s["length"], 0) + 1
    rows = []
    overhead_min = 6.0  # carregar modelo + compilação, por modelo e por worker
    if perf is not None and not perf.empty:
        pm = perf.groupby("length")["prefill_s"].median().to_dict()
        if all(L in pm for L in counts):
            sec = sum(counts[L] * pm[L] for L in counts)  # por modelo, lote de 1 (conservador)
            for g in (1, 2, 4, 8):
                h = (n_models * (sec / g / 60 + overhead_min)) / 60
                rows.append({"base": "prefill isolado (conservador)", "gpus": g, "horas": round(h, 2)})
    if throughput:
        tok = sum(r["prompt_tokens"] for r in throughput)
        secs = sum(r["seconds"] for r in throughput)
        if secs > 0:
            rate = tok / secs  # tokens/s por worker, com lotes
            total = sum(L * c for L, c in counts.items())
            for g in (1, 2, 4, 8):
                h = (n_models * (total / rate / g / 60 + overhead_min)) / 60
                rows.append({"base": "vazão em lote do smoke", "gpus": g, "horas": round(h, 2)})
    return pd.DataFrame(rows) if rows else None


# --------------------------------------------------------------------------- relatório


def _pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def gpu_description(gpus: list[dict]) -> str:
    """'N× nome (X GiB)'. O nvidia-smi devolve '[N/A]' em alguns campos (por exemplo, GPUs
    particionadas com MIG); nesse caso a memória é omitida."""
    if not gpus:
        return "–"
    mib = str(gpus[0].get("memory_mib", "")).strip()
    mem = f" ({int(mib) // 1024} GiB)" if mib.isdigit() else ""
    return f"{len(gpus)}× {gpus[0].get('name', '?')}{mem}"


def write_report(rdir: Path, cfg: dict, info: dict, env: dict, summary: pd.DataFrame, lstars: dict,
                 paired: pd.DataFrame, gates: dict, perf: pd.DataFrame, estimate: pd.DataFrame | None,
                 depth_tab: pd.DataFrame, deviations: list[dict], failures: pd.DataFrame | None = None) -> None:
    lengths = list(cfg["experiment"]["lengths"])
    thr = cfg["experiment"]["threshold"]
    models = cfg["models"]
    L = []
    title = f"# Etapa 1 — Janela de contexto: relatório da execução `{info.get('run_id')}`"
    L.append(title + "\n")
    if info.get("backend") == "fake":
        L.append("> ⚠️ **RESULTADOS SIMULADOS** (backend `fake`, sem modelo de verdade). Servem só para testar o pipeline.\n")
    if info.get("mode") == "smoke":
        L.append("> Execução **smoke**: poucas repetições por célula. Confirma que o pipeline funciona em todos os "
                 "comprimentos; os números não substituem a execução completa.\n")

    # 1. L*
    L.append("## 1. L\\* por modelo\n")
    rows = []
    for m in models:
        r = lstars.get(m["key"], {})
        valid = model_valid(gates.get(m["key"], []))
        rows.append([
            m["label"] + ("" if valid else " ⚠️"),
            fmt_L(r.get("L_S")), fmt_L(r.get("L_S_strict_depth")), fmt_L(r.get("L_S_ci")),
            fmt_L(r.get("L_MK")), fmt_L(r.get("L_MV")), _pct(r.get("C0")),
        ])
    L.append(_md_table(["Modelo", "L\\* (S)", "L\\*<sub>estrito</sub> (S)", "L\\* conservador (S)", "L\\* (MK)",
                        "L\\* (MV)", "C0"], rows))
    L.append(
        f"\n- **L\\***: maior L com acurácia estrita média de S ≥ {thr:.2f} nele e em todos os comprimentos menores.\n"
        f"- **L\\*<sub>estrito</sub>**: idem, exigindo ≥ {thr:.2f} em **toda** profundidade.\n"
        f"- **L\\* conservador**: idem, usando o limite inferior do IC 95 % (bootstrap) em vez da média.\n"
        "- MK e MV usam a acurácia estrita média (no MV, a fração dos 4 valores recuperados).\n"
        "- ⚠️ = algum gate essencial (G1, G2, G3 ou G6) falhou para o modelo; o L\\* dele não é confiável.\n"
    )

    # 2. comparação
    L.append("## 2. Lumi × controle\n")
    for task in ("S", "MK", "MV"):
        t = summary[summary["task"] == task]
        if t.empty:
            continue
        hdr = ["L"] + [f"{m['label']}" for m in models]
        has_pair = not paired.empty and (paired["task"] == task).any()
        if has_pair:
            hdr += ["Δ (alvo − controle)", "p (pareado)"]
        rows = []
        for Lx in lengths:
            row = [plots.fmt_len(Lx)]
            for m in models:
                c = t[(t["model"] == m["key"]) & (t["length"] == Lx)]
                if len(c):
                    row.append(f"{c['acc_strict'].iloc[0]:.2f} [{c['ci95_low'].iloc[0]:.2f}, {c['ci95_high'].iloc[0]:.2f}]")
                else:
                    row.append("–")
            if has_pair:
                p = paired[(paired["task"] == task) & (paired["length"] == Lx)]
                row += [f"{p['delta'].iloc[0]:+.2f}", f"{p['p_value'].iloc[0]:.3g}"] if len(p) else ["–", "–"]
            rows.append(row)
        L.append(f"**{plots.TASK_NAME[task]}** — acurácia estrita média [IC 95 %]\n")
        L.append(_md_table(hdr, rows) + "\n")
    L.append("p: teste exato dos sinais sobre os pares (mesmo prompt nos dois modelos); com notas 0/1 equivale ao "
             "McNemar exato. " + ("Os prompts são idênticos token a token nos dois modelos (mesmo tokenizer e template)."
                                  if info.get("paired_identical") else
                                  "Os prompts usam as mesmas sementes, mas não são idênticos token a token.") + "\n")
    if failures is not None and not failures.empty:
        L.append("**Modos de falha** (S, MK e MV juntos; fração dos prompts de cada comprimento)\n")
        rows = []
        for _, r in failures.iterrows():
            label = next(m["label"] for m in models if m["key"] == r["model"])
            rows.append([label, plots.fmt_len(int(r["length"])), int(r["n"]), f"{r['erro']:.2f}",
                         f"{r['cortada_no_limite']:.2f}", f"{r['sem_numero']:.2f}", f"{r['com_think']:.2f}",
                         f"{r['numero_distrator']:.2f}"])
        L.append(_md_table(["Modelo", "L", "n", "Erro (estrito)", "Cortada no limite (laço)", "Sem número",
                            "Com <think>", "Nº distrator (MK)"], rows))
        L.append("\nUma resposta correta tem poucos tokens; ser cortada no limite de `max_tokens` indica laço ou "
                 "degeneração. `<think>` fora do C0/contextos curtos aparece como parte desse colapso, não como erro "
                 "de configuração (o gate G3 cobre o C0 e os contextos curtos). Ver `failure_modes.csv`.\n")
    L.append("![Acurácia vs. comprimento](acc_vs_len.png)\n")
    for m in models:
        L.append(f"![Heatmap S — {m['label']}](heatmap_S_{m['key']}.png)\n")

    # 3. recomendação
    L.append("## 3. Recomendação para a Etapa 2\n")
    for line in recommendation(lstars, cfg):
        L.append(f"- {line}")
    L.append("\nTabela de decisão (§1 da especificação): L\\* ≥ 32k → documento inteiro; L\\* ≈ 8k → chunks de "
             "4k–8k; L\\* < 8k → reavaliar.\n")

    # 4. gates e versões
    L.append("## 4. Gates, versões e duração\n")
    rows = []
    for m in models:
        for gte in gates.get(m["key"], []):
            rows.append([m["label"], gte["id"], gte["nome"], "✅" if gte["ok"] else "❌", gte["detalhe"]])
    L.append(_md_table(["Modelo", "Gate", "Critério", "", "Detalhe"], rows) + "\n")
    pk = env.get("packages", {})
    gpus = env.get("gpus", [])
    gpu_desc = gpu_description(gpus)
    engines = info.get("engine_info", {})
    eng_args = next((e[0].get("args") for e in engines.values() if e), None)
    dur = info.get("duration_s")
    rows = [
        ["Imagem (config / ambiente)", f"`{cfg['vllm'].get('image')}` / `{env.get('image_tag_env') or '–'}`"],
        ["vLLM", pk.get("vllm") or "–"], ["transformers", pk.get("transformers") or "–"],
        ["torch (CUDA)", f"{pk.get('torch') or '–'} ({env.get('torch_cuda') or '–'})"],
        ["Driver NVIDIA (CUDA do driver)", f"{env.get('driver_version') or '–'} ({env.get('driver_cuda_version') or '–'})"],
        ["GPUs usadas", f"{gpu_desc}; workers: {len(info.get('gpus', []))}"],
        ["Modelos", "; ".join(f"`{m['hf_id']}@{m['revision'][:12]}`" for m in models)],
        ["Commit do código", f"`{env.get('git', {}).get('commit', '?')}`"],
        ["Início / fim", f"{info.get('started_at', '?')} / {info.get('finished_at', '?')}"],
        ["Duração total", f"{dur / 3600:.2f} h ({dur / 60:.0f} min)" if dur else "–"],
    ]
    if eng_args:
        rows.append(["Opções do vLLM", "`" + json.dumps({k: v for k, v in eng_args.items()
                                                       if k not in ('model', 'tokenizer', 'revision', 'tokenizer_revision')}) + "`"])
    L.append(_md_table(["Item", "Valor"], rows) + "\n")
    timing = info.get("timing", {})
    if timing:
        L.append("Tempo por fase: " + "; ".join(f"{k} {v / 60:.1f} min" for k, v in timing.items()) + ".\n")

    # 5. desempenho
    L.append("## 5. Desempenho (secundário)\n")
    if perf is not None and not perf.empty:
        rows = []
        for _, r in perf.iterrows():
            label = next(m["label"] for m in models if m["key"] == r["model"])
            rows.append([label, plots.fmt_len(int(r["length"])), f"{r['prefill_s']:.3f}", _int(r["prefill_tok_s"]),
                         "–" if pd.isna(r["decode_tok_s"]) else f"{r['decode_tok_s']:.1f}",
                         f"{r['kv_cache_gib_theoretical']:.2f}"])
        L.append(_md_table(["Modelo", "L", "Prefill (s)", "Prefill (tok/s)", "Decode (tok/s)", "KV teórico (GiB)"], rows))
        L.append("\nMedido com lote de 1 (latência isolada), mediana das repetições; ver `perf.csv`. "
                 "\"–\" = nenhuma medida de decode válida (a geração parou antes dos tokens pedidos).\n")
        L.append("![Prefill](perf.png)\n")
    else:
        L.append("Sem medidas de desempenho nesta execução.\n")
    mem = info.get("model_memory", {})
    if mem:
        k0 = next(iter(mem))
        mm = mem[k0]
        L.append(
            f"Memória teórica ({k0}): {mm['attention_layers']} camadas de atenção × 2 (K, V) × {mm.get('kv_heads', 8)} cabeças KV × "
            f"{mm.get('head_dim', 128)} dimensões × 2 bytes "
            f"= **{mm['kv_bytes_per_token'] / 1024:.0f} KiB/token** (≈ {mm['kv_gib_at_max_pos']} GiB em "
            f"{_int(mm['max_position_embeddings'])} tokens); estado Mamba (fp32) ≈ **{mm['mamba_state_mb_per_seq']} MB por "
            f"sequência**, constante. Chaves `rope_*` no config: {'sim' if mm['has_rope_keys'] else 'não'}.\n"
        )
    if estimate is not None and not estimate.empty:
        L.append("### Estimativa para a execução completa\n")
        rows = [[r["base"], r["gpus"], f"{r['horas']:.1f} h"] for _, r in estimate.iterrows()]
        L.append(_md_table(["Base da estimativa", "GPUs", "Tempo estimado"], rows))
        L.append("\nInclui ~6 min por modelo para carregar o motor. Use a maior das duas estimativas, com folga, "
                 "no `max_time` do job.\n")

    # 6. detalhes
    L.append("## 6. Detalhes do experimento\n")
    pool = info.get("pool", {})
    if pool:
        L.append(f"- Palheiro: `{pool.get('source', {}).get('repo')}` @ `{str(pool.get('source', {}).get('revision'))[:12]}`, "
                 f"{pool.get('n_docs')} artigos, {pool.get('n_tokens', 0) / 1e6:.2f}M tokens, {pool.get('n_sentences')} frases.")
    for k, s in info.get("prompt_sets", {}).items():
        st = s.get("stats", {})
        L.append(f"- Prompts ({k}): {st.get('n_prompts')} | desvio máx. de comprimento {st.get('max_abs_len_dev', 0):.4f} | "
                 f"{st.get('n_resampled', 0)} com início re-sorteado | C0 com ~{st.get('c0_tokens_mean', 0):.0f} tokens.")
    if not depth_tab.empty:
        # no L* de cada modelo: é na borda da capacidade que um efeito de posição apareceria
        cols = []
        for m in models:
            Lm = lstars.get(m["key"], {}).get("L_S")
            if Lm is not None:
                cols.append((m, Lm))
        if cols:
            L.append("\n**Acurácia de S por profundidade no L\\* de cada modelo:**\n")
            depths = sorted(depth_tab["depth"].unique())
            hdr = ["Profundidade"] + [f"{m['label']} @ {plots.fmt_len(Lm)}" for m, Lm in cols]
            rows = []
            for d in depths:
                row = [f"{round(d * 100)}%"]
                for m, Lm in cols:
                    c = depth_tab[(depth_tab["model"] == m["key"]) & (depth_tab["length"] == Lm) & (depth_tab["depth"] == d)]
                    row.append(_pct(float(c["acc_strict"].iloc[0])) if len(c) else "–")
                rows.append(row)
            L.append(_md_table(hdr, rows) + "\n")
    if deviations:
        L.append(f"- {len(deviations)} desvio(s) da especificação registrado(s) em `DESVIOS.md`.")
    L.append("\nArquivos: `summary.csv`, `by_depth_S.csv`, `paired_tests.csv`, `failure_modes.csv`, `perf.csv`, `lstar.json`, "
             "`raw/results.jsonl` (uma linha por prompt e modelo), `config_used.yaml`, `env.json`, `logs/`.\n")
    (rdir / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_deviations(rdir: Path, rows: list[dict]) -> None:
    lines = ["# Desvios em relação à especificação\n"]
    if not rows:
        lines.append("Nenhum desvio registrado.\n")
    for i, r in enumerate(rows, 1):
        lines.append(f"## {i}. {r['what']}\n")
        lines.append(f"- **Motivo:** {r['why']}")
        if r.get("impact"):
            lines.append(f"- **Impacto:** {r['impact']}")
        lines.append(f"- Código: `{r['code']}` · registrado em {r['at']}\n")
    (rdir / "DESVIOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- principal


def analyze_run(run_id: str) -> None:
    rdir = results_dir(run_id)
    cfg = yaml.safe_load((rdir / "config_used.yaml").read_text(encoding="utf-8"))
    info = read_json(rdir / "run_info.json")
    env = read_json(rdir / "env.json") if (rdir / "env.json").exists() else {}
    df = load_results(rdir)
    lengths = list(cfg["experiment"]["lengths"])
    thr = float(cfg["experiment"]["threshold"])

    summary = build_summary(df, cfg)
    summary.to_csv(rdir / "summary.csv", index=False, float_format="%.4f")
    depth_tab = depth_table(df)
    depth_tab.to_csv(rdir / "by_depth_S.csv", index=False, float_format="%.4f")
    paired = paired_tests(df, cfg)
    paired.to_csv(rdir / "paired_tests.csv", index=False, float_format="%.4g")
    lstars = lstar_table(summary, cfg)
    write_json(rdir / "lstar.json", lstars)

    # figuras
    for m in cfg["models"]:
        d = depth_tab[depth_tab["model"] == m["key"]]
        if d.empty:
            continue
        cells = {(float(r.depth), int(r.length)): float(r.acc_strict) for r in d.itertuples()}
        depths = sorted({float(x) for x in d["depth"]})
        plots.heatmap_s(cells, lengths, depths, f"{m['label']} — S (1 agulha): comprimento × profundidade",
                        rdir / f"heatmap_S_{m['key']}.png")
    curves = []
    for i, m in enumerate(cfg["models"]):
        for task in ("S", "MK", "MV"):
            t = summary[(summary["model"] == m["key"]) & (summary["task"] == task)].sort_values("length")
            if t.empty:
                continue
            curves.append({"model_label": m["label"], "model_idx": i, "task": task, "x": t["length"].tolist(),
                           "y": t["acc_strict"].tolist(), "lo": t["ci95_low"].tolist(), "hi": t["ci95_high"].tolist()})
    if curves:
        plots.acc_vs_len(curves, lengths, thr, "Acurácia estrita por comprimento — S, MK e MV",
                         rdir / "acc_vs_len.png")
        plots.acc_vs_len_panels(curves, lengths, thr, rdir / "acc_vs_len_panels.png")

    perf_rows = list(read_jsonl(rdir / "raw" / "perf.jsonl")) if (rdir / "raw" / "perf.jsonl").exists() else []
    perf = perf_table(perf_rows, info, cfg)
    if not perf.empty:
        perf.to_csv(rdir / "perf.csv", index=False, float_format="%.4f")
        series = []
        for i, m in enumerate(cfg["models"]):
            p = perf[perf["model"] == m["key"]]
            if len(p):
                series.append({"model_label": m["label"], "model_idx": i, "x": p["length"].tolist(),
                               "prefill_s": p["prefill_s"].tolist()})
        plots.perf_plot(series, lengths, rdir / "perf.png")
    else:
        pd.DataFrame(columns=["model", "length", "prefill_s", "decode_tok_s"]).to_csv(rdir / "perf.csv", index=False)

    throughput = list(read_jsonl(rdir / "raw" / "throughput.jsonl")) if (rdir / "raw" / "throughput.jsonl").exists() else []
    estimate = None
    if info.get("mode") == "smoke":
        try:
            estimate = estimate_full(perf, throughput, load_config(CONFIG_DIR / "stage1.yaml"), len(cfg["models"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("Não foi possível estimar a execução completa: %s", exc)

    failures = failure_table(df, cfg)
    failures.to_csv(rdir / "failure_modes.csv", index=False, float_format="%.4f")
    gates = compute_gates(df, summary, info, cfg)
    write_json(rdir / "gates.json", gates)
    dev = Deviations(rdir / "deviations.jsonl")
    think_max = int(cfg["experiment"].get("think_gate_max_length", 8192))
    ok_rows = df[df["status"] == "ok"]
    out_scope = ok_rows[(ok_rows["task"] != "C0") & (ok_rows["length"] > think_max)]
    n_out = int(out_scope["think_leak"].astype(bool).sum())
    if n_out:
        lens = [plots.fmt_len(x) for x in sorted({int(x) for x in out_scope.loc[out_scope["think_leak"].astype(bool), "length"]})]
        dev.add(
            "think-gate-scope",
            f"Gate G3 restrito ao C0 e a L ≤ {plots.fmt_len(think_max)}; {n_out} saída(s) com <think> acima disso "
            f"({', '.join(lens)}) contadas como modo de falha.",
            "A especificação trata <think> na saída como erro de configuração, mas o template está verificado (G1) "
            "e os <think> só aparecem em contextos longos, em saídas degeneradas (laços até max_tokens).",
            "Nenhum sobre a acurácia: essas saídas já pontuam 0. A taxa aparece em 'Modos de falha' no relatório.",
        )
    missing = int((df["status"] != "ok").sum())
    if missing:
        dev.add("missing-outputs", f"{missing} prompt(s) sem saída do modelo.", "Falha de worker sem nova tentativa.",
                "Células afetadas têm n menor; ver n_missing em summary.csv.")
    relaxed = int(df["gap_relaxed"].astype(bool).sum()) if "gap_relaxed" in df else 0
    if relaxed:
        dev.add("needle-gap", f"{relaxed} prompt(s) com distância mínima entre agulhas reduzida à metade.",
                "Palheiro curto para o espaçamento pedido.", "Pequeno.")
    deviations = dev.rows()
    write_deviations(rdir, deviations)
    write_report(rdir, cfg, info, env, summary, lstars, paired, gates, perf, estimate, depth_tab, deviations, failures)

    for m in cfg["models"]:
        r = lstars.get(m["key"], {})
        log.info("%s: L*(S)=%s | L*_estrito=%s | L*(MK)=%s | L*(MV)=%s | C0=%s", m["label"], r.get("L_S"),
                 r.get("L_S_strict_depth"), r.get("L_MK"), r.get("L_MV"), _pct(r.get("C0")))
        for gte in gates.get(m["key"], []):
            if not gte["ok"]:
                log.warning("Gate %s (%s) FALHOU para %s: %s", gte["id"], gte["nome"], m["label"], gte["detalhe"])
    log.info("Relatório: %s", rdir / "REPORT.md")
