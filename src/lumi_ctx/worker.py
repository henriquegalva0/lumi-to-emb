"""Worker de geração: um processo por GPU e por modelo.

Lançado pelo orquestrador (runner.py) com CUDA_VISIBLE_DEVICES já definido.
Carrega o motor, opcionalmente mede desempenho, e consome a fila até esvaziar.
Cada lote concluído é gravado de forma atômica; se o processo cair, o lote em
andamento volta para a fila.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback

import numpy as np

from .common import in_workdir, read_json, results_dir, setup_logging, work_dir, write_json, write_jsonl
from .engine import ENGINE_INIT_FAILED, FakeEngine
from .prompts import load_prompt_set, load_tokenizer
from . import workqueue

log = logging.getLogger("lumi_ctx.worker")


def pick_perf_prompt(index: list[dict], length: int) -> dict | None:
    cands = [r for r in index if r["task"] == "S" and r["length"] == length]
    if not cands:
        return None
    cands.sort(key=lambda r: (abs((r["depth"] or 0) - 0.5), r["rep"]))
    return cands[0]


def run_perf(engine, pset, mm, cfg: dict, model_key: str, label: str, out_path) -> None:
    """Latência isolada (lote de 1): prefill = gerar 1 token; decode = tokens/s depois do 1º."""
    reps = int(cfg["perf"]["reps"])
    n_dec = int(cfg["perf"]["decode_tokens"])
    rows = []
    lengths = cfg["experiment"]["lengths"]
    warm = False
    for length in lengths:
        row = pick_perf_prompt(pset.index, length)
        if row is None:
            continue
        ids = pset.ids(row, mm)
        if not warm:
            engine.generate([ids], [1])
            engine.generate([ids], [n_dec + 1], ignore_eos=True)
            warm = True
        for r in range(reps):
            t0 = time.perf_counter()
            engine.generate([ids], [1])
            t1 = time.perf_counter()
            res = engine.generate([ids], [n_dec + 1], ignore_eos=True)[0]
            t2 = time.perf_counter()
            n_gen = len(res.token_ids) or (n_dec + 1)
            prefill = t1 - t0
            decode_time = max((t2 - t1) - prefill, 1e-9)
            rows.append(
                {
                    "model": model_key, "worker": label, "length": length, "n_tokens": row["n_tokens"],
                    "uid": row["uid"], "rep": r, "prefill_s": prefill, "total_s": t2 - t1,
                    "gen_tokens": n_gen, "decode_tok_s": (n_gen - 1) / decode_time,
                }
            )
            log.info("perf L=%d rep=%d prefill=%.3fs decode=%.1f tok/s", length, r, prefill, rows[-1]["decode_tok_s"])
    write_jsonl(out_path, rows)


def _maybe_simulate_crash() -> None:
    """Só para os testes: LUMI_TEST_CRASH_ONCE=<arquivo> derruba um worker uma única vez."""
    flag = os.environ.get("LUMI_TEST_CRASH_ONCE")
    if not flag:
        return
    try:
        os.remove(flag)
    except FileNotFoundError:
        return
    log.error("Falha simulada (teste) com um lote em andamento")
    os._exit(1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--profile", default="engine", choices=["engine", "fallback_engine"])
    ap.add_argument("--perf", action="store_true")
    args = ap.parse_args(argv)

    wdir = work_dir(args.run_id)
    setup_logging(results_dir(args.run_id) / "logs" / f"worker_{args.model}_{args.label}.log")
    plan = read_json(wdir / "plan.json")
    import yaml

    cfg = yaml.safe_load((results_dir(args.run_id) / "config_used.yaml").read_text(encoding="utf-8"))
    model = next(m for m in cfg["models"] if m["key"] == args.model)
    mplan = plan["models"][args.model]
    pset = load_prompt_set(in_workdir(mplan["prompt_dir"]))
    mm = np.memmap(pset.tokens_path, dtype=np.int32, mode="r")
    qdir = wdir / "queue" / args.model
    odir = wdir / "outputs" / args.model
    odir.mkdir(parents=True, exist_ok=True)

    try:
        if plan["backend"] == "fake":
            tok = load_tokenizer(mplan["tok_path"])
            engine = FakeEngine(model, cfg, tok, profile=args.profile)
        else:
            from .engine import VLLMEngine

            engine = VLLMEngine(model, cfg, mplan["stop_ids"], profile=args.profile)
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        log.error("Falha ao iniciar o motor (%s):\n%s", args.profile, err)
        write_json(wdir / f"engine_fail_{args.model}_{args.label}_{args.profile}.json", {"error": err})
        return ENGINE_INIT_FAILED
    write_json(wdir / f"engine_info_{args.model}_{args.label}.json", engine.info())

    if args.perf and cfg["perf"].get("enabled", True):
        try:
            run_perf(engine, pset, mm, cfg, args.model, args.label, wdir / "perf" / f"{args.model}.jsonl")
        except Exception:  # noqa: BLE001 - desempenho é secundário; segue com a acurácia
            log.error("Benchmark de desempenho falhou; seguindo sem ele:\n%s", traceback.format_exc())
            write_json(wdir / "perf" / f"{args.model}.failed.json", {"error": traceback.format_exc()})

    max_tokens = int(cfg["generation"]["max_tokens"])
    max_tokens_mv = int(cfg["generation"]["max_tokens_mv"])
    n_done = 0
    while True:
        claimed = workqueue.claim(qdir, args.label)
        if claimed is None:
            break
        run_path, chunk = claimed
        _maybe_simulate_crash()
        rows = [pset.index[i] for i in chunk["items"]]
        prompts = [pset.ids(r, mm) for r in rows]
        limits = [max_tokens_mv if r["task"] == "MV" else max_tokens for r in rows]
        t0 = time.time()
        results = engine.generate(prompts, limits)
        dt = time.time() - t0
        records = []
        for r, res in zip(rows, results):
            if res.n_prompt and res.n_prompt != r["n_tokens"]:
                log.warning("%s: motor recebeu %d tokens (esperado %d)", r["uid"], res.n_prompt, r["n_tokens"])
            records.append(
                {
                    "uid": r["uid"], "model": args.model, "worker": args.label, "chunk": chunk["chunk"],
                    "text": res.text, "token_ids": res.token_ids, "finish_reason": res.finish_reason,
                    "n_prompt_engine": res.n_prompt, "n_out": len(res.token_ids), "metrics": res.metrics,
                }
            )
        write_jsonl(odir / f"{chunk['chunk']}.jsonl", records)
        write_json(
            odir / f"{chunk['chunk']}.meta.json",
            {"chunk": chunk["chunk"], "worker": args.label, "seconds": dt, "n_prompts": len(rows),
             "prompt_tokens": chunk["tokens"], "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        )
        workqueue.complete(run_path)
        n_done += len(rows)
        log.info(
            "lote %s: %d prompts, %.1fM tokens em %.1fs (%.0f tok/s)",
            chunk["chunk"], len(rows), chunk["tokens"] / 1e6, dt, chunk["tokens"] / max(dt, 1e-9),
        )
    log.info("Worker %s/%s terminou: %d prompts", args.model, args.label, n_done)
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
