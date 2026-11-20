"""Linha de comando do Lumi-Embed: python -m lumi_embed <comando> (normalmente via run_embed.sh)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback

from .utils import EmbedError, deep_merge, load_config, results_dir, setup_logging, wpath

log = logging.getLogger("lumi_embed")


def base_cfg(*extra: str) -> dict:
    cfg = load_config("base.yaml")
    for e in extra:
        cfg = deep_merge(cfg, load_config(e))
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lumi_embed", description="Lumi-4B → embedder PT-BR (Etapas 2–5)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="pipeline de dados")
    p.add_argument("step", choices=["sources", "synth", "decontam", "mine", "mix"])
    p.add_argument("--only", default=None, help="sources: lista separada por vírgulas")
    p.add_argument("--generator", default=None, help="synth: chave do gerador (configs/embed/data.yaml)")

    sub.add_parser("export-base", help="exporta o lumi-4b sem treino como checkpoint R0")
    p = sub.add_parser("verify", help="T1–T4 num checkpoint real")
    p.add_argument("ckpt")
    sub.add_parser("bench", help="throughput com e sem kernels do Hub")

    p = sub.add_parser("train", help="treina um checkpoint (configs/embed/train/<ID>.yaml)")
    p.add_argument("ckpt")
    p.add_argument("--max-minutes", type=float, default=None)

    p = sub.add_parser("merge", help="grade de merges e escolha no dev (configs/embed/merge/<ID>.yaml)")
    p.add_argument("name")

    p = sub.add_parser("eval", help="avalia um checkpoint nas suítes do mteb")
    p.add_argument("ckpt")
    p.add_argument("--suite", action="append", required=True)

    p = sub.add_parser("baseline", help="avalia um modelo de comparação (Tabela 14)")
    p.add_argument("hf_id")
    p.add_argument("--suite", action="append", required=True)

    sub.add_parser("report", help="gera as Tabelas 12–15 e atualiza o model card")

    p = sub.add_parser("publish", help="envia um checkpoint ao Hub (branch próprio ou main)")
    p.add_argument("ckpt")
    p.add_argument("--release", action="store_true")
    sub.add_parser("card", help="envia o model card e a documentação ao Hub")

    args = ap.parse_args(argv)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = getattr(args, "ckpt", None) or getattr(args, "name", None) or getattr(args, "step", None) or ""
    setup_logging(wpath("logs", "embed", f"{args.cmd}_{tag}_{ts}.log".replace("/", "_")))
    try:
        if args.cmd == "data":
            cfg = base_cfg("data.yaml")
            if args.step == "sources":
                from .data.sources import run_sources

                run_sources(cfg, args.only.split(",") if args.only else None)
            elif args.step == "synth":
                from .data.synth import run_synth

                gens = [args.generator] if args.generator else list(cfg["synth"]["generators"])
                for g in gens:
                    log.info("%s", run_synth(cfg, g))
            elif args.step == "decontam":
                from .data.decontaminate import run_decontaminate

                run_decontaminate(cfg)
            elif args.step == "mine":
                from .data.mine import run_mine

                run_mine(cfg)
            elif args.step == "mix":
                from .data.mix import run_mix

                run_mix(cfg)
        elif args.cmd == "export-base":
            from .verify import export_base

            export_base(base_cfg())
        elif args.cmd == "verify":
            from .verify import verify_checkpoint

            verify_checkpoint(args.ckpt)
        elif args.cmd == "bench":
            from .verify import bench

            res = bench(base_cfg())
            from .utils import write_json

            write_json(results_dir("bench.json"), res)
        elif args.cmd == "train":
            cfg = load_config(f"train/{args.ckpt}.yaml")
            if cfg["train"].get("stage") == "s0":
                from .train.mntp import train_mntp

                train_mntp(cfg, args.ckpt, args.max_minutes)
            else:
                from .eval.dev import dev_eval
                from .train.contrastive import train

                train(cfg, args.ckpt, max_minutes=args.max_minutes, dev_eval_fn=dev_eval)
        elif args.cmd == "merge":
            from .merge.run import run_merge_grid

            run_merge_grid(load_config(f"merge/{args.name}.yaml"))
        elif args.cmd == "eval":
            from .eval.run import evaluate_checkpoint

            evaluate_checkpoint(base_cfg("eval.yaml"), args.ckpt, args.suite)
        elif args.cmd == "baseline":
            from .eval.run import evaluate_baseline

            evaluate_baseline(base_cfg("eval.yaml"), args.hf_id, args.suite)
        elif args.cmd == "report":
            from .eval.report import build_report

            print(build_report(base_cfg("eval.yaml")["eval"]))
        elif args.cmd == "publish":
            from .publish import publish_checkpoint

            publish_checkpoint(base_cfg()["hub"]["repo"], args.ckpt, release=args.release)
        elif args.cmd == "card":
            from .publish import push_card

            push_card(base_cfg()["hub"]["repo"])
    except EmbedError as exc:
        log.error("ERRO: %s", exc)
        print(f"\nERRO: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception:  # noqa: BLE001
        log.error("Erro inesperado:\n%s", traceback.format_exc())
        return 1
    if args.cmd == "data" and args.step in VLLM_STEPS:
        _exit_now(0)
    return 0


# Etapas que sobem um motor do vLLM. Ao encerrar o processo, a desmontagem do motor de embeddings
# aborta ("terminate called without an active exception", core dump) DEPOIS de tudo gravado, e o job
# aparece como falho. Com o trabalho concluído e os logs gravados, o processo sai direto.
VLLM_STEPS = ("synth", "mine")


def _exit_now(code: int) -> None:
    import os

    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
