"""Linha de comando: python -m lumi_ctx <comando> (normalmente chamada pelo run_stage1.sh)."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback

from .common import Stage1Error


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lumi_ctx", description="Etapa 1: teste de janela de contexto (NIAH PT-BR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="confere ambiente, tokenizers, template e acesso ao Hub (sem GPU)")
    p.add_argument("--config", default="stage1.yaml")

    p = sub.add_parser("prepare", help="monta o palheiro e os prompts (CPU)")
    p.add_argument("--config", default="stage1.yaml")

    p = sub.add_parser("run", help="execução smoke ou completa")
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--config", default=None, help="padrão: smoke.yaml ou stage1.yaml")
    p.add_argument("--run-id", default=None, help="reusar um run_id existente retoma a execução")
    p.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    p.add_argument("--fake-gpus", type=int, default=2, help="workers simulados no backend fake")
    p.add_argument("--resume-latest", action="store_true", help="retoma a última execução incompleta do mesmo modo")

    p = sub.add_parser("analyze", help="refaz a análise de uma execução (CPU)")
    p.add_argument("--run-id", required=True)

    args = ap.parse_args(argv)
    from . import runner

    try:
        if args.cmd == "check":
            return runner.cmd_check(args.config)
        if args.cmd == "prepare":
            return runner.cmd_prepare(args.config)
        if args.cmd == "run":
            return runner.cmd_run(args.mode, args.config, args.run_id, args.backend, args.fake_gpus, args.resume_latest)
        if args.cmd == "analyze":
            return runner.cmd_analyze(args.run_id)
    except Stage1Error as exc:
        logging.getLogger("lumi_ctx").error("ERRO: %s", exc)
        print(f"\nERRO: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
        return 130
    except Exception:  # noqa: BLE001
        logging.getLogger("lumi_ctx").error("Erro inesperado:\n%s", traceback.format_exc())
        print("\nErro inesperado; veja o traceback acima e rode './run_stage1.sh diagnose'.", file=sys.stderr)
        return 1
    return 0
