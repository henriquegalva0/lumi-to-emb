#!/usr/bin/env python3
"""Resumo de checkpoints de treino: tempo, perda, trajetória no dev e resultado final.

Só biblioteca padrão: roda na sessão de login. Lê checkpoints/<ID>/summary.json e train_log.jsonl.
Uso (na raiz do clone):
  python3 jobs/embed/ckpt_summary.py "C1-sweep-*"
  python3 jobs/embed/ckpt_summary.py C1 C2
"""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
pats = sys.argv[1:] or ["*"]
dirs = sorted({d for p in pats for d in (root / "checkpoints").glob(p) if d.is_dir()})
if not dirs:
    sys.exit(f"Nenhum checkpoint casa com {pats} em {root / 'checkpoints'}")
for d in dirs:
    print(f"\n=== {d.name}")
    log = d / "train_log.jsonl"
    recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()] if log.exists() else []
    if recs:
        losses = [r["loss"] for r in recs if "loss" in r]
        sec = [r["sec"] for r in recs if "sec" in r]
        print(f"passos: {recs[-1]['step']} | perda: {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"(mín {min(losses):.4f}) | lr pico: {max(r['lr'] for r in recs):.2e} | "
              f"{sum(sec) / len(sec):.1f} s/passo")
        traj = [(r["step"], r["dev"].get("mean_ndcg@10")) for r in recs if r.get("dev")]
        if traj:
            print("dev nDCG@10 por passo: " + "  ".join(f"{s}:{v:.4f}" for s, v in traj))
    s = d / "summary.json"
    if not s.exists():
        print("(sem summary.json: treino não terminou ou é um checkpoint sem treino)")
        continue
    summ = json.loads(s.read_text())
    dev = summ.get("dev") or {}
    mins = summ.get("train_minutes")
    print(f"final: dev nDCG@10 = {dev.get('mean_ndcg@10')} | {summ.get('n_train')} exemplos | "
          f"{mins} min | kernels: {(summ.get('hub_kernels') or {}).get('active')}")
    for task, m in sorted((dev.get("tasks") or {}).items()):
        print(f"   {task:24s} nDCG@10 {m.get('ndcg@10'):.4f}  MRR@10 {m.get('mrr@10'):.4f}  "
              f"R@10 {m.get('recall@10'):.4f}  (n={m.get('n')}, pool={m.get('pool')})")
