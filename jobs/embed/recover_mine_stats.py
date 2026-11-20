#!/usr/bin/env python3
"""Recria data/embed/mined/stats_<família>.json a partir dos logs da mineração.

Jobs de mineração anteriores à retomada por família só gravavam as estatísticas no fim. Este script
lê as linhas `Minerando <família>` / `{'in': ...}` de todos os logs e grava o arquivo de cada família
cujo .jsonl existe (a linha de estatísticas só aparece depois de o arquivo ser gravado por completo).
Assim um novo `mine` pula essas famílias. Só biblioteca padrão: roda na sessão de login.

Uso (na raiz do clone):  python3 jobs/embed/recover_mine_stats.py
"""

import ast
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
mined = root / "data" / "embed" / "mined"
found = 0
for log in sorted((root / "logs").glob("run_embed_mine_*.log")):
    fam = None
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"Minerando (\w+):", line)
        if m:
            fam = m.group(1)
            continue
        m = re.search(r"  (\{'in'.*\})\s*$", line)
        if m and fam and (mined / f"{fam}.jsonl").exists():
            p = mined / f"stats_{fam}.json"
            if not p.exists():
                p.write_text(json.dumps(ast.literal_eval(m.group(1)), indent=2) + "\n")
                print(f"recuperado: {fam} ({log.name})")
                found += 1
for p in sorted(mined.glob("stats_*.json")):
    print(f"pronto: {p.stem[6:]} -> {json.loads(p.read_text()).get('out')} exemplos")
if not found:
    print("nada a recuperar")
