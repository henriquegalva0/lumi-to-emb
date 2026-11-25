#!/usr/bin/env python3
"""Apaga as versões de um merge que não viraram o modelo final, para liberar disco (~8 GB cada).

O merge grava todas as versões da grade (média, SLERP, interpolação com a base). Depois da escolha no
dev, só checkpoints/<MERGE>/model é usado. As notas de todas as versões ficam no merge_report.json, e
cada versão pode ser refeita a partir dos candidatos, porque o merge é determinístico. O script:
- só mexe nas versões listadas no merge_report.json (nunca nos candidatos, no próprio merge ou nos releases);
- só apaga a versão escolhida se checkpoints/<MERGE>/model tiver os mesmos arquivos de pesos (nomes e tamanhos);
- mostra o que faria e quanto disco libera (arquivos com hard links em outro lugar não liberam nada);
- só apaga com --apply, e registra o que apagou em checkpoints/<MERGE>/pruned.json.
Só biblioteca padrão: roda na sessão de login (apagar é instantâneo).
Uso (na raiz do clone):  python3 jobs/embed/prune_merge.py MERGE [MERGE ...] [--apply]
"""

import json
import shutil
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
ck = root / "checkpoints"
apply = "--apply" in sys.argv[1:]
merges = [a for a in sys.argv[1:] if a != "--apply"]
if not merges:
    sys.exit(__doc__)


def weights(d: Path) -> dict[str, int]:
    return {f.name: f.stat().st_size for f in d.glob("*.safetensors")}


def freed(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and f.stat().st_nlink == 1)


total = 0.0
for m in merges:
    rep = ck / m / "merge_report.json"
    if not rep.exists():
        print(f"{m}: sem {rep} (o merge não terminou); nada a fazer.")
        continue
    r = json.loads(rep.read_text())
    final = ck / m / "model"
    print(f"{m} (escolhida: {r['selected']}; fica em {final})")
    done = []
    for v in (x for x in r["dev"] if x.startswith(f"{m}-")):
        d = ck / v
        if not d.exists():
            continue
        if v == r["selected"]:
            w = weights(d / "model")
            if not w or weights(final) != w:
                print(f"  {v}: é a escolhida e {final} não tem os mesmos pesos; mantida.")
                continue
        gb = freed(d) / 1e9
        total += gb
        print(f"  {v}: {'apagada' if apply else 'seria apagada'} ({gb:.1f} GB; dev {r['dev'][v]['mean_ndcg@10']})")
        if apply:
            shutil.rmtree(d)
            done.append(v)
    if done:
        log = ck / m / "pruned.json"
        hist = json.loads(log.read_text()) if log.exists() else []
        hist.append({"when": time.strftime("%Y-%m-%d %H:%M"), "removed": done})
        log.write_text(json.dumps(hist, indent=2) + "\n")
print(f"Total: {total:.1f} GB " + ("liberados." if apply else "a liberar. Para apagar, rode de novo com --apply."))
