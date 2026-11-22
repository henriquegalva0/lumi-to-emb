#!/usr/bin/env python3
"""Cria checkpoints/lumi-emb a partir do relatório do merge, pela regra fixada antes dos testes.

Regra (seção 4.5 do plano; docs/embed/PIPELINE.md): o modelo publicado é o melhor no dev entre os
merges e os checkpoints finais. Se o merge escolhido empata ou supera o melhor checkpoint individual
no dev, o lumi-emb é o merge; senão, é o melhor individual. Os benchmarks NÃO entram na escolha
(escolher pelo MTEB-BR seria selecionar no próprio teste que reportamos).

Monta checkpoints/lumi-emb/model a partir de <origem>/model e grava checkpoints/lumi-emb/choice.json com
a proveniência. Os arquivos de pesos (>= 64 MB) entram como hard links: instantâneo e sem ocupar disco
(copiar 8 GB na sessão de login morre se a conexão cair). Nada escreve nesses arquivos depois: o merge
e o --force apagam e recriam a pasta, o que só remove o link. Os arquivos pequenos são copiados, para o
meta.json do lumi-emb não alterar o da origem. Só biblioteca padrão: roda na sessão de login.
Uso (na raiz do clone):  python3 jobs/embed/choose_release.py [MERGE=M-C] [NOME=lumi-emb] [--force]
"""

import json
import os
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
args = [a for a in sys.argv[1:] if a != "--force"]
force = "--force" in sys.argv[1:]
merge = args[0] if args else "M-C"
name = args[1] if len(args) > 1 else "lumi-emb"
ck = root / "checkpoints"
rep_file = ck / merge / "merge_report.json"
if not rep_file.exists():
    sys.exit(f"Sem {rep_file}: o merge {merge} ainda não terminou.")
r = json.loads(rep_file.read_text())
src = merge if r["merge_beats_best_individual_on_dev"] else r["best_individual"]
src_model = ck / src / "model"
if not (src_model / "config.json").exists():
    sys.exit(f"Checkpoint de origem incompleto: {src_model}")
out = ck / name
if out.exists() and not force:
    sys.exit(f"{out} já existe. Confira {out / 'choice.json'}; para refazer, rode de novo com --force.")


def link_or_copy(s: str, d: str) -> str:
    if os.path.getsize(s) >= 64 << 20:
        try:
            os.link(s, d)
            return d
        except OSError:  # outro sistema de arquivos ou sem suporte: copia
            pass
    return shutil.copy2(s, d)


dev = r["dev"]
choice = {
    "name": name,
    "source": src,
    "merge": merge,
    "merge_selected_variant": r["selected"],
    "best_individual": r["best_individual"],
    "rule": "merge se dev(merge) >= dev(melhor individual); senão o melhor individual (dev mean nDCG@10, "
            "desempate Borda). Benchmarks não entram na escolha.",
    "dev_mean_ndcg@10": {"merge": dev[r["selected"]]["mean_ndcg@10"],
                         "best_individual": dev[r["best_individual"]]["mean_ndcg@10"]},
    "merge_report": str(rep_file.relative_to(root)),
}
tmp = ck / f"{name}.tmp"
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir()
shutil.copytree(src_model, tmp / "model", copy_function=link_or_copy)
meta_file = tmp / "model" / "meta.json"
meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
meta["release"] = {k: choice[k] for k in ("name", "source", "merge_selected_variant", "rule")}
meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
(tmp / "choice.json").write_text(json.dumps(choice, indent=2, ensure_ascii=False) + "\n")
if (ck / src / "summary.json").exists():
    shutil.copy(ck / src / "summary.json", tmp / "summary.json")
if out.exists():
    shutil.rmtree(out)
tmp.rename(out)
label = f"{src} ({r['selected']})" if src == merge else src
print(f"{name} = {label}")
print(f"dev: merge {choice['dev_mean_ndcg@10']['merge']:.4f} | melhor individual {r['best_individual']} "
      f"{choice['dev_mean_ndcg@10']['best_individual']:.4f}")
print(f"gravado em {out}")
