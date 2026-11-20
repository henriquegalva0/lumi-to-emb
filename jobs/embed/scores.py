#!/usr/bin/env python3
"""Notas de todos os modelos avaliados: MTEB-BR (média das 22), MTEB-PT por família e a suíte smoke.

Só biblioteca padrão: roda na sessão de login, lendo o cache do mteb (results/embed/mteb). A nota de
cada tarefa é a média do main_score sobre os splits/subconjuntos, como no relatório. A média do
MTEB-BR só aparece com as 22 tarefas (senão, "— (n/22)").
Uso (na raiz do clone):  python3 jobs/embed/scores.py [filtro-no-nome-do-modelo]
"""

import json
import statistics
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
cache = root / "results" / "embed" / "mteb" / "results"
FAM = {
    "STS": ["Assin2STS", "SICK-BR-STS", "STSBenchmarkMultilingualSTS"],
    "Cls": ["MassiveIntentClassification", "BrazilianToxicTweetsClassification",
            "HateSpeechPortugueseClassification", "MultiHateClassification", "TweetSentimentClassification"],
    "Ret": ["MultiLongDocRetrieval", "WikipediaRetrievalMultilingual", "WebFAQRetrieval"],
    "Rer": ["MultiLongDocReranking", "WikipediaRerankingMultilingual", "XGlueWPRReranking"],
}
SMOKE = ["FaqBacenRetrieval", "AssinSTS", "Assin2STS", "SICK-BR-STS"]


def scores(model_dir: Path) -> dict[str, float]:
    out = {}
    for f in model_dir.rglob("*.json"):
        if f.name == "model_meta.json":
            continue
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        v = [e["main_score"] for es in (j.get("scores") or {}).values() if isinstance(es, list)
             for e in es if isinstance(e, dict) and "main_score" in e]
        if v:
            out[j.get("task_name", f.stem)] = sum(v) / len(v)
    return out


br_file = root / "results" / "embed" / "suites" / "mteb_br.json"
br = json.loads(br_file.read_text()) if br_file.exists() else []
filt = sys.argv[1] if len(sys.argv) > 1 else ""
if not cache.exists():
    sys.exit(f"Sem resultados em {cache}")
print(f"{'modelo':42s} {'MTEB-BR':>14s}  " + "  ".join(f"{k:>5s}" for k in FAM) + "   smoke (" + ", ".join(SMOKE) + ")")
for d in sorted(p for p in cache.iterdir() if p.is_dir() and filt in p.name):
    s = scores(d)
    got = [t for t in br if t in s]
    mbr = f"{100 * statistics.mean(s[t] for t in got):.2f}" if br and len(got) == len(br) else f"— ({len(got)}/{len(br)})"
    fams = "  ".join(f"{100 * statistics.mean(s[t] for t in ts):5.1f}" if all(t in s for t in ts) else f"{'—':>5s}"
                     for ts in FAM.values())
    smoke = " ".join(f"{100 * s[t]:.1f}" if t in s else "—" for t in SMOKE)
    print(f"{d.name:42s} {mbr:>14s}  {fams}   {smoke}")
