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
# Categorias do MTEB-BR (mteb_pt.TASKS_BY_CATEGORY, v2.0.0), para ver onde estão as diferenças.
BR_CAT = {
    "Cls": ["HateBR", "ToxSynPT", "FactckBrClassification", "PortuLexRRIP"],
    "MLCls": ["BrighterEmotionMultilabelClassification"],
    "Pair": ["AssinRTE", "InferBR"],
    "STS": ["AssinSTS", "Assin2STS"],
    "Clust": ["WikipediaPTCategoriesClusteringP2P", "MedPTClustering", "JurisTCUClusteringP2P",
              "SciELOClusteringP2P", "StackoverflowPtClustering"],
    "Ret": ["Quati", "JurisTCU", "BRTaxQAR", "FaQuADIR", "MedPTRetrieval", "FaqBacenRetrieval"],
    "Rer": ["QuatiReranking", "JurisTCUReranking"],
}


def fam_cols(s: dict[str, float], groups: dict[str, list[str]], width: int) -> str:
    return "  ".join(f"{100 * statistics.mean(s[t] for t in ts):{width}.1f}" if all(t in s for t in ts)
                     else f"{'—':>{width}s}" for ts in groups.values())


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
models = [(d.name, scores(d)) for d in sorted(p for p in cache.iterdir() if p.is_dir() and filt in p.name)]
print("MTEB-BR (média das 22) | MTEB-PT por família | smoke")
print(f"{'modelo':42s} {'MTEB-BR':>14s}  " + "  ".join(f"{k:>5s}" for k in FAM) + "   smoke (" + ", ".join(SMOKE) + ")")
for name, s in models:
    got = [t for t in br if t in s]
    mbr = f"{100 * statistics.mean(s[t] for t in got):.2f}" if br and len(got) == len(br) else f"— ({len(got)}/{len(br)})"
    smoke = " ".join(f"{100 * s[t]:.1f}" if t in s else "—" for t in SMOKE)
    print(f"{name:42s} {mbr:>14s}  {fam_cols(s, FAM, 5)}   {smoke}")
print("\nMTEB-BR por categoria (média das tarefas de cada uma; entre parênteses, quantas)")
print(f"{'modelo':42s} " + "  ".join(f"{k + f'({len(v)})':>8s}" for k, v in BR_CAT.items()))
for name, s in models:
    print(f"{name:42s} {fam_cols(s, BR_CAT, 8)}")
