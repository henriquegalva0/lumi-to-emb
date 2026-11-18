"""Tabelas de resultados (seção 5.4–5.6 do plano), geradas só a partir dos arquivos de resultado.

Regra A.3: nenhuma célula do Lumi-Embed é preenchida sem um JSON de resultado correspondente;
células sem resultado ficam "—". Os únicos números digitados aqui são valores da literatura
já presentes no plano, sempre com a referência entre colchetes.

Saídas: results/embed/tables/*.csv e docs/embed/RESULTS.md; o bloco entre os marcadores
<!-- RESULTS:START --> e <!-- RESULTS:END --> do model card (docs/huggingface/README.md) é
regenerado.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path

from ..utils import REPO_ROOT, ckpt_dir, results_dir, wpath
from .names import model_name
from .stats import bootstrap_mean, load_scores, paired_bootstrap

log = logging.getLogger("lumi_embed.report")

CHECKPOINTS = ["R0", "R0e", "R1", "C1", "C2", "C2-real", "C2-syn", "C2-nohn", "C2-naivehn", "C2-ib", "C2-direct",
               "C3-long", "B2a", "B2f", "M-C", "M-B", "lumi-emb"]
MTEB_BR_CATEGORIES = {
    "Classificação (4)": ["HateBR", "ToxSynPT", "FactckBrClassification", "PortuLexRRIP"],
    "Multilabel (1)": ["BrighterEmotionMultilabelClassification"],
    "Pares (2)": ["AssinRTE", "InferBR"],
    "STS (2)": ["AssinSTS", "Assin2STS"],
    "Clusterização (5)": ["WikipediaPTCategoriesClusteringP2P", "MedPTClustering", "JurisTCUClusteringP2P",
                          "SciELOClusteringP2P", "StackoverflowPtClustering"],
    "Retrieval (6)": ["Quati", "JurisTCU", "BRTaxQAR", "FaQuADIR", "MedPTRetrieval", "FaqBacenRetrieval"],
    "Reranking (2)": ["QuatiReranking", "JurisTCUReranking"],
}
MTEB_BR_TASKS = [t for ts in MTEB_BR_CATEGORIES.values() for t in ts]
MTEB_PT_FAMILIES = {
    "PT STS": ["Assin2STS", "SICK-BR-STS", "STSBenchmarkMultilingualSTS"],
    "PT Cls": ["MassiveIntentClassification", "BrazilianToxicTweetsClassification",
               "HateSpeechPortugueseClassification", "MultiHateClassification", "TweetSentimentClassification"],
    "PT Ret": ["MultiLongDocRetrieval", "WikipediaRetrievalMultilingual", "WebFAQRetrieval"],
    "PT Rer": ["MultiLongDocReranking", "WikipediaRerankingMultilingual", "XGlueWPRReranking"],
}
ABLATIONS = [
    ("A1", "O estágio S1 é necessário?", "C2", "C2-direct"),
    ("A2", "Dados reais (mesma escala)", "C2", "C2-real"),
    ("A2", "Dados sintéticos (mesma escala)", "C2", "C2-syn"),
    ("A3", "Sem negativos difíceis", "C2", "C2-nohn"),
    ("A3", "Top-k ingênuo", "C2", "C2-naivehn"),
    ("A4", "In-batch + máscara de falso negativo", "C2", "C2-ib"),
    ("A5", "Bidirecional (atenção)", "B2a", "C2"),
    ("A5", "Bidirecional (atenção + BiMamba)", "B2f", "C2"),
    ("A7", "Echo embeddings sem treino", "R0e", "R0"),
    ("A9", "Merge causal vs C2", "M-C", "C2"),
    ("A13", "Contexto longo", "C3-long", "C2"),
]
# Valores da literatura reproduzidos do plano (Tabelas 8–10 e 14), com a fonte.
LITERATURE = [
    {"model": "Qwen3-Embedding-4B", "rel": "Mesmo porte, Transformer, líder aberto em PT", "mteb_br": "0,6621 [16]",
     "mteb_pt": "a medir", "mmteb": "a medir", "longembed": "a medir", "hf": "Qwen/Qwen3-Embedding-4B"},
    {"model": "Qwen3-Embedding-8B", "rel": "Líder aberto no MTEB-BR", "mteb_br": "0,6704 [16]", "mteb_pt": "a medir",
     "mmteb": "70,58 [10]", "longembed": "a medir", "hf": "Qwen/Qwen3-Embedding-8B"},
    {"model": "Qwen3-Embedding-0.6B", "rel": "Baseline pequeno", "mteb_br": "a medir",
     "mteb_pt": "80,7 / 52,3 / 78,9 / 85,8 [17]", "mmteb": "64,33 [9]", "longembed": "a medir",
     "hf": "Qwen/Qwen3-Embedding-0.6B"},
    {"model": "multilingual-e5-large-instruct", "rel": "Baseline multilíngue clássico", "mteb_br": "a medir",
     "mteb_pt": "81,3 / 53,0 / 70,1 / 84,5 [17]", "mmteb": "a medir", "longembed": "a medir",
     "hf": "intfloat/multilingual-e5-large-instruct"},
    {"model": "gte-multilingual-base", "rel": "Baseline multilíngue", "mteb_br": "a medir",
     "mteb_pt": "80,4 / 50,2 / 76,9 / 84,2 [17]", "mmteb": "a medir", "longembed": "a medir",
     "hf": "Alibaba-NLP/gte-multilingual-base"},
    {"model": "llama-embed-nemotron-8b", "rel": "Mesma família NVIDIA; receita aberta", "mteb_br": "a medir",
     "mteb_pt": "a medir", "mmteb": "69,46 [10]", "longembed": "a medir", "hf": "nvidia/llama-embed-nemotron-8b"},
    {"model": "Mamba2 1.3B / Codestral Mamba2 7B", "rel": "Recorrentes puros", "mteb_br": "—", "mteb_pt": "—",
     "mmteb": "55,2 / 59,4 (eixo tarefa) [5]", "longembed": "40,8 / 44,5 [5]", "hf": None},
]


def _fmt(x, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.2f}" if pct else f"{x:.4f}"


def _ci(b: dict) -> str:
    if not b or b.get("n", 0) == 0:
        return "—"
    return f"{b['mean']:.4f} [{b['lo']:.4f}, {b['hi']:.4f}]"


def _mean(scores: dict, tasks: list[str], require_all: bool = True):
    vals = [scores[t] for t in tasks if t in scores]
    if not vals or (require_all and len(vals) < len(tasks)):
        return None
    return sum(vals) / len(vals)


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)


def collect(extra_models: list[str] | None = None) -> dict[str, dict]:
    cache = results_dir("mteb")
    all_scores = {c: load_scores(cache, model_name(c)) for c in CHECKPOINTS}
    for m in extra_models or []:
        all_scores[m] = load_scores(cache, m)
    return all_scores


def build_report(ecfg: dict) -> str:
    baselines = [m["hf"] for m in LITERATURE if m["hf"]]
    S = collect(baselines)
    tables = results_dir("tables")
    md = ["# Lumi-Embed — resultados\n",
          "Gerado por `lumi_embed report` a partir de `results/embed/`. Células \"—\" = sem resultado ainda. "
          "IC 95 % por bootstrap sobre tarefas; p-valores por bootstrap pareado (unidade = tarefa), como no "
          "MTEB-BR [16].\n"]

    # Tabela 12 — visão geral (listas de tarefas gravadas pela própria avaliação)
    lists = {}
    for suite in ("mmteb", "eng_v2", "longembed"):
        f = results_dir("suites", f"{suite}.json")
        if f.exists():
            lists[suite] = json.loads(f.read_text())
    h12 = ["Checkpoint", "MTEB-BR Mean22 [IC 95%]", "PT STS", "PT Cls", "PT Ret", "PT Rer", "MMTEB", "MTEB eng v2",
           "LongEmbed"]
    rows12 = []
    for c in CHECKPOINTS:
        s = S[c]
        br = [s[t] for t in MTEB_BR_TASKS if t in s]
        mean22 = bootstrap_mean(br) if len(br) == len(MTEB_BR_TASKS) else {"n": 0}
        rows12.append([c, _ci(mean22)] + [_fmt(_mean(s, ts)) for ts in MTEB_PT_FAMILIES.values()]
                      + [_fmt(_mean(s, lists.get(k))) if lists.get(k) else "—" for k in ("mmteb", "eng_v2", "longembed")])
    _write_csv(tables / "table12_overview.csv", h12, rows12)
    md += ["## Tabela 12 — Visão geral\n", _table(h12, rows12), ""]

    # Tabela 13 — MTEB-BR por categoria
    h13 = ["Checkpoint"] + list(MTEB_BR_CATEGORIES)
    rows13 = [[c] + [_fmt(_mean(S[c], ts)) for ts in MTEB_BR_CATEGORIES.values()] for c in CHECKPOINTS]
    _write_csv(tables / "table13_mteb_br_categories.csv", h13, rows13)
    md += ["## Tabela 13 — MTEB-BR por categoria\n", _table(h13, rows13), ""]

    # Tabela 13b — por tarefa (CSV completo; no markdown, só alguns checkpoints)
    h13b = ["Tarefa"] + CHECKPOINTS
    rows13b = [[t] + [_fmt(S[c].get(t)) for c in CHECKPOINTS] for t in MTEB_BR_TASKS]
    _write_csv(tables / "table13b_mteb_br_tasks.csv", h13b, rows13b)
    keep = ["R0", "R1", "C2", "B2a", "B2f", "M-C", "M-B", "lumi-emb"]
    md += ["## Tabela 13b — MTEB-BR por tarefa (seleção; completo em CSV)\n",
           _table(["Tarefa"] + keep, [[t] + [_fmt(S[c].get(t)) for c in keep] for t in MTEB_BR_TASKS]), ""]

    # Tabela 14 — comparação com modelos correspondentes
    h14 = ["Modelo", "Relação com o lumi-emb", "MTEB-BR Mean22", "MTEB-PT (STS/Cls/Ret/Rer)", "MMTEB", "LongEmbed"]
    rows14 = []
    for c in ("lumi-emb", "R0"):
        s = S[c]
        fam = " / ".join(_fmt(_mean(s, ts), pct=True) for ts in MTEB_PT_FAMILIES.values())
        rows14.append([f"{c} (este trabalho)", "—" if c == "lumi-emb" else "Ponto de partida",
                       _fmt(_mean(s, MTEB_BR_TASKS)), fam, "—", "—"])
    for m in LITERATURE:
        s = S.get(m["hf"], {}) if m["hf"] else {}
        measured_br = _mean(s, MTEB_BR_TASKS)
        measured_pt = [_mean(s, ts) for ts in MTEB_PT_FAMILIES.values()]
        br = m["mteb_br"] if measured_br is None else f"{measured_br:.4f} (medido)"
        pt = m["mteb_pt"] if all(v is None for v in measured_pt) else " / ".join(
            _fmt(v, pct=True) for v in measured_pt) + " (medido)"
        rows14.append([m["model"], m["rel"], br, pt, m["mmteb"], m["longembed"]])
    _write_csv(tables / "table14_comparison.csv", h14, rows14)
    md += ["## Tabela 14 — Comparação com modelos correspondentes\n", _table(h14, rows14),
           "\nValores com referência entre colchetes vêm da literatura (ver o plano); \"(medido)\" = avaliado aqui "
           "com o mesmo protocolo; \"a medir\" = ainda não avaliado.\n"]

    # Tabela 15 — ablações (Δ no MTEB-BR, bootstrap pareado sobre as tarefas)
    h15 = ["ID", "Pergunta", "Comparação", "Δ Mean22 [IC 95%]", "p", "Tarefas"]
    rows15 = []
    for aid, q, a, b in ABLATIONS:
        sa = {t: v for t, v in S[a].items() if t in MTEB_BR_TASKS}
        sb = {t: v for t, v in S[b].items() if t in MTEB_BR_TASKS}
        r = paired_bootstrap(sa, sb)
        cell = "—" if r["n"] == 0 else f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
        rows15.append([aid, q, f"{a} − {b}", cell, "—" if r["n"] == 0 else f"{r['p']:.3f}", r["n"]])
    _write_csv(tables / "table15_ablations.csv", h15, rows15)
    md += ["## Tabela 15 — Ablações (MTEB-BR)\n", _table(h15, rows15), ""]

    # Dev (seleção de checkpoints e merges)
    hdev, rdev = ["Checkpoint", "dev nDCG@10 médio", "Passos", "Minutos de treino"], []
    for c in CHECKPOINTS:
        f = ckpt_dir(c) / "summary.json"
        if f.exists():
            js = json.loads(f.read_text())
            rdev.append([c, _fmt((js.get("dev") or {}).get("mean_ndcg@10")), js.get("steps", "—"),
                         js.get("train_minutes", "—")])
    if rdev:
        md += ["## Dev set (seleção de checkpoints)\n", _table(hdev, rdev), ""]
    text = "\n".join(md) + "\n"
    out = REPO_ROOT / "docs" / "embed" / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    (wpath("results", "embed", "RESULTS.md")).write_text(text, encoding="utf-8")
    update_card_block(text)
    return text


def update_card_block(results_md: str) -> None:
    card = REPO_ROOT / "docs" / "huggingface" / "README.md"
    if not card.exists():
        return
    t = card.read_text(encoding="utf-8")
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in t and b in t:
        body = results_md.split("\n", 2)[-1].replace("\n## ", "\n#### ")  # sem o título; abaixo de "### Resultados"
        body = "### Resultados\n" + body
        t = t.split(a)[0] + a + "\n" + body + "\n" + b + t.split(b)[1]
        card.write_text(t, encoding="utf-8")
