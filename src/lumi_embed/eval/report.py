"""Tabelas de resultados (seção 5.4–5.6 do plano), geradas só a partir dos arquivos de resultado.

Regra A.3: nenhuma célula do Lumi-Embed é preenchida sem um JSON de resultado correspondente;
células sem resultado ficam "—". Os únicos números digitados aqui são valores da literatura
já presentes no plano, sempre com a referência entre colchetes. Notas × 100.

Saídas: results/embed/tables/*.csv, docs/embed/RESULTS.md (completo) e, no model card
(docs/huggingface/README.md), o bloco entre <!-- RESULTS:START --> e <!-- RESULTS:END -->,
regenerado numa versão resumida (só linhas com resultado). Os destaques (bloco <!-- HIGHLIGHTS -->
do README e do card), as lições das ablações (bloco <!-- ABLATIONS -->) e as figuras
(docs/embed/figures/) saem das mesmas notas.

O lumi-emb é uma cópia de um checkpoint (checkpoints/lumi-emb/choice.json). Ele é avaliado com o
próprio nome; a origem (ex.: M-C) recebe as mesmas notas, porque os pesos são os mesmos.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from pathlib import Path

from ..utils import REPO_ROOT, ckpt_dir, results_dir, wpath
from .names import model_name
from .stats import bootstrap_mean, load_scores, paired_bootstrap

log = logging.getLogger("lumi_embed.report")

CHECKPOINTS = ["R0", "R0e", "R1", "C1", "C2", "C2-s2", "C2-s3", "C2-real", "C2-syn", "C2-nohn", "C2-naivehn",
               "C2-ib", "C2-direct", "C3-long", "B2a", "B2f", "M-C", "M-B", "M-L", "lumi-emb", "lumi-emb-long"]
RELEASES = ["lumi-emb", "lumi-emb-long"]  # cópias de um checkpoint escolhido (choice.json)
LONG_TAG = "L8192"                         # sufixo do cache do MLDR-pt em 8k (eval.yaml, mldr_pt_long)
LONG_MAX = 8192                            # maior comprimento do currículo do estágio longo (train/_long.yaml)
# Modelo no main do Hub (o mesmo de hub.release em configs/embed/base.yaml). As comparações pareadas e os
# destaques usam ele quando já tem as 22 tarefas do MTEB-BR; senão, o lumi-emb.
MAIN = "lumi-emb-long"
SEEDS = ["C2", "C2-s2", "C2-s3"]  # mesma receita, sementes 1–3: variação entre execuções
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
EXTRA_SUITES = {"mmteb": "MMTEB", "eng_v2": "MTEB eng v2", "longembed": "LongEmbed"}
# Idiomas de cada suíte: as notas do MTEB-BR/MTEB-PT usam só os subconjuntos em português.
PT = ("por",)
SUITE_LANGS = {"mmteb": None, "eng_v2": ("eng",), "longembed": ("eng",)}
ABLATIONS = [
    ("A1", "O estágio S1 é necessário?", "C2", "C2-direct"),
    ("S2", "O S2 (negativos difíceis) melhora o C1?", "C2", "C1"),
    ("A2", "Só dados reais no S2", "C2", "C2-real"),
    ("A2", "Só dados sintéticos no S2", "C2", "C2-syn"),
    ("A2", "Reais contra sintéticos", "C2-real", "C2-syn"),
    ("A3", "Sem negativos difíceis", "C2", "C2-nohn"),
    ("A3", "Top-k ingênuo", "C2", "C2-naivehn"),
    ("A4", "In-batch + máscara de falso negativo", "C2", "C2-ib"),
    ("A5", "Bidirecional (atenção)", "B2a", "C2"),
    ("A5", "Bidirecional (atenção + BiMamba)", "B2f", "C2"),
    ("A7", "Echo embeddings sem treino", "R0e", "R0"),
    ("A9", "Merge causal vs C2", "M-C", "C2"),
    ("A13", "Contexto longo", "C3-long", "C2"),
]
QWEN4B = "Qwen/Qwen3-Embedding-4B"
QWEN8B = "Qwen/Qwen3-Embedding-8B"
# Valores da literatura reproduzidos do plano (Tabelas 8–10 e 14), com a fonte (× 100).
LITERATURE = [
    {"model": "Qwen3-Embedding-4B", "rel": "Mesmo porte, Transformer, líder aberto em PT", "mteb_br": "66.21 [16]",
     "mteb_pt": "a medir", "mmteb": "a medir", "longembed": "a medir", "hf": QWEN4B},
    {"model": "Qwen3-Embedding-8B", "rel": "Líder aberto no MTEB-BR", "mteb_br": "67.04 [16]", "mteb_pt": "a medir",
     "mmteb": "70.58 [10]", "longembed": "a medir", "hf": "Qwen/Qwen3-Embedding-8B"},
    {"model": "Qwen3-Embedding-0.6B", "rel": "Baseline pequeno", "mteb_br": "a medir",
     "mteb_pt": "80.7 / 52.3 / 78.9 / 85.8 [17]", "mmteb": "64.33 [9]", "longembed": "a medir",
     "hf": "Qwen/Qwen3-Embedding-0.6B"},
    {"model": "multilingual-e5-large-instruct", "rel": "Baseline multilíngue clássico", "mteb_br": "a medir",
     "mteb_pt": "81.3 / 53.0 / 70.1 / 84.5 [17]", "mmteb": "a medir", "longembed": "a medir",
     "hf": "intfloat/multilingual-e5-large-instruct"},
    {"model": "gte-multilingual-base", "rel": "Baseline multilíngue", "mteb_br": "a medir",
     "mteb_pt": "80.4 / 50.2 / 76.9 / 84.2 [17]", "mmteb": "a medir", "longembed": "a medir",
     "hf": "Alibaba-NLP/gte-multilingual-base"},
    {"model": "llama-embed-nemotron-8b", "rel": "Mesma família NVIDIA; receita aberta", "mteb_br": "a medir",
     "mteb_pt": "a medir", "mmteb": "69.46 [10]", "longembed": "a medir", "hf": "nvidia/llama-embed-nemotron-8b"},
    {"model": "Mamba2 1.3B / Codestral Mamba2 7B", "rel": "Recorrentes puros", "mteb_br": "—", "mteb_pt": "—",
     "mmteb": "55.2 / 59.4 (eixo tarefa) [5]", "longembed": "40.8 / 44.5 [5]", "hf": None},
]
BASELINE_NAMES = {m["hf"]: m["model"] for m in LITERATURE if m["hf"]}
MLDR_HEAD = ["Modelo", "MLDR-pt, 512 tokens", "MLDR-pt, 8k tokens"]
MLDR_NOTE = ("MultiLongDocRetrieval em português (documentos longos da Wikipédia e do mC4). No MTEB-PT ele é "
             "avaliado com 512 tokens, como todo o protocolo; aqui, também com os documentos em até 8k tokens.")
CONTAM_NOTE = ("A auditoria posterior (`results/embed/contamination/audit.md`) achou textos de treino que "
               "coincidem com textos destas tarefas, que o índice usado no filtro não cobria: {tasks}. São no "
               "máximo 0,1 % dos exemplos de cada conjunto. Para medir o efeito, a média é refeita sem essas "
               "tarefas, e a comparação pareada é repetida.")
A2_NOTE = ("No A2, os conjuntos do S2 não têm o mesmo tamanho: S2-real e S2-syn ficaram menores que o S2 "
           "completo e diferentes entre si (tamanhos em `data/embed/sets/manifest.json` e no DATA_CARD). A versão "
           "na mesma escala, prevista no plano, ainda não rodou.")
# Lições das ablações (bloco ABLATIONS do README e do card). Δ = com o que foi testado − sem ele.
LESSONS = [
    ("Estágio S1 (pares fracos em grande escala) antes do S2", "C2", "C2-direct"),
    ("Negativos difíceis filtrados (PercPos) em vez do top-k ingênuo", "C2", "C2-naivehn"),
    ("Dados reais e sintéticos juntos no S2, em vez de só sintéticos¹", "C2", "C2-syn"),
    ("Dados reais e sintéticos juntos no S2, em vez de só reais¹", "C2", "C2-real"),
    ("Merge de checkpoints (M-C) em vez do C2 sozinho", "M-C", "C2"),
    ("4 negativos difíceis por exemplo no S2, em vez de só os do lote", "C2", "C2-nohn"),
    ("Negativos do lote somados aos difíceis, com máscara de falso negativo", "C2-ib", "C2"),
    ("Atenção bidirecional (Ba, com MNTP antes)", "B2a", "C2"),
    ("Atenção bidirecional + BiMamba (Bf, com MNTP antes)", "B2f", "C2"),
    ("Echo embeddings (texto repetido), sem treino", "R0e", "R0"),
]
RESULTS_URL = "https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/RESULTS.md"
NA = {"n": 0}


# ---------------------------------------------------------------- formatação e médias
def _pct(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{nd}f}"


def _ci(b: dict) -> str:
    if not b or b.get("n", 0) == 0:
        return "—"
    return f"{100 * b['mean']:.2f} [{100 * b['lo']:.2f}, {100 * b['hi']:.2f}]"


def _delta(r: dict) -> str:
    return "—" if r.get("n", 0) == 0 else f"{100 * r['delta']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]"


def _p(r: dict) -> str:
    if r.get("n", 0) == 0:
        return "—"
    return "< 0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"


def _mean(scores: dict, tasks: list[str] | None, require_all: bool = True):
    if not tasks:
        return None
    vals = [scores[t] for t in tasks if t in scores]
    if not vals or (require_all and len(vals) < len(tasks)):
        return None
    return sum(vals) / len(vals)


def _complete(s: dict) -> bool:
    return all(t in s for t in MTEB_BR_TASKS)


def _mean22(s: dict) -> dict:
    return bootstrap_mean([s[t] for t in MTEB_BR_TASKS]) if _complete(s) else NA


def _paired(a: dict, b: dict, tasks: list[str] | None = None) -> dict:
    """Bootstrap pareado nas tarefas do MTEB-BR (todas, por padrão); só com as 22 dos dois lados."""
    if not (_complete(a) and _complete(b)):
        return NA
    tasks = tasks or MTEB_BR_TASKS
    return paired_bootstrap({t: a[t] for t in tasks}, {t: b[t] for t in tasks})


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


def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _filled(row: list, first: int = 1) -> bool:
    return any(c not in ("—", "", None) for c in row[first:])


# ---------------------------------------------------------------- coleta
def collect(extra_models: list[str] | None = None, languages: tuple[str, ...] | None = PT) -> dict[str, dict]:
    """{checkpoint ou hf_id: {tarefa: nota}} nos idiomas pedidos; a origem do lumi-emb herda as notas
    dele (mesmos pesos)."""
    cache = results_dir("mteb")
    S = {c: load_scores(cache, model_name(c), languages) for c in CHECKPOINTS}
    for m in extra_models or []:
        S[m] = load_scores(cache, m, languages)
    for rel in RELEASES:
        src = release_info(rel).get("source")
        if src in S and src != rel:
            if S[rel] and not S[src]:
                S[src] = S[rel]
            elif S[src] and not S[rel]:
                S[rel] = S[src]
    return S


def release_info(name: str = "lumi-emb") -> dict:
    return _json(ckpt_dir(name) / "choice.json")


def main_ref(S: dict) -> str:
    return MAIN if _complete(S.get(MAIN, {})) else "lumi-emb"


def long_values(S: dict, baselines: list[str]) -> dict[str, tuple]:
    """{checkpoint ou hf_id: (MLDR-pt em 512, MLDR-pt em 8k)} para quem tem o resultado em 8k."""
    out = {}
    for c in ["lumi-emb", "C3-long", "M-L", "lumi-emb-long"] + list(baselines):
        name = c if c in BASELINE_NAMES else model_name(c)
        long = load_scores(results_dir("mteb"), f"{name}-{LONG_TAG}", PT).get("MultiLongDocRetrieval")
        if long is not None:
            out[c] = (S.get(c, {}).get("MultiLongDocRetrieval"), long)
    return out


def mldr_long_rows(S: dict, baselines: list[str]) -> list[list]:
    """MLDR-pt em 512 (MTEB-PT) e em 8k (suíte mldr_pt_long), para quem tem o resultado em 8k."""
    return [[BASELINE_NAMES.get(c, c), _pct(v512), _pct(v8k)] for c, (v512, v8k) in long_values(S, baselines).items()]


def suite_lists() -> dict[str, list[str]]:
    lists = {}
    for suite in EXTRA_SUITES:
        f = results_dir("suites", f"{suite}.json")
        if f.exists():
            lists[suite] = json.loads(f.read_text())
    return lists


TRAINED_SETS = ("S1/train", "S2/train", "S2-real/train", "S2-syn/train")


def contaminated_tasks() -> list[str]:
    """Tarefas do MTEB-BR com texto de treino (query ou positivo) acusado pela auditoria posterior."""
    audit = _json(results_dir("contamination", "audit.json"))
    out = set()
    for key, st in (audit.get("sets") or {}).items():
        if key in TRAINED_SETS:
            for h in st.get("hits") or []:
                if h.get("field") in ("query", "positive"):
                    out.update(t for t in h.get("tasks", []) if t in MTEB_BR_TASKS)
    return [t for t in MTEB_BR_TASKS if t in out]


def contamination_rows(S: dict, measured: list[str]) -> tuple[list[str], list[list], list[str]]:
    flagged = contaminated_tasks()
    if not flagged:
        return [], [], []
    clean = [t for t in MTEB_BR_TASKS if t not in flagged]
    ref = main_ref(S)
    h = ["Modelo", "Mean22", f"Mean{len(clean)} (sem as {len(flagged)})", f"{ref} − modelo (22)",
         f"{ref} − modelo ({len(clean)})", f"p ({len(clean)})"]
    rows = []
    for c in dict.fromkeys([ref, "lumi-emb", "C2"] + measured):
        s = S[c]
        if not _complete(s):
            continue
        r22, rc = (NA, NA) if c == ref else (_paired(S[ref], s), _paired(S[ref], s, clean))
        rows.append([BASELINE_NAMES.get(c, c), _pct(_mean(s, MTEB_BR_TASKS)), _pct(_mean(s, clean)),
                     "—" if c == ref else f"{100 * r22['delta']:+.2f}" if r22.get("n") else "—",
                     _delta(rc), _p(rc)])
    return h, rows, flagged


def merge_grid(name: str = "M-C") -> tuple[list[str], list[list]]:
    rep = _json(ckpt_dir(name) / "merge_report.json")
    if not rep:
        return [], []
    rows = []
    for vid, m in sorted(rep["dev"].items(), key=lambda kv: -kv[1]["mean_ndcg@10"]):
        tag = "escolhido" if vid == rep["selected"] else ("melhor individual" if vid == rep["best_individual"] else "")
        rows.append([vid, f"{m['mean_ndcg@10']:.4f}", rep.get("borda", {}).get(vid, "—"), tag])
    return ["Candidato", "dev nDCG@10", "Borda", ""], rows


LEMB_BY_LENGTH = ["LEMBPasskeyRetrieval", "LEMBNeedleRetrieval"]


def split_scores(model: str, task: str) -> dict[str, float]:
    """{split: nota} de uma tarefa no cache do mteb (o LongEmbed tem um split por comprimento)."""
    base = results_dir("mteb") / "results" / model.replace("/", "__")
    best, mtime = {}, -1.0
    for f in base.rglob(f"{task}.json") if base.exists() else []:
        if f.stat().st_mtime > mtime:
            j = _json(f)
            best = {sp: float(es[0]["main_score"]) for sp, es in (j.get("scores") or {}).items()
                    if isinstance(es, list) and es and "main_score" in es[0]}
            mtime = f.stat().st_mtime
    return best


def lemb_length_rows(models: list[tuple[str, str]]) -> tuple[list[str], list[list]]:
    """Passkey e Needle por comprimento (split test_<n>), para cada modelo com resultado."""
    rows, lengths = [], set()
    data = {}
    for label, model in models:
        for t in LEMB_BY_LENGTH:
            sc = split_scores(model, t)
            if sc:
                data[(label, t)] = sc
                lengths.update(sc)
    order = sorted(lengths, key=lambda sp: int(sp.rsplit("_", 1)[-1]) if sp.rsplit("_", 1)[-1].isdigit() else 0)
    for (label, t), sc in data.items():
        rows.append([label, t.replace("LEMB", "").replace("Retrieval", "")] + [_pct(sc.get(sp), 0) for sp in order])
    heads = [sp.replace("test_", "") for sp in order]
    return (["Modelo", "Tarefa"] + heads, rows) if rows else ([], [])


def dev_info(c: str) -> dict:
    return _json(ckpt_dir(c) / "summary.json")


# ---------------------------------------------------------------- seções
def seed_table(S: dict) -> tuple[list[str], list[list]]:
    done = [c for c in SEEDS if _complete(S[c])]
    if len(done) < 2:
        return [], []
    rows = []
    metrics = [("MTEB-BR Mean22", MTEB_BR_TASKS)] + list(MTEB_BR_CATEGORIES.items()) + list(MTEB_PT_FAMILIES.items())
    for name, ts in metrics:
        vals = [_mean(S[c], ts) for c in done]
        if any(v is None for v in vals):
            continue
        rows.append([name] + [_pct(v) for v in vals] + [_pct(statistics.mean(vals)), _pct(statistics.stdev(vals))])
    return ["Métrica"] + done + ["Média", "Desvio-padrão"], rows


def hypotheses(S: dict, lists: dict, SX: dict | None = None) -> list[list]:
    """Situação das hipóteses pelo critério de falseamento pré-registrado (tabela do model card)."""
    def verdict(r: dict, refuted: bool) -> str:
        return "—" if r.get("n", 0) == 0 else ("Refutada" if refuted else "Não refutada")

    rows = []
    r = _paired(S["C2"], S.get(QWEN4B, {}))
    rows.append(["H1", "Paridade prática com Transformers de porte similar", "C2 − Qwen3-Embedding-4B",
                 _delta(r), _p(r), verdict(r, r.get("hi", 0) < 0)])
    r = _paired(S["C2"], S["C2-direct"])
    rows.append(["H2", "O S1 melhora o resultado final", "C2 − C2-direct", _delta(r), _p(r),
                 verdict(r, r.get("delta", 0) <= 0)])
    r = _paired(S["C2-real"], S["C2-syn"])
    rows.append(["H3", "Reais ajudam mais nas tarefas nativas", "C2-real − C2-syn", _delta(r), _p(r),
                 verdict(r, r.get("delta", 0) <= 0)])
    r = _paired(S["C2"], S["C2-naivehn"])
    rows.append(["H4", "Negativos positive-aware superam o top-k ingênuo", "C2 − C2-naivehn", _delta(r), _p(r),
                 verdict(r, r.get("delta", 0) <= 0)])
    ra, rf, rbf = _paired(S["B2a"], S["C2"]), _paired(S["B2f"], S["C2"]), _paired(S["B2f"], S["B2a"])
    h5 = NA if 0 in (ra.get("n", 0), rf.get("n", 0)) else rbf
    rows.append(["H5", "O bidirecional melhora a média", "B2a − C2; B2f − C2", f"{_delta(ra)}; {_delta(rf)}",
                 f"{_p(ra)}; {_p(rf)}", verdict(h5, h5.get("delta", 0) <= 0 or (ra.get("delta", 0) <= 0 and rf.get("delta", 0) <= 0))])
    rep = _json(ckpt_dir("M-C") / "merge_report.json")
    if rep:
        best = rep["best_individual"]
        dm, db = rep["dev"][rep["selected"]]["mean_ndcg@10"], rep["dev"][best]["mean_ndcg@10"]
        rows.append(["H6", "O merge supera o melhor checkpoint individual (dev)",
                     f"M-C ({rep['selected']}) − {best}, nDCG@10 no dev", f"{100 * (dm - db):+.2f}", "—",
                     "Refutada" if dm <= db else "Não refutada"])
    else:
        rows.append(["H6", "O merge supera o melhor checkpoint individual (dev)", "M-C − melhor individual",
                     "—", "—", "—"])
    le, SL = lists.get("longembed"), (SX or {}).get("longembed", S)
    cand = [(c, _mean(SL.get(c, {}), le)) for c in ("C3-long", "C2", "lumi-emb")]
    who, ours = next(((c, v) for c, v in cand if v is not None), ("C3-long", None))
    theirs = _mean(SL.get(QWEN4B, {}), le)
    ok = ours is not None and theirs is not None
    rows.append(["H7", "Vantagem relativa em entradas longas", f"{who} − Qwen3-Embedding-4B, LongEmbed",
                 f"{100 * (ours - theirs):+.2f}" if ok else "—", "—",
                 ("Refutada" if ours < theirs else "Não refutada") if ok else "—"])
    return rows


# ---------------------------------------------------------------- relatório completo
def build_report(ecfg: dict) -> str:
    baselines = [m["hf"] for m in LITERATURE if m["hf"]]
    S = collect(baselines)
    lists = suite_lists()
    SX = {k: collect(baselines, SUITE_LANGS[k]) for k in EXTRA_SUITES}

    def xmean(c: str, k: str):
        return _mean(SX[k].get(c, {}), lists.get(k))
    rel = release_info()
    measured = [b for b in baselines if _complete(S[b])]
    tables = results_dir("tables")
    md = ["# Lumi-Embed — resultados\n",
          "Gerado por `lumi_embed report` a partir de `results/embed/`. Notas × 100. Células \"—\" = sem "
          "resultado ainda. IC 95 % por bootstrap sobre tarefas; p-valores por bootstrap pareado (unidade = "
          "tarefa), como no MTEB-BR [16]. Mesmo protocolo para todos os modelos: mesmas instruções em português "
          "(só na query) e comprimento máximo de 512 tokens.\n"]
    if rel:
        md.append(f"O **lumi-emb** é o checkpoint `{rel.get('source')}` "
                  f"({rel.get('merge_selected_variant')}), escolhido no dev pela regra: {rel.get('rule')}\n")
    rel_long = release_info("lumi-emb-long")
    if rel_long:
        md.append(f"O **lumi-emb-long** é o checkpoint `{rel_long.get('source')}` "
                  f"({rel_long.get('merge_selected_variant')}), escolhido no dev curto + longo pela mesma regra.\n")

    # Tabela 12 — visão geral
    h12 = ["Modelo", "MTEB-BR Mean22 [IC 95%]"] + list(MTEB_PT_FAMILIES) + list(EXTRA_SUITES.values())
    rows12 = []
    for c in CHECKPOINTS + measured:
        s = S[c]
        rows12.append([BASELINE_NAMES.get(c, c), _ci(_mean22(s))] + [_pct(_mean(s, ts)) for ts in MTEB_PT_FAMILIES.values()]
                      + [_pct(xmean(c, k)) for k in EXTRA_SUITES])
    _write_csv(tables / "table12_overview.csv", h12, rows12)
    md += ["## Tabela 12 — Visão geral\n", _table(h12, rows12), ""]

    # Tabela 13 — MTEB-BR por categoria
    h13 = ["Modelo"] + list(MTEB_BR_CATEGORIES)
    rows13 = [[BASELINE_NAMES.get(c, c)] + [_pct(_mean(S[c], ts)) for ts in MTEB_BR_CATEGORIES.values()]
              for c in CHECKPOINTS + measured]
    _write_csv(tables / "table13_mteb_br_categories.csv", h13, rows13)
    md += ["## Tabela 13 — MTEB-BR por categoria\n", _table(h13, rows13), ""]

    # Tabela 13b — por tarefa (CSV completo; no markdown, uma seleção)
    cols = CHECKPOINTS + measured
    _write_csv(tables / "table13b_mteb_br_tasks.csv", ["Tarefa"] + [BASELINE_NAMES.get(c, c) for c in cols],
               [[t] + [_pct(S[c].get(t)) for c in cols] for t in MTEB_BR_TASKS])
    keep = ["R0", "C1", "C2", "C2-real", "C2-syn", "lumi-emb", "lumi-emb-long"] + ([QWEN4B] if QWEN4B in measured else [])
    md += ["## Tabela 13b — MTEB-BR por tarefa (seleção; completo em CSV)\n",
           _table(["Tarefa"] + [BASELINE_NAMES.get(c, c) for c in keep],
                  [[t] + [_pct(S[c].get(t)) for c in keep] for t in MTEB_BR_TASKS]), ""]

    # Variação entre sementes
    hs, rs = seed_table(S)
    if rs:
        _write_csv(tables / "seeds.csv", hs, rs)
        md += ["## Variação entre sementes (C2)\n", _table(hs, rs),
               "\nMesma receita e mesmos dados; muda só a semente. Diferenças entre checkpoints menores que cerca "
               "de 2 desvios-padrão não se distinguem da variação entre execuções.\n"]

    # Tabela 14 — comparação com modelos correspondentes
    h14 = ["Modelo", "Relação", "MTEB-BR Mean22", "MTEB-PT (STS/Cls/Ret/Rer)", "MMTEB", "LongEmbed"]
    rows14 = []
    rel14 = {"lumi-emb-long": "Modelo publicado (até 8k tokens)", "lumi-emb": "Versão curta (até 512 tokens)",
             "R0": "Ponto de partida"}
    for c in rel14:
        s = S[c]
        rows14.append([f"{c} (este trabalho)", rel14[c],
                       _pct(_mean(s, MTEB_BR_TASKS)), " / ".join(_pct(_mean(s, ts), 1) for ts in MTEB_PT_FAMILIES.values()),
                       _pct(xmean(c, "mmteb")), _pct(xmean(c, "longembed"))])
    for m in LITERATURE:
        s = S.get(m["hf"], {}) if m["hf"] else {}
        mbr, mpt = _mean(s, MTEB_BR_TASKS), [_mean(s, ts) for ts in MTEB_PT_FAMILIES.values()]
        mmt, mle = (xmean(m["hf"], "mmteb"), xmean(m["hf"], "longembed")) if m["hf"] else (None, None)
        rows14.append([m["model"], m["rel"],
                       m["mteb_br"] if mbr is None else f"{_pct(mbr)} (medido)",
                       m["mteb_pt"] if all(v is None for v in mpt) else " / ".join(_pct(v, 1) for v in mpt) + " (medido)",
                       m["mmteb"] if mmt is None else f"{_pct(mmt)} (medido)",
                       m["longembed"] if mle is None else f"{_pct(mle)} (medido)"])
    _write_csv(tables / "table14_comparison.csv", h14, rows14)
    md += ["## Tabela 14 — Comparação com modelos correspondentes\n", _table(h14, rows14),
           "\nValores com referência entre colchetes vêm da literatura (ver o plano); \"(medido)\" = avaliado aqui "
           "com o mesmo protocolo; \"a medir\" = ainda não avaliado.\n"]

    # Tabela 14b — comparação pareada no MTEB-BR
    ref = main_ref(S)
    pairs = [(ref, b) for b in measured]
    if ref != "lumi-emb":
        pairs += [(ref, "lumi-emb")] + [("lumi-emb", b) for b in measured]
    pairs += [("C2", QWEN4B)] if QWEN4B in measured else []
    h14b = ["Comparação", "Δ Mean22 [IC 95%]", "p", "Vitórias (tarefas)"]
    rows14b = []
    for a, b in pairs:
        r = _paired(S[a], S[b])
        wins = sum(S[a][t] > S[b][t] for t in MTEB_BR_TASKS) if r.get("n") else None
        rows14b.append([f"{a} − {BASELINE_NAMES.get(b, b)}", _delta(r), _p(r),
                        "—" if wins is None else f"{wins}/{len(MTEB_BR_TASKS)}"])
    if rows14b:
        _write_csv(tables / "table14b_paired.csv", h14b, rows14b)
        md += ["## Tabela 14b — Comparação pareada no MTEB-BR\n", _table(h14b, rows14b), ""]

    # MLDR-pt em 8k (RAG com documentos longos)
    rml = mldr_long_rows(S, baselines)
    if rml:
        _write_csv(tables / "mldr_pt_long.csv", MLDR_HEAD, rml)
        md += ["## MLDR-pt com documentos de até 8k tokens\n", _table(MLDR_HEAD, rml), "", MLDR_NOTE, ""]

    # LongEmbed por comprimento
    lemb_models = [(c, model_name(c)) for c in ("lumi-emb", "C3-long", "lumi-emb-long")] \
        + [(BASELINE_NAMES[b], b) for b in baselines]
    hl, rl = lemb_length_rows(lemb_models)
    if rl:
        _write_csv(tables / "longembed_by_length.csv", hl, rl)
        md += ["## LongEmbed por comprimento (tokens)\n", _table(hl, rl), ""]

    # Sensibilidade à contaminação (auditoria posterior)
    hc, rc, flagged = contamination_rows(S, measured)
    if rc:
        _write_csv(tables / "contamination_sensitivity.csv", hc, rc)
        md += ["## Sensibilidade à contaminação\n", CONTAM_NOTE.format(tasks=", ".join(flagged)), "",
               _table(hc, rc), ""]

    # Tabela 15 — ablações
    h15 = ["ID", "Pergunta", "Comparação", "Δ Mean22 [IC 95%]", "p", "Tarefas"]
    rows15 = []
    for aid, q, a, b in ABLATIONS:
        r = _paired(S[a], S[b])
        rows15.append([aid, q, f"{a} − {b}", _delta(r), _p(r), r.get("n", 0)])
    _write_csv(tables / "table15_ablations.csv", h15, rows15)
    md += ["## Tabela 15 — Ablações (MTEB-BR)\n", _table(h15, rows15),
           f"\n{A2_NOTE}\n"]

    # Hipóteses
    hh = ["ID", "Hipótese", "Teste", "Δ [IC 95%]", "p", "Situação"]
    rh = hypotheses(S, lists, SX)
    _write_csv(tables / "hypotheses.csv", hh, rh)
    md += ["## Hipóteses pré-registradas: situação\n", _table(hh, rh),
           "\n\"Refutada\" / \"Não refutada\" aplicam o critério de falseamento do plano; \"—\" = falta executar. "
           "H6 usa o dev (critério do plano), sem IC.\n"]

    # Dev
    hdev, rdev = ["Checkpoint", "dev nDCG@10 médio", "Passos", "Minutos de treino", "Observação"], []
    for c in CHECKPOINTS:
        js = dev_info(c)
        if not js:
            continue
        note = ""
        relc = release_info(c) if c in RELEASES else {}
        if relc:
            note = f"= {relc.get('source')} ({relc.get('merge_selected_variant')})"
        elif js.get("selected"):
            note = f"= {js['selected']}"
        rdev.append([c, f"{(js.get('dev') or {}).get('mean_ndcg@10', float('nan')):.4f}",
                     js.get("steps", "—"), js.get("train_minutes", "—"), note])
    if rdev:
        md += ["## Dev set (seleção de checkpoints)\n", _table(hdev, rdev), ""]
    for mname in ("M-C", "M-L"):
        hg, rg = merge_grid(mname)
        if rg:
            _write_csv(tables / f"merge_grid_{mname}.csv", hg, rg)
            md += [f"## Grade do merge {mname} (dev)\n", _table(hg, rg), ""]

    text = "\n".join(md) + "\n"
    out = REPO_ROOT / "docs" / "embed" / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    (wpath("results", "embed", "RESULTS.md")).write_text(text, encoding="utf-8")
    update_card_block(card_block(S, lists, measured, rows14b, rs and (hs, rs), rows15, rh, rdev, xmean))
    figs = figure_spec(S, measured)
    from .figures import render

    made = {p.name.rsplit("-", 1)[0] for p in render(figs, REPO_ROOT / "docs" / "embed" / "figures")}
    _replace_block(REPO_ROOT / "README.md", "HIGHLIGHTS", highlights(S, measured, made, _img_github))
    _replace_block(REPO_ROOT / "docs" / "huggingface" / "README.md", "HIGHLIGHTS",
                   highlights(S, measured, made, _img_hub))
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "huggingface" / "README.md"):
        _replace_block(path, "ABLATIONS", lessons(S))
    return text


# ---------------------------------------------------------------- bloco do model card
def card_block(S, lists, measured, rows14b, seeds, rows15, rh, rdev, xmean) -> str:
    """Versão resumida para o model card: só linhas com resultado; o completo fica no RESULTS.md."""
    md = ["### Resultados\n",
          f"Notas × 100, geradas pelos arquivos de avaliação ([tabelas completas]({RESULTS_URL})). Mesmo protocolo "
          "para todos os modelos: mesmas instruções em português, só na query, e até 512 tokens. IC 95 % e "
          "p-valores por bootstrap sobre as tarefas, como no MTEB-BR.\n"]
    order = sorted(measured, key=lambda b: -_mean(S[b], MTEB_BR_TASKS))
    models = [("**lumi-emb-long** (este trabalho)", "lumi-emb-long"), ("**lumi-emb** (este trabalho)", "lumi-emb")] \
        + [(BASELINE_NAMES[b], b) for b in order] + [("Lumi-4B sem treino (R0)", "R0")]
    h = ["Modelo", "Mean22 [IC 95%]"] + [k.replace("Clusterização", "Clust.") for k in MTEB_BR_CATEGORIES]
    rows = [[n, _ci(_mean22(S[c]))] + [_pct(_mean(S[c], ts), 1) for ts in MTEB_BR_CATEGORIES.values()]
            for n, c in models]
    md += ["#### MTEB-BR (22 tarefas nativas em PT-BR)\n", _table(h, [r for r in rows if _filled(r)]), ""]
    ref = main_ref(S)
    r14 = [r for r in rows14b if _filled(r) and r[0].startswith(f"{ref} − ")]
    if r14:
        md += ["#### Comparação pareada no MTEB-BR\n",
               _table(["Comparação", "Δ Mean22 [IC 95%]", "p", "Vitórias (tarefas)"], r14), ""]
    hpt = ["Modelo"] + list(MTEB_PT_FAMILIES)
    rpt = [[n] + [_pct(_mean(S[c], ts), 1) for ts in MTEB_PT_FAMILIES.values()] for n, c in models]
    md += ["#### MTEB-PT (por família)\n", _table(hpt, [r for r in rpt if _filled(r)]), ""]
    rex = [[n] + [_pct(xmean(c, k)) for k in EXTRA_SUITES] for n, c in models]
    rex = [r for r in rex if _filled(r)]
    if rex:
        md += ["#### Outras suítes\n", _table(["Modelo"] + list(EXTRA_SUITES.values()), rex), ""]
    hcn, rcn, flagged = contamination_rows(S, measured)
    if rcn:
        md += ["#### Sensibilidade à contaminação\n", CONTAM_NOTE.format(tasks=", ".join(flagged)), "",
               _table(hcn, rcn), ""]
    rml = mldr_long_rows(S, list(BASELINE_NAMES))
    if rml:
        md += ["#### MLDR-pt com documentos de até 8k tokens\n", _table(MLDR_HEAD, rml), "", MLDR_NOTE, ""]
    hl, rl = lemb_length_rows([(c, model_name(c)) for c in ("lumi-emb", "C3-long", "lumi-emb-long")]
                              + [(BASELINE_NAMES[b], b) for b in BASELINE_NAMES])
    if rl:
        md += ["#### LongEmbed por comprimento (tokens)\n", _table(hl, rl), ""]
    rc = []
    devs = {r[0]: r for r in rdev}
    for c in CHECKPOINTS:
        row = [c, _ci(_mean22(S[c])), devs[c][1] if c in devs else "—", devs[c][4] if c in devs else ""]
        if _filled(row[:3]):
            rc.append(row)
    md += ["#### Checkpoints deste trabalho\n",
           _table(["Checkpoint", "MTEB-BR Mean22 [IC 95%]", "dev nDCG@10", "Observação"], rc), ""]
    for mname in ("M-C", "M-L"):
        hg, rg = merge_grid(mname)
        if rg:
            md += [f"#### Grade do merge {mname} (dev)\n", _table(hg, rg), ""]
    if seeds:
        hs, rs = seeds
        md += ["#### Variação entre sementes (C2)\n", _table(hs, rs[:1 + len(MTEB_BR_CATEGORIES)]), ""]
    r15 = [r for r in rows15 if r[-1]]
    if r15:
        md += ["#### Ablações (MTEB-BR)\n", _table(["ID", "Pergunta", "Comparação", "Δ Mean22 [IC 95%]", "p", "Tarefas"],
                                                   r15), f"\n{A2_NOTE}\n"]
    md += ["#### Hipóteses pré-registradas\n", _table(["ID", "Hipótese", "Teste", "Δ [IC 95%]", "p", "Situação"], rh),
           "\n\"Refutada\" / \"Não refutada\" aplicam o critério de falseamento do plano; \"—\" = ainda não "
           "executado.\n"]
    return "\n".join(md)


def _replace_block(path: Path, tag: str, block: str) -> None:
    """Troca o texto entre <!-- TAG:START --> e <!-- TAG:END --> (se os marcadores existirem)."""
    if not path.exists():
        return
    t = path.read_text(encoding="utf-8")
    a, b = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if a in t and b in t:
        t = t.split(a)[0] + a + "\n" + block.rstrip() + "\n\n" + b + t.split(b)[1]
        path.write_text(t, encoding="utf-8")


def update_card_block(block: str) -> None:
    _replace_block(REPO_ROOT / "docs" / "huggingface" / "README.md", "RESULTS", block)


# ---------------------------------------------------------------- destaques e figuras (README e card)
PT_FAMILY_NAMES = {"PT STS": "STS", "PT Cls": "Classificação", "PT Ret": "Retrieval", "PT Rer": "Reranking"}
FIG_ALT = {
    "mldr-pt": "MLDR-pt: nDCG@10 em 512 tokens e com o documento inteiro (até 8k)",
    "mteb-br-categories": "MTEB-BR por categoria: lumi-emb-long, Qwen3-Embedding-4B e Qwen3-Embedding-8B",
    "mteb-br-ranking": "MTEB-BR: média das 22 tarefas por modelo",
    "longembed-length": "LongEmbed Passkey e Needle por comprimento",
}


def _diff(a: float, b: float) -> str:
    """a − b com os dois já arredondados como aparecem nas tabelas (o Δ bate com a conta de cabeça)."""
    return f"{round(100 * a, 2) - round(100 * b, 2):+.2f}"


def _img_github(name: str) -> str:
    base = "docs/embed/figures"
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{base}/{name}-dark.png">\n'
            f'  <img alt="{FIG_ALT.get(name, name)}" src="{base}/{name}-light.png" width="100%">\n</picture>')


def _img_hub(name: str) -> str:
    return f"![{FIG_ALT.get(name, name)}](docs/figures/{name}-light.png)"


def _label(c: str) -> str:
    return "Lumi-4B sem treino (R0)" if c == "R0" else BASELINE_NAMES.get(c, c)


def figure_spec(S: dict, measured: list[str]) -> dict:
    """Dados de cada figura, só do que já tem resultado (sem resultado, a figura não sai)."""
    ref = main_ref(S)
    trio = ([("lumi-emb", "muted")] if ref != "lumi-emb" else []) + [(ref, "s1"), (QWEN4B, "s2")]
    spec = {}
    lv = long_values(S, list(BASELINE_NAMES))
    data = [(_label(c), role, 100 * lv[c][0], 100 * lv[c][1]) for c, role in trio
            if c in lv and lv[c][0] is not None]
    if len(data) >= 2 and any(role == "s1" for _, role, *_ in data):
        spec["mldr-pt"] = ("mldr_slope", {"data": data})
    curves = {}
    for task, title in (("LEMBPasskeyRetrieval", "Passkey"), ("LEMBNeedleRetrieval", "Needle")):
        series = {}
        for c, _ in trio:
            sc = split_scores(c if c in BASELINE_NAMES else model_name(c), task)
            pts = {int(sp.rsplit("_", 1)[-1]): 100 * v for sp, v in sc.items() if sp.rsplit("_", 1)[-1].isdigit()}
            if pts:
                series[_label(c)] = pts
        if _label(ref) in series and len(series) >= 2:
            curves[title] = series
    if curves:
        spec["longembed-length"] = ("length_curves", {"data": curves, "roles": {_label(c): r for c, r in trio},
                                                      "trained": LONG_MAX})
    have = [(c, r) for c, r in ((ref, "s1"), (QWEN4B, "s2"), (QWEN8B, "s3")) if c == ref or c in measured]
    if QWEN4B in measured:
        cats = list(MTEB_BR_CATEGORIES)
        vals = {c: [100 * _mean(S[c], MTEB_BR_CATEGORIES[k]) for k in cats] for c, _ in have}
        order = sorted(range(len(cats)), key=lambda i: -(vals[ref][i] - vals[QWEN4B][i]))
        spec["mteb-br-categories"] = ("category_dots", {
            "cats": [cats[i] for i in order],
            "series": [(_label(c), r, [vals[c][i] for i in order]) for c, r in have]})
    rows = [(_label(c), "s1" if c == ref else "s1b" if c == "lumi-emb" else "muted", 100 * _mean(S[c], MTEB_BR_TASKS))
            for c in dict.fromkeys([ref, "lumi-emb", "R0"] + measured) if _complete(S.get(c, {}))]
    if len(rows) >= 3:
        spec["mteb-br-ranking"] = ("ranking_bars", {
            "rows": rows, "title": "MTEB-BR: média das 22 tarefas nativas em PT-BR",
            "subtitle": "Azul: este trabalho. Mesmo protocolo para todos (instruções em português, até 512 tokens)"})
    return spec


def lessons(S: dict) -> str:
    """Tabela das ablações já avaliadas, lida como efeito do componente: ajuda, empate ou piora (pelo IC)."""
    rows = []
    for q, a, b in LESSONS:
        r = _paired(S.get(a, {}), S.get(b, {}))
        if r.get("n"):
            effect = "**ajuda**" if r["lo"] > 0 else "**piora**" if r["hi"] < 0 else "empate"
            rows.append([q, f"{a} − {b}", _delta(r), _p(r), effect])
    if not rows:
        return "_As lições aparecem quando houver ablações com as 22 tarefas do MTEB-BR._"
    md = [_table(["O que foi testado", "Comparação", "Δ MTEB-BR [IC 95%]", "p", "Efeito"], rows), "",
          "Δ = média das 22 tarefas do MTEB-BR com o que foi testado menos a média sem ele (× 100), com IC de 95 % "
          "e p por bootstrap pareado nas tarefas. \"Ajuda\" ou \"piora\" quando o IC não cruza o zero."]
    seeds = [100 * _mean(S[c], MTEB_BR_TASKS) for c in SEEDS if _complete(S.get(c, {}))]
    if len(seeds) >= 2:
        md[-1] += (f" Para comparar, o C2 treinado com {len(seeds)} sementes varia {statistics.stdev(seeds):.2f} "
                   "(desvio-padrão) na média.")
    if any("¹" in r[0] for r in rows):
        md += ["", "¹ Conjuntos de tamanhos diferentes: o S2-real e o S2-syn são menores que o S2 completo."]
    return "\n".join(md)


def highlights(S: dict, measured: list[str], made: set[str], img) -> str:
    """Destaques do modelo publicado contra as referências medidas aqui, com as figuras já geradas."""
    ref = main_ref(S)
    s = S.get(ref, {})
    if not _complete(s):
        return "_Os destaques aparecem quando o modelo publicado tiver as 22 tarefas do MTEB-BR._"
    fig = lambda name: [img(name), ""] if name in made else []  # noqa: E731
    rows = []
    lv = long_values(S, list(BASELINE_NAMES))
    if ref != "lumi-emb" and ref in lv and "lumi-emb" in lv:
        v512, v8k = lv[ref]
        old = lv["lumi-emb"][1]
        note = f"{_pct(old)} sem o estágio longo ({_diff(v8k, old)})"
        if v512 is not None:
            note += f"; {_pct(v512)} lendo só os primeiros 512 tokens"
        rows.append(["Documentos longos em PT (MLDR-pt, até 8k tokens)", f"**{_pct(v8k)}**", note])
    r0 = _mean(S.get("R0", {}), MTEB_BR_TASKS)
    rows.append(["MTEB-BR (22 tarefas nativas em PT-BR)", f"**{_pct(_mean(s, MTEB_BR_TASKS))}**",
                 f"{_pct(r0)} do Lumi-4B sem treino" if r0 is not None else "—"])
    groups = [(f"{k.split(' (')[0]} no MTEB-BR ({len(ts)} tarefas)", ts) for k, ts in MTEB_BR_CATEGORIES.items()]
    groups += [(f"{PT_FAMILY_NAMES[k]} no MTEB-PT", ts) for k, ts in MTEB_PT_FAMILIES.items()]
    for name, ts in groups:
        ours = _mean(s, ts)
        base = {b: v for b in measured if (v := _mean(S[b], ts)) is not None}
        if ours is None or not base:
            continue
        second = max(base, key=base.get)
        if ours > base[second]:
            rows.append([name, f"**{_pct(ours)}**", f"maior média entre os {len(base) + 1} modelos comparados "
                                                     f"(depois: {_label(second)}, {_pct(base[second])})"])
        elif QWEN4B in base and ours > base[QWEN4B]:
            rows.append([name, f"**{_pct(ours)}**", f"acima do Qwen3-Embedding-4B, de mesmo porte ({_pct(base[QWEN4B])})"])
    md = [_table(["", ref, "Comparação"], rows), ""] + fig("mldr-pt")
    notes = []
    ties = [(b, r) for b in measured if (r := _paired(s, S[b])).get("n") and r["p"] >= 0.05]
    if ties:
        notes.append("- **Empate estatístico** na média do MTEB-BR com "
                     + " e ".join(f"{_label(b)} (Δ {100 * r['delta']:+.2f}, p = {_p(r)})" for b, r in ties) + ".")
    if ref != "lumi-emb" and _complete(S.get("lumi-emb", {})):
        r = _paired(s, S["lumi-emb"])
        if r["p"] >= 0.05:
            notes.append(f"- **O contexto longo não custou nada nos textos curtos:** {ref} − lumi-emb = "
                         f"{100 * r['delta']:+.2f} no MTEB-BR (p = {_p(r)}).")
        else:
            notes.append(f"- Textos curtos: {ref} − lumi-emb = {_delta(r)} no MTEB-BR (p = {_p(r)}).")
    md += notes + ([""] if notes else [])
    if QWEN4B in measured:
        q4, q8 = S[QWEN4B], S.get(QWEN8B, {})
        cat_of = {t: k.split(" (")[0] for k, ts in MTEB_BR_CATEGORIES.items() for t in ts}
        wins = sorted((t for t in MTEB_BR_TASKS if s[t] > q4[t]), key=lambda t: q4[t] - s[t])
        md += ["### Onde ele se destaca\n"] + fig("mteb-br-categories")
        if wins:
            h8 = _complete(q8)
            md += [f"Nas tarefas abaixo ({len(wins)} das 22 do MTEB-BR), o {ref} supera o Qwen3-Embedding-4B, "
                   "um Transformer de mesmo porte"
                   + (", e as marcadas com ✓ superam também o Qwen3-Embedding-8B, com o dobro de parâmetros." if h8
                      else "."), "",
                   _table(["Tarefa", "Categoria", ref, "Qwen3-Embedding-4B", "Δ"] + (["> 8B"] if h8 else []),
                          [[t, cat_of[t], f"**{_pct(s[t])}**", _pct(q4[t]), _diff(s[t], q4[t])]
                           + (["✓" if s[t] > q8[t] else ""] if h8 else []) for t in wins]), ""]
    md += fig("mteb-br-ranking")
    if "longembed-length" in made:
        md += ["### Contexto longo\n", f"O estágio longo treinou com documentos de até {LONG_MAX // 1024}k tokens. "
               "Nessa faixa, o modelo encontra a informação escondida bem melhor do que antes dele.", ""]
        md += fig("longembed-length")
    return "\n".join(md)
