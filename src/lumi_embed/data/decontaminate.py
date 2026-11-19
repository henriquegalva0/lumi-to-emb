"""Controle de contaminação (seção 2.5 do plano): zero-shot limpo.

1. Exclusão por dataset: nenhuma fonte de treino pode ser tarefa de avaliação (lista na config).
2. Índice dos textos de avaliação: todas as strings (corpus, queries, sentenças) das tarefas
   do MTEB-BR (22), MTEB-PT (14), LongEmbed e MTEB eng v2, carregadas pelo próprio mteb.
   - textos inteiros normalizados (hash) → igualdade exata;
   - 13-gramas de palavras (com passo) → sobreposição parcial.
3. Filtro: um exemplo sai se a query ou o positivo aparecem inteiros na avaliação, ou se a
   fração dos seus 13-gramas presentes no índice passa do limiar; negativos contaminados saem
   sozinhos. O relatório (por fonte e por tarefa de avaliação) vai para results/ e docs/.

Hash estável entre execuções (blake2b por palavra + hash polinomial dos n-gramas em uint64).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter

import numpy as np

from ..utils import EmbedError, data_dir, iter_jsonl, results_dir, write_json
from .schema import norm_text

log = logging.getLogger("lumi_embed.decontam")

WORD_RE = re.compile(r"\w+", re.UNICODE)
P = np.uint64(1099511628211)  # primo FNV-64


class Hasher:
    def __init__(self):
        self.cache: dict[str, int] = {}

    def word_ids(self, text: str) -> np.ndarray:
        out = []
        for w in WORD_RE.findall(text.lower()):
            h = self.cache.get(w)
            if h is None:
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
                self.cache[w] = h
            out.append(h)
        return np.array(out, dtype=np.uint64)

    def ngrams(self, text: str, n: int, stride: int = 1) -> np.ndarray:
        w = self.word_ids(text)
        if len(w) < n:
            return np.empty(0, dtype=np.uint64)
        with np.errstate(over="ignore"):
            h = np.zeros(len(w) - n + 1, dtype=np.uint64)
            for j in range(n):
                h = h * P + w[j : len(w) - n + 1 + j]
        return h[::stride]

    @staticmethod
    def text_hash(text: str) -> int:
        t = " ".join(WORD_RE.findall(norm_text(text).lower()))
        return int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")


# --------------------------------------------------------------------------- textos de avaliação


def _iter_strings(obj):
    """Todas as strings de uma estrutura de dados do mteb (dicts, listas, datasets.Dataset)."""
    try:
        import datasets
    except ImportError:  # pragma: no cover
        datasets = None
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)
    elif datasets is not None and isinstance(obj, datasets.DatasetDict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif datasets is not None and isinstance(obj, datasets.Dataset):
        for col, feat in obj.features.items():
            dtype = getattr(feat, "dtype", None) or getattr(getattr(feat, "feature", None), "dtype", None)
            if dtype in ("string", "large_string"):
                for v in obj[col]:
                    yield from _iter_strings(v)


def eval_tasks(benchmarks: list[str]):
    import mteb

    tasks, names = [], set()
    for b in benchmarks:
        if b == "mteb_br":
            import mteb_pt
            import mteb_pt.register  # noqa: F401

            ts = mteb.get_tasks(tasks=mteb_pt.HEADLINE_TASKS)
        else:
            ts = mteb.get_benchmark(b).tasks
        for t in ts:
            if t.metadata.name not in names:
                names.add(t.metadata.name)
                tasks.append(t)
    return tasks


def build_index(cfg: dict) -> dict:
    """Constrói (ou reaproveita) o índice de n-gramas dos textos de avaliação."""
    dcfg = cfg["decontamination"]
    out_dir = data_dir("decontam")
    idx_file = out_dir / "eval_index.npz"
    meta_file = out_dir / "eval_index.json"
    key = {k: dcfg[k] for k in ("benchmarks", "ngram", "stride")}
    if idx_file.exists() and meta_file.exists() and json.loads(meta_file.read_text()).get("key") == key:
        z = np.load(idx_file)
        return {"ngrams": z["ngrams"], "texts": z["texts"], "meta": json.loads(meta_file.read_text())}
    hasher = Hasher()
    n, stride = int(dcfg["ngram"]), int(dcfg["stride"])
    grams, texts, per_task = [], [], {}
    t0 = time.time()
    for task in eval_tasks(dcfg["benchmarks"]):
        name = task.metadata.name
        try:
            task.load_data()
        except Exception as exc:  # noqa: BLE001
            raise EmbedError(f"Não foi possível carregar a tarefa de avaliação {name}: {exc}") from exc
        count = 0
        for s in _iter_strings(task.dataset):
            if len(s) < 20:
                continue
            texts.append(Hasher.text_hash(s))
            g = hasher.ngrams(s, n, stride)
            if len(g):
                grams.append(g)
            count += 1
        per_task[name] = count
        log.info("  %s: %d textos", name, count)
        if hasattr(task, "unload_data"):
            task.unload_data()
    ng = np.unique(np.concatenate(grams)) if grams else np.empty(0, dtype=np.uint64)
    tx = np.unique(np.array(texts, dtype=np.uint64))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(idx_file, ngrams=ng, texts=tx)
    meta = {"key": key, "tasks": per_task, "n_ngrams": int(len(ng)), "n_texts": int(len(tx)),
            "seconds": round(time.time() - t0, 1)}
    meta_file.write_text(json.dumps(meta, indent=2))
    return {"ngrams": ng, "texts": tx, "meta": meta}


# --------------------------------------------------------------------------- filtro

# Texto "spinado" (fazendas de conteúdo que trocam palavras por sinônimos automaticamente): a troca
# de "o" por "este" no fim de palavras deixa marcas como "situaçãeste" e "atuaçãeste". Em português
# correto, "ã"/"õ" seguido de um demonstrativo dentro da mesma palavra praticamente não ocorre.
SPUN = re.compile(r"\w[ãõ](?:este|esta|esse|essa|aquele|aquela|isto|isso)\b", re.IGNORECASE)


def is_spun(text: str) -> bool:
    return bool(SPUN.search(text or ""))


def overlap_ratio(hasher: Hasher, text: str, index_ngrams: np.ndarray, n: int) -> float:
    g = hasher.ngrams(text, n, 1)
    if len(g) == 0 or len(index_ngrams) == 0:
        return 0.0
    pos = np.searchsorted(index_ngrams, g)
    pos[pos >= len(index_ngrams)] = 0
    return float((index_ngrams[pos] == g).mean())


def is_contaminated(hasher: Hasher, text: str, index: dict, n: int, thr: float) -> str | None:
    if Hasher.text_hash(text) in _text_set(index):
        return "exact"
    if overlap_ratio(hasher, text, index["ngrams"], n) >= thr:
        return "ngram"
    return None


def _text_set(index: dict) -> set:
    if "_set" not in index:
        index["_set"] = set(int(x) for x in index["texts"])
    return index["_set"]


def check_dataset_exclusions(cfg: dict, files: list[str]) -> list[str]:
    banned = [b.lower() for b in cfg["decontamination"]["excluded_datasets"]]
    hits = set()
    for f in files:
        for ex in iter_jsonl(data_dir(f), limit=2000):
            ds = str((ex.get("source") or {}).get("dataset", "")).lower()
            if any(b in ds for b in banned):
                hits.add(f"{f}: {ds}")
    return sorted(hits)


def run_decontaminate(cfg: dict) -> dict:
    dcfg = cfg["decontamination"]
    files = [f for f in dcfg["inputs"] if data_dir(f).exists()]
    hits = check_dataset_exclusions(cfg, files)
    if hits:
        raise EmbedError("Fontes de treino que são tarefas de avaliação (proibido): " + "; ".join(hits))
    index = build_index(cfg)
    hasher = Hasher()
    n, thr = int(dcfg["ngram"]), float(dcfg["threshold"])
    quality = bool(dcfg.get("drop_spun_text", True))
    report = {"benchmarks": dcfg["benchmarks"], "ngram": n, "stride": int(dcfg["stride"]), "threshold": thr,
              "eval_index": index["meta"], "files": {}}
    for f in files:
        reasons: Counter = Counter()
        kept = neg_dropped = total = 0
        out = data_dir("clean", f.split("/", 1)[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for ex in iter_jsonl(data_dir(f)):
                total += 1
                if quality and (is_spun(ex["query"]) or is_spun(ex["positive"])):
                    reasons["spun"] += 1
                    continue
                why = is_contaminated(hasher, ex["query"], index, n, thr) or \
                    is_contaminated(hasher, ex["positive"], index, n, thr)
                if why:
                    reasons[why] += 1
                    continue
                for field in ("negatives", "naive_negatives", "bm25_negatives"):
                    if ex.get(field):
                        before = len(ex[field])
                        ex[field] = [x for x in ex[field] if not (
                            (quality and is_spun(x["text"] if isinstance(x, dict) else x))
                            or is_contaminated(hasher, x["text"] if isinstance(x, dict) else x, index, n, thr))]
                        neg_dropped += before - len(ex[field])
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
        report["files"][f] = {"in": total, "kept": kept, "dropped": dict(reasons), "negatives_dropped": neg_dropped}
        log.info("%s: %d → %d (removidos %s; negativos %d)", f, total, kept, dict(reasons), neg_dropped)
    write_json(results_dir("contamination", "report.json"), report)
    (results_dir("contamination") / "report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(r: dict) -> str:
    lines = ["# Relatório de contaminação\n",
             f"Política: zero-shot limpo. Benchmarks indexados: {', '.join(r['benchmarks'])}. "
             f"{r['ngram']}-gramas de palavras (passo {r['stride']} no índice), limiar de sobreposição "
             f"{r['threshold']:.0%}, mais igualdade exata de textos normalizados.\n",
             f"Índice: {r['eval_index']['n_texts']} textos, {r['eval_index']['n_ngrams']} n-gramas, "
             f"{len(r['eval_index']['tasks'])} tarefas.\n",
             "Além da contaminação, sai o texto \"spinado\" (sinônimos trocados automaticamente, com marcas "
             "como \"situaçãeste\"), vindo de fazendas de conteúdo na web.\n",
             "| Arquivo | Entrada | Mantidos | Removidos (exato) | Removidos (n-grama) | Removidos (spinado) "
             "| Negativos removidos |",
             "|---|---|---|---|---|---|---|"]
    for f, s in r["files"].items():
        lines.append(f"| `{f}` | {s['in']} | {s['kept']} | {s['dropped'].get('exact', 0)} | "
                     f"{s['dropped'].get('ngram', 0)} | {s['dropped'].get('spun', 0)} | {s['negatives_dropped']} |")
    lines.append("\nTarefas indexadas (número de textos):\n")
    lines += [f"- {t}: {c}" for t, c in sorted(r["eval_index"]["tasks"].items())]
    return "\n".join(lines) + "\n"
