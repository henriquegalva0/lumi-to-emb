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
# 2: o extrator passou a ler task.corpus/task.queries (antes, 13 tarefas entravam vazias no índice)
INDEX_VERSION = 2


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


# Tarefas no formato antigo do mteb (ex.: retrieval do mteb_pt e do LongEmbed) guardam os textos em
# task.corpus / task.queries, e não em task.dataset: o mteb só os converte na hora de avaliar.
LEGACY_ATTRS = ("corpus", "queries", "instructions")


def task_strings(task):
    """Todas as strings de uma tarefa carregada: task.dataset e os atributos do formato antigo."""
    yield from _iter_strings(getattr(task, "dataset", None))
    for attr in LEGACY_ATTRS:
        yield from _iter_strings(getattr(task, attr, None))


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


def _index_task(task, hasher: "Hasher", n: int, stride: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Hashes dos textos e dos n-gramas de uma tarefa de avaliação."""
    name = task.metadata.name
    try:
        task.load_data()
    except Exception as exc:  # noqa: BLE001
        raise EmbedError(f"Não foi possível carregar a tarefa de avaliação {name}: {exc}") from exc
    texts, grams, count = [], [], 0
    for s in task_strings(task):
        if len(s) < 20:
            continue
        texts.append(Hasher.text_hash(s))
        g = hasher.ngrams(s, n, stride)
        if len(g):
            grams.append(g)
        count += 1
    if hasattr(task, "unload_data"):
        task.unload_data()
    if count == 0:
        raise EmbedError(f"A tarefa de avaliação {name} carregou sem nenhum texto: o índice de contaminação "
                         "ficaria vazio para ela. Veja onde a tarefa guarda os dados (task_strings).")
    tx = np.unique(np.array(texts, dtype=np.uint64))
    ng = np.unique(np.concatenate(grams)) if grams else np.empty(0, dtype=np.uint64)
    return tx, ng, count


def build_index(cfg: dict) -> dict:
    """Constrói (ou reaproveita) o índice de n-gramas dos textos de avaliação.

    Cada tarefa é salva assim que indexada (data/embed/decontam/tasks_n<N>_s<S>/<tarefa>.npz): se o job
    morrer no meio (memória, tempo), o reenvio continua de onde parou. `skip_tasks` exclui tarefas do
    índice (desvio da política; registre o motivo na ficha dos dados).
    """
    dcfg = cfg["decontamination"]
    out_dir = data_dir("decontam")
    idx_file = out_dir / "eval_index.npz"
    meta_file = out_dir / "eval_index.json"
    skip = sorted(dcfg.get("skip_tasks") or [])
    key = {k: dcfg[k] for k in ("benchmarks", "ngram", "stride")}
    key["index_version"] = INDEX_VERSION
    if skip:  # só entra na chave quando usada: índices montados antes desta opção continuam válidos
        key["skip_tasks"] = skip
    if idx_file.exists() and meta_file.exists() and json.loads(meta_file.read_text()).get("key") == key:
        z = np.load(idx_file)
        return {"ngrams": z["ngrams"], "texts": z["texts"], "meta": json.loads(meta_file.read_text())}
    hasher = Hasher()
    n, stride = int(dcfg["ngram"]), int(dcfg["stride"])
    task_dir = out_dir / f"tasks_n{n}_s{stride}"
    task_dir.mkdir(parents=True, exist_ok=True)
    grams, texts, per_task, redone = [], [], {}, []
    t0 = time.time()
    for task in eval_tasks(dcfg["benchmarks"]):
        name = task.metadata.name
        if name in skip:
            per_task[name] = "excluída (skip_tasks)"
            log.warning("  %s: excluída do índice (decontamination.skip_tasks)", name)
            continue
        f = task_dir / f"{name}.npz"
        if f.exists() and int(np.load(f)["count"]) == 0:
            log.warning("  %s: índice salvo vazio (versão anterior do extrator); refazendo", name)
            f.unlink()
            redone.append(name)
        if f.exists():
            z = np.load(f)
            tx, ng, count = z["texts"], z["ngrams"], int(z["count"])
            log.info("  %s: %d textos (índice salvo)", name, count)
        else:
            tx, ng, count = _index_task(task, hasher, n, stride)
            tmp = task_dir / f"{name}.tmp.npz"
            np.savez(tmp, texts=tx, ngrams=ng, count=np.array(count))
            tmp.replace(f)
            log.info("  %s: %d textos", name, count)
        texts.append(tx)
        grams.append(ng)
        per_task[name] = count
    ng = np.unique(np.concatenate(grams)) if grams else np.empty(0, dtype=np.uint64)
    tx = np.unique(np.concatenate(texts)) if texts else np.empty(0, dtype=np.uint64)
    np.savez(idx_file, ngrams=ng, texts=tx)
    meta = {"key": key, "tasks": per_task, "n_ngrams": int(len(ng)), "n_texts": int(len(tx)),
            "redone_empty": redone, "seconds": round(time.time() - t0, 1)}
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


# --------------------------------------------------------------------------- auditoria posterior

AUDIT_SETS = ["S1", "S2", "S2-real", "S2-syn"]  # conjuntos usados pelos checkpoints treinados


def _texts_of(ex: dict):
    yield "query", ex.get("query") or ""
    yield "positive", ex.get("positive") or ""
    for field in ("negatives", "naive_negatives", "bm25_negatives"):
        for x in ex.get(field) or []:
            yield "negative", x["text"] if isinstance(x, dict) else x


def run_audit(cfg: dict) -> dict:
    """Confere os conjuntos já montados (data/embed/sets) contra o índice atual, sem alterar nada.

    Serve para verificar, depois do treino, tarefas que o índice usado no filtro não cobria. Cada
    acerto é atribuído às tarefas que entraram vazias no índice do filtro (lidas do relatório do
    filtro, results/embed/contamination/report.json) e `outra` quando não está em nenhuma delas.
    Todos os acertos ficam em `hits` (id, campo, tarefas). Saída: results/embed/contamination/audit.*.
    """
    dcfg = cfg["decontamination"]
    index = build_index(cfg)
    hasher = Hasher()
    n, stride, thr = int(dcfg["ngram"]), int(dcfg["stride"]), float(dcfg["threshold"])
    task_dir = data_dir("decontam") / f"tasks_n{n}_s{stride}"
    focus = audit_focus(index)
    per = {}
    for t in focus:
        z = np.load(task_dir / f"{t}.npz")
        per[t] = {"ngrams": z["ngrams"], "texts": z["texts"]}
    report = {"threshold": thr, "ngram": n, "focus_tasks": focus, "index_tasks": index["meta"]["tasks"], "sets": {}}
    for name in AUDIT_SETS:
        for split in ("train", "dev"):
            path = data_dir("sets", name, f"{split}.jsonl")
            if not path.exists():
                continue
            st = {"examples": 0, "examples_hit": 0, "by_field": Counter(), "by_reason": Counter(),
                  "by_task": Counter(), "examples_by_task": Counter(), "samples": [], "hits": []}
            for ex in iter_jsonl(path):
                st["examples"] += 1
                hit, ex_tasks = False, set()
                for field, text in _texts_of(ex):
                    if not text:
                        continue
                    why = is_contaminated(hasher, text, index, n, thr)
                    if not why:
                        continue
                    hit = True
                    st["by_field"][field] += 1
                    st["by_reason"][why] += 1
                    tasks = [t for t in focus if is_contaminated(hasher, text, per[t], n, thr)] or ["outra"]
                    for t in tasks:
                        st["by_task"][t] += 1
                    ex_tasks.update(tasks)
                    st["hits"].append({"id": ex.get("id"), "field": field, "reason": why, "tasks": tasks})
                    if len(st["samples"]) < 20:
                        st["samples"].append({"id": ex.get("id"), "field": field, "reason": why, "tasks": tasks,
                                              "source": ex.get("source"), "text": text[:300]})
                st["examples_hit"] += hit
                for t in ex_tasks:
                    st["examples_by_task"][t] += 1
            key = f"{name}/{split}"
            report["sets"][key] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items()}
            log.info("auditoria %s: %d de %d exemplos com algum texto da avaliação %s", key, st["examples_hit"],
                     st["examples"], dict(st["by_task"]))
    write_json(results_dir("contamination", "audit.json"), report)
    (results_dir("contamination") / "audit.md").write_text(render_audit(report), encoding="utf-8")
    return report


def audit_focus(index: dict) -> list[str]:
    """Tarefas a conferir à parte: as que tinham 0 textos no índice usado pelo filtro (relatório dele).

    Não depende de qual execução refez o índice (um job reiniciado já encontraria os caches prontos).
    """
    rep = results_dir("contamination", "report.json")
    if rep.exists():
        tasks = json.loads(rep.read_text()).get("eval_index", {}).get("tasks", {})
        focus = sorted(t for t, c in tasks.items() if c == 0)
    else:
        focus = sorted(index["meta"].get("redone_empty") or [])
    return [t for t in focus if isinstance(index["meta"]["tasks"].get(t), int) and index["meta"]["tasks"][t] > 0]


def render_audit(r: dict) -> str:
    lines = ["# Auditoria de contaminação dos conjuntos de treino\n",
             "Confere os conjuntos já montados contra o índice da avaliação corrigido (versão "
             f"{INDEX_VERSION}), sem alterar os dados. Mesmo critério do filtro: texto inteiro igual ou "
             f"≥ {r['threshold']:.0%} dos {r['ngram']}-gramas no índice. Tarefas conferidas à parte (entraram "
             "vazias no índice usado no filtro): " + (", ".join(r["focus_tasks"]) or "nenhuma") + ".\n",
             "| Conjunto | Exemplos | Com acerto | Query | Positivo | Negativo | Exemplos por tarefa |",
             "|---|---|---|---|---|---|---|"]
    for k, s in r["sets"].items():
        f = s["by_field"]
        by = s.get("examples_by_task") or s["by_task"]
        tasks = ", ".join(f"{t}: {c}" for t, c in sorted(by.items(), key=lambda kv: -kv[1])) or "—"
        lines.append(f"| `{k}` | {s['examples']} | {s['examples_hit']} | {f.get('query', 0)} | "
                     f"{f.get('positive', 0)} | {f.get('negative', 0)} | {tasks} |")
    samples = [(k, x) for k, s in r["sets"].items() for x in s["samples"]]
    if samples:
        lines.append("\nExemplos (até 20 por conjunto):\n")
        lines += [f"- `{k}` {x['field']} ({x['reason']}; {', '.join(x['tasks'])}): {x['text'][:160]!r}"
                  for k, x in samples[:60]]
    return "\n".join(lines) + "\n"
