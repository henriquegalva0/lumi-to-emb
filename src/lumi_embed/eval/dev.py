"""Avaliação no dev set (seção 2.5): escolhe checkpoints e coeficientes de merge sem tocar nos testes.

Para cada exemplo do dev, a query precisa recuperar o seu positivo num conjunto formado por
todos os positivos e negativos do dev da mesma tarefa, mais um conjunto fixo de distratores
(positivos do treino, nunca do dev). Sem os distratores, cada query concorria com poucas centenas
de documentos e os modelos ficavam todos perto de 0,98: as comparações (escolha de LR, de
checkpoints e de merges) caíam no ruído. Métricas por tarefa: MRR@10, nDCG@10 e Recall@10 (um
relevante por query). Agregado: média simples entre tarefas (retrieval, sts, bitext, ...).
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..modeling.embedder import format_query

DISTRACTORS_FILE = "dev_distractors.jsonl"
DISTRACTORS_PER_TASK = {"retrieval": 20000, "sts": 5000, "bitext": 5000}


@dataclass
class DevSet:
    rows: list[dict]
    distractors: dict[str, list[str]] = field(default_factory=dict)


def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def dev_distractors(sets_dir: Path, per_task: dict[str, int] | None = None) -> dict[str, list[str]]:
    """Distratores fixos por tarefa: positivos do S1/S2 de treino, sorteados por hash (determinístico).

    Montados uma vez em <sets>/dev_distractors.jsonl; textos que são positivos no dev ficam de fora.
    """
    per_task = per_task or DISTRACTORS_PER_TASK
    f = Path(sets_dir) / DISTRACTORS_FILE
    if not f.exists():
        dev_pos = set()
        for name in ("S1", "S2"):
            p = Path(sets_dir) / name / "dev.jsonl"
            if p.exists():
                dev_pos |= {json.loads(x)["positive"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}
        heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
        seen: set[int] = set()
        for name in ("S2", "S1"):
            p = Path(sets_dir) / name / "train.jsonl"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    task, text = r.get("task", "retrieval"), r["positive"]
                    n = per_task.get(task, 0)
                    if not n or text in dev_pos:
                        continue
                    h = _h(text)
                    if h in seen:
                        continue
                    seen.add(h)
                    heap = heaps[task]  # max-heap (por -h) com os n menores hashes
                    if len(heap) < n:
                        heapq.heappush(heap, (-h, text))
                    elif -heap[0][0] > h:
                        heapq.heapreplace(heap, (-h, text))
        if not heaps:
            return {}
        tmp = f.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for task in sorted(heaps):
                for neg_h, text in sorted(heaps[task], key=lambda x: -x[0]):
                    fh.write(json.dumps({"task": task, "text": text}, ensure_ascii=False) + "\n")
        tmp.replace(f)
    out: dict[str, list[str]] = defaultdict(list)
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["task"]].append(r["text"])
    return dict(out)


def load_dev_set(path: Path) -> DevSet:
    """Dev de <sets>/<conjunto>/dev.jsonl com os distratores de <sets>/."""
    path = Path(path)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return DevSet(rows, dev_distractors(path.parent.parent))


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for r in rows:
        g[r.get("task", "retrieval")].append(r)
    return g


@torch.no_grad()
def dev_eval(model, rows: "list[dict] | DevSet", batch_size: int = 64, k: int = 10) -> dict:
    distractors: dict[str, list[str]] = {}
    if isinstance(rows, DevSet):
        rows, distractors = rows.rows, rows.distractors
    if not rows:
        return {}
    spec = model.spec
    out: dict = {"tasks": {}}
    for task, items in sorted(_group(rows).items()):
        symmetric = task in ("sts", "bitext", "clustering")
        docs, doc_index = [], {}

        def add(text: str) -> int:
            if text not in doc_index:
                doc_index[text] = len(docs)
                docs.append(text)
            return doc_index[text]

        targets = []
        for r in items:
            targets.append(add(r["positive"]))
            for n in r.get("negatives") or []:
                add(n["text"] if isinstance(n, dict) else n)
        for t in distractors.get(task, []):
            add(t)
        queries = [format_query(r["query"], r.get("instruction"), spec) for r in items]
        if symmetric:
            # nas tarefas simétricas a instrução vai nos dois lados
            inst = items[0].get("instruction")
            doc_inputs = [format_query(d, inst, spec) for d in docs]
        else:
            doc_inputs = docs
        q = model.encode(queries, batch_size=batch_size)
        d = model.encode(doc_inputs, batch_size=batch_size)
        scores = q @ d.T
        tgt = torch.tensor(targets)
        pos_scores = scores[torch.arange(len(items)), tgt].unsqueeze(1)
        ranks = (scores > pos_scores).sum(dim=1) + 1  # 1 = primeiro lugar
        mrr = float(torch.where(ranks <= k, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float)).mean())
        ndcg = float(torch.where(ranks <= k, 1.0 / torch.log2(ranks.float() + 1), torch.zeros(len(items))).mean())
        rec = float((ranks <= k).float().mean())
        out["tasks"][task] = {"n": len(items), "pool": len(docs), f"mrr@{k}": round(mrr, 4),
                              f"ndcg@{k}": round(ndcg, 4), f"recall@{k}": round(rec, 4)}
    vals = [v[f"ndcg@{k}"] for v in out["tasks"].values()]
    out[f"mean_ndcg@{k}"] = round(sum(vals) / len(vals), 4)
    return out


def borda(results: dict[str, dict], metric: str = "ndcg@10") -> dict[str, int]:
    """Votos Borda entre candidatos, tarefa a tarefa (empates dividem os pontos)."""
    names = list(results)
    tasks = sorted({t for r in results.values() for t in r.get("tasks", {})})
    votes = dict.fromkeys(names, 0.0)
    for t in tasks:
        vals = {n: results[n]["tasks"].get(t, {}).get(metric, -math.inf) for n in names}
        for n in names:
            votes[n] += sum(1.0 for m in names if vals[n] > vals[m]) + 0.5 * sum(
                1.0 for m in names if m != n and vals[n] == vals[m])
    return {n: round(v, 1) for n, v in votes.items()}
