"""Avaliação no dev set (seção 2.5): escolhe checkpoints e coeficientes de merge sem tocar nos testes.

Para cada exemplo do dev, a query precisa recuperar o seu positivo num conjunto formado por
todos os positivos e negativos do dev da mesma tarefa. Métricas por tarefa: MRR@10,
nDCG@10 e Recall@10 (um relevante por query). Agregado: média simples entre tarefas
(retrieval, sts, bitext, ...), para nenhuma tarefa dominar pelo número de exemplos.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from ..modeling.embedder import format_query


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for r in rows:
        g[r.get("task", "retrieval")].append(r)
    return g


@torch.no_grad()
def dev_eval(model, rows: list[dict], batch_size: int = 64, k: int = 10) -> dict:
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
