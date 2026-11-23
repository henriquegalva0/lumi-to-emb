"""Composição dos conjuntos por estágio (seção 2.7, Tabela 3 do plano) e dev set.

Componentes (atribuídos pela origem de cada exemplo):
  natural          pares estruturais da Wikipédia PT (título↔lead, seção↔texto)
  synth_retrieval  consultas sintéticas sobre documentos-semente
  synth_other      STS sintético (paráfrase + contraste)
  mmarco           mMARCO-PT (amostra)
  en_bitext        inglês (GooAQ, AllNLI) e bitexto PT↔EN
  labeled_pt       rotulados em PT — vazio por política (todos os candidatos são tarefas de
                   avaliação; ver docs). A proporção dele é redistribuída e o desvio registrado.
  long             consultas sobre trechos de documentos longos (S-long), mais uma fração de
                   exemplos curtos do S2 ("replay"), para o estágio longo não esquecer os textos curtos

O dev é separado primeiro (queries nunca vistas no treino), depois o S2 e por fim o S1, de
forma que os estágios não compartilhem exemplos. As variantes de ablação (S2-real, S2-syn,
S2-naivehn, S2-nohn) e a R1 são derivadas daqui.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict

from ..utils import EmbedError, data_dir, iter_jsonl, results_dir, write_json

log = logging.getLogger("lumi_embed.mix")

REAL = ("natural", "mmarco", "en_bitext", "labeled_pt")
SYNTH = ("synth_retrieval", "synth_other")


def component(ex: dict) -> str:
    src = ex.get("source") or {}
    if src.get("long"):
        return "long"
    if "gen" in src:
        return "synth_retrieval" if ex["task"] == "retrieval" else "synth_other"
    ds = str(src.get("dataset", ""))
    if "mmarco" in ds:
        return "mmarco"
    if "wikipedia" in ds:
        return "natural"
    if ex.get("lang", "").startswith("en") or ex["task"] == "bitext":
        return "en_bitext"
    return "labeled_pt"


def _h(s: str) -> float:
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) / 16**12


def _with_negs(ex: dict, field: str, k: int) -> dict | None:
    negs = ex.get(field) or []
    if len(negs) < k:
        return None
    out = {kk: v for kk, v in ex.items() if kk not in ("negatives", "naive_negatives", "bm25_negatives")}
    out["negatives"] = negs[:k]
    return out


def allocate(pools: dict[str, list[dict]], proportions: dict[str, float], size: int, k: int,
             taken: set[str]) -> tuple[list[dict], dict]:
    """Amostra `size` exemplos seguindo as proporções; pula usados e os sem negativos suficientes."""
    avail = {c: p for c, p in proportions.items() if p > 0 and pools.get(c)}
    missing = {c: p for c, p in proportions.items() if p > 0 and not pools.get(c)}
    total_p = sum(avail.values())
    if not avail:
        raise EmbedError("Nenhum componente disponível para montar o conjunto.")
    out, realized = [], {}
    for c, p in avail.items():
        want = int(round(size * p / total_p))
        got = 0
        for ex in pools[c]:
            if got >= want:
                break
            if ex["id"] in taken:
                continue
            item = _with_negs(ex, "negatives", k)
            if item is None:
                continue
            out.append(item)
            taken.add(ex["id"])
            got += 1
        realized[c] = got
    info = {"requested": size, "realized": realized, "redistributed_from": list(missing),
            "proportions_requested": proportions}
    return out, info


LONG_DEV_TASK = "retrieval_long"  # no dev, o longo é uma tarefa à parte (métrica e distratores próprios)


def audit_hit_ids(set_key: str = "S2/train") -> set[str]:
    """Ids de exemplos acusados pela auditoria de contaminação (ficam fora do replay)."""
    f = results_dir("contamination", "audit.json")
    if not f.exists():
        return set()
    sets = json.loads(f.read_text()).get("sets") or {}
    return {h["id"] for h in (sets.get(set_key) or {}).get("hits") or [] if h.get("id")}


def _write(name: str, split: str, rows: list[dict]) -> int:
    path = data_dir("sets", name, f"{split}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "split": split}, ensure_ascii=False) + "\n")
    return len(rows)


def run_mix(cfg: dict, only: list[str] | None = None) -> dict:
    """Monta todos os conjuntos; com `only`, grava só os pedidos (os outros são recalculados, de forma
    determinística, para as exclusões entre conjuntos, mas não regravados) e atualiza o manifesto."""
    mcfg = cfg["mix"]

    def put(name: str, split: str, rows: list[dict]) -> int:
        return len(rows) if only and name not in only else _write(name, split, rows)

    seed = int(cfg.get("seed", 1234))
    pools: dict[str, list[dict]] = defaultdict(list)
    for f in mcfg["inputs"]:
        p = data_dir(f)
        if not p.exists():
            log.warning("Entrada ausente: %s", p)
            continue
        for ex in iter_jsonl(p):
            pools[component(ex)].append(ex)
    for c in pools:
        pools[c].sort(key=lambda e: _h(f"{seed}|{e['id']}"))  # ordem determinística
    log.info("Componentes: %s", {c: len(v) for c, v in pools.items()})
    taken: set[str] = set()
    manifest = {"seed": seed, "pool_sizes": {c: len(v) for c, v in pools.items()}, "sets": {}}

    # dev: queries nunca vistas, das mesmas distribuições do treino
    dev = []
    for c in ("natural", "synth_retrieval", "synth_other", "mmarco", "en_bitext"):
        n = int(mcfg["dev_per_component"])
        got = 0
        for ex in pools.get(c, []):
            if got >= n:
                break
            item = _with_negs(ex, "negatives", 1)
            if item:
                dev.append(item)
                taken.add(ex["id"])
                got += 1
    long_dev = []
    for ex in pools.get("long", [])[: int(mcfg["dev_long"])]:
        item = _with_negs(ex, "negatives", 1)
        if item:
            long_dev.append(item)
            taken.add(ex["id"])

    s2, info2 = allocate(pools, mcfg["s2"]["proportions"], int(mcfg["s2"]["size"]), int(mcfg["s2"]["negatives"]), taken)
    s1, info1 = allocate(pools, mcfg["s1"]["proportions"], int(mcfg["s1"]["size"]), int(mcfg["s1"]["negatives"]), taken)
    rng = random.Random(seed)
    for rows in (s1, s2):
        rng.shuffle(rows)
    manifest["sets"]["S1"] = {**info1, "train": put("S1", "train", s1), "dev": put("S1", "dev", dev)}
    manifest["sets"]["S2"] = {**info2, "train": put("S2", "train", s2), "dev": put("S2", "dev", dev)}

    # ablações derivadas do S2 (mesmos exemplos, ou mesma escala a partir de um só tipo de dado)
    k2 = int(mcfg["s2"]["negatives"])
    by_id = {e["id"]: e for c in pools.values() for e in c}
    naive = [x for x in (_with_negs(by_id[r["id"]], "naive_negatives", k2) for r in s2) if x]
    manifest["sets"]["S2-naivehn"] = {"train": put("S2-naivehn", "train", naive)}
    nohn = [{**r, "negatives": []} for r in s2]
    manifest["sets"]["S2-nohn"] = {"train": put("S2-nohn", "train", nohn)}
    for name, comps in (("S2-real", REAL), ("S2-syn", SYNTH)):
        props = {c: p for c, p in mcfg["s2"]["proportions"].items() if c in comps}
        used = {r["id"] for r in s1} | {r["id"] for r in dev}  # podem repetir exemplos do S2, nunca do S1/dev
        rows, info = allocate(pools, props, int(mcfg["s2"]["size"]), k2, set(used))
        rng.shuffle(rows)
        manifest["sets"][name] = {**info, "train": put(name, "train", rows)}

    # contexto longo: exemplos longos + replay de exemplos curtos do S2 (sem os acusados pela auditoria)
    longs = [x for x in (_with_negs(e, "negatives", k2) for e in pools.get("long", []) if e["id"] not in taken) if x]
    frac = float((mcfg.get("long") or {}).get("replay_s2", 0.0))
    replay, excluded = [], audit_hit_ids()
    if longs and frac > 0:
        cand = [r for r in s2 if r["id"] not in excluded]
        n_rep = min(len(cand), int(round(len(longs) * frac / (1 - frac))))
        replay = random.Random(seed + 17).sample(cand, n_rep)
    long_train = longs + replay
    random.Random(seed + 18).shuffle(long_train)
    long_dev_rows = [{**r, "task": LONG_DEV_TASK} for r in long_dev]
    manifest["sets"]["long"] = {"train": put("long", "train", long_train),
                                "dev": put("long", "dev", long_dev_rows),
                                "long_examples": len(longs), "replay_s2": len(replay),
                                "replay_fraction": frac, "replay_excluded_by_audit": len(excluded)}

    # R1: receita do Mamba Retriever (mMARCO-PT + negativos BM25 oficiais)
    r1_path = data_dir(mcfg["r1_input"])
    if r1_path.exists():
        r1 = []
        for ex in iter_jsonl(r1_path):
            if ex.get("r1") and len(ex.get("bm25_negatives") or []) >= 1:
                item = {k: v for k, v in ex.items() if k not in ("bm25_negatives", "negatives")}
                item["negatives"] = [{"text": t, "score": None} for t in ex["bm25_negatives"][: int(mcfg["r1_negatives"])]]
                r1.append(item)
        manifest["sets"]["R1"] = {"train": put("R1", "train", r1)}
    if only:  # mantém o registro dos conjuntos que não foram regravados
        old_file = data_dir("sets", "manifest.json")
        old = json.loads(old_file.read_text()) if old_file.exists() else {"sets": {}}
        old.setdefault("sets", {}).update({k: v for k, v in manifest["sets"].items() if k in only})
        old["pool_sizes"] = manifest["pool_sizes"]
        manifest = old
    write_json(data_dir("sets", "manifest.json"), manifest)
    log.info("Conjuntos: %s", {k: v.get("train") for k, v in manifest["sets"].items()})
    return manifest
