"""Fontes reais (D_real), sementes da geração sintética e texto cru do S0 — seção 2.2 do plano.

Cada função lê uma fonte com revisão fixada, converte para o schema único e grava em
data/embed/raw/<nome>.jsonl (ou seeds/, sets/s0/). Nenhuma fonte aqui é tarefa de
avaliação (MTEB-BR, MTEB-PT, LongEmbed, MTEB eng v2); a sobreposição de documentos com os
corpora de avaliação é medida e removida depois, em decontaminate.py.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

from lumi_ctx.haystack import EMPTY_PARENS, MAGIC_RE, SPACES, clean_article  # noqa: F401

from ..utils import data_dir, write_json
from .hub import download, iter_rows
from .schema import make_example, norm_text

log = logging.getLogger("lumi_embed.sources")

HEADING_RE = re.compile(r"^[A-ZÀ-Ý0-9][^.!?;:]{1,78}$")
SENT_END = re.compile(r"[.!?…][\"”’»)\]]*$")


def _write(path: Path, rows: list[dict]) -> int:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return len(rows)


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def _clean_par(text: str) -> str:
    return SPACES.sub(" ", EMPTY_PARENS.sub("", text)).strip()


# --------------------------------------------------------------------------- Wikipédia PT


def wiki_sections(text: str, min_par: int = 120) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """(parágrafos do lead, [(título de seção, parágrafos)]) a partir do texto bruto da Wikipédia."""
    lines = [ln.strip() for ln in text.split("\n")]
    lead: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for i, ln in enumerate(lines):
        if not ln:
            continue
        # próxima linha de prosa (pula tabelas e listas, até 3 linhas à frente)
        ahead = [x for x in lines[i + 1 : i + 6] if x][:3]
        nxt = next((x for x in ahead if x.count("|") < 2 and not x.startswith(("*", "#", "-", "{", "!"))), "")
        is_heading = bool(HEADING_RE.match(ln)) and not SENT_END.search(ln) and len(nxt) >= min_par
        if is_heading:
            current = []
            sections.append((ln, current))
            continue
        if len(ln) >= min_par and SENT_END.search(ln) and ln.count("|") < 2:
            (lead if current is None else current).append(_clean_par(ln))
    return lead, [(h, ps) for h, ps in sections if ps]


def wiki_pairs(cfg: dict, seed: int) -> dict:
    """Pares naturais título↔lead e (título: seção)↔texto da seção."""
    src = cfg["sources"]["wikipedia"]
    inst = cfg["instructions"]["title"]
    n_target, max_words = int(src["n_pairs"]), int(src.get("max_words", 300))
    rng = random.Random(seed)
    rows, seen_docs = [], 0
    for art in iter_rows(src["repo"], src["revision"], src["files"], ["id", "title", "text"], seed=seed + 11):
        text = art["text"] or ""
        if len(text) < 2000 or MAGIC_RE.search(text):
            continue
        seen_docs += 1
        lead, sections = wiki_sections(text)
        meta = {"dataset": src["repo"], "revision": src["revision"][:12], "doc": art["id"], "kind": None}
        if lead:
            rows.append(make_example("retrieval", "pt-BR", inst, art["title"],
                                     _truncate_words(" ".join(lead), max_words), {**meta, "kind": "title-lead"}))
        for h, ps in rng.sample(sections, min(len(sections), int(src.get("sections_per_article", 2)))):
            rows.append(make_example("retrieval", "pt-BR", inst, f"{art['title']}: {h}",
                                     _truncate_words(" ".join(ps), max_words), {**meta, "kind": "section"}))
        if len(rows) >= n_target:
            break
    n = _write(data_dir("raw", "wiki_pt.jsonl"), rows[:n_target])
    return {"wiki_pt": n, "articles_used": seen_docs}


# --------------------------------------------------------------------------- mMARCO-PT


def _read_tsv(path: str, keep=None, cols=2):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < cols:
                continue
            if keep is None or parts[0] in keep:
                yield parts


def mmarco_pt(cfg: dict, seed: int) -> dict:
    """mMARCO-PT (tradução do MS MARCO): queries de treino, positivo e negativos BM25 oficiais.

    `r1_queries` queries recebem até 63 negativos BM25 (receita do Mamba Retriever, R1);
    as demais ficam com poucos (a mineração com o teacher troca esses negativos depois).
    """
    src = cfg["sources"]["mmarco"]
    rng = random.Random(seed)
    q_path = download(src["repo"], src["revision"], src["queries"])
    t_path = download(src["repo"], src["revision"], src["triples"])
    c_path = download(src["repo"], src["revision"], src["collection"])
    queries = {qid: text for qid, text in _read_tsv(q_path)}
    # Só ~metade das queries de treino aparece nos trios oficiais (positivo + negativos BM25);
    # sorteamos entre elas, senão metade das escolhidas ficaria sem positivo.
    in_triples = {parts[0] for parts in _read_tsv(t_path, cols=3)}
    qids = sorted(q for q in queries if q in in_triples)
    log.info("mMARCO: %d queries de treino, %d com trios BM25", len(queries), len(qids))
    rng.shuffle(qids)
    chosen = qids[: int(src["n_queries"])]
    r1 = set(chosen[: int(src["r1_queries"])])
    chosen = set(chosen)
    pos: dict[str, str] = {}
    negs: dict[str, list[str]] = {}
    for qid, p, n in _read_tsv(t_path, chosen, cols=3):
        pos.setdefault(qid, p)
        lst = negs.setdefault(qid, [])
        cap = int(src["r1_negatives"]) if qid in r1 else int(src.get("other_negatives", 4))
        if len(lst) < cap and n not in lst:
            lst.append(n)
    need = set(pos.values()) | {n for lst in negs.values() for n in lst}
    passages = {pid: text for pid, text in _read_tsv(c_path, need)}
    inst = cfg["instructions"]["qa"]
    rows = []
    for qid in sorted(pos):
        p = passages.get(pos[qid])
        if not p:
            continue
        bm25 = [passages[n] for n in negs.get(qid, []) if n in passages]
        rows.append(make_example("retrieval", "pt-BR", inst, queries[qid], p,
                                 {"dataset": src["repo"], "revision": src["revision"][:12], "qid": qid},
                                 bm25_negatives=bm25, r1=qid in r1))
    n = _write(data_dir("raw", "mmarco_pt.jsonl"), rows)
    return {"mmarco_pt": n, "r1_queries": sum(1 for r in rows if r["r1"])}


# --------------------------------------------------------------------------- bitexto e inglês


def bitext(cfg: dict, seed: int) -> dict:
    """Pares PT↔EN (OPUS via sentence-transformers): metade PT→EN, metade EN→PT."""
    out, stats = [], {}
    inst = cfg["instructions"]["bitext"]
    for src in cfg["sources"]["bitext"]:
        n_take, k = int(src["n"]), 0
        for row in iter_rows(src["repo"], src["revision"], src["files"], ["english", "non_english"], seed=seed + 3):
            en, pt = norm_text(row["english"]), norm_text(row["non_english"])
            if len(en.split()) < 4 or len(pt.split()) < 4:
                continue
            meta = {"dataset": src["repo"], "revision": src["revision"][:12]}
            if k % 2 == 0:
                out.append(make_example("bitext", "pt-BR→en", inst, pt, en, meta))
            else:
                out.append(make_example("bitext", "en→pt-BR", inst, en, pt, meta))
            k += 1
            if k >= n_take:
                break
        stats[src["repo"]] = k
    stats["bitext"] = _write(data_dir("raw", "bitext.jsonl"), out)
    return stats


def english(cfg: dict, seed: int) -> dict:
    """Pequena fração em inglês para reter desempenho (MTEB eng v2): GooAQ e AllNLI."""
    srcs = cfg["sources"]["english"]
    out = []
    g = srcs["gooaq"]
    for k, row in enumerate(iter_rows(g["repo"], g["revision"], g["files"], ["question", "answer"], seed=seed + 5)):
        if k >= int(g["n"]):
            break
        out.append(make_example("retrieval", "en", cfg["instructions"]["en_qa"], row["question"], row["answer"],
                                {"dataset": g["repo"], "revision": g["revision"][:12]}))
    a = srcs["allnli"]
    for k, row in enumerate(iter_rows(a["repo"], a["revision"], a["files"], ["anchor", "positive", "negative"],
                                      seed=seed + 6)):
        if k >= int(a["n"]):
            break
        out.append(make_example("sts", "en", cfg["instructions"]["en_sts"], row["anchor"], row["positive"],
                                {"dataset": a["repo"], "revision": a["revision"][:12]},
                                negatives=[{"text": norm_text(row["negative"]), "score": None, "kind": "nli"}]))
    return {"english": _write(data_dir("raw", "english.jsonl"), out)}


# --------------------------------------------------------------------------- sementes e S0


def _window(paragraphs: list[str], rng: random.Random, min_words: int, max_words: int) -> str | None:
    if not paragraphs:
        return None
    start = rng.randrange(len(paragraphs))
    words: list[str] = []
    for p in paragraphs[start:]:
        words.extend(p.split())
        if len(words) >= max_words:
            break
    if len(words) < min_words:
        return None
    return " ".join(words[:max_words])


def seeds(cfg: dict, seed: int) -> dict:
    """Documentos-semente (Wikipédia PT + FineWeb-2 PT, sem domínios .pt) e sementes longas."""
    scfg = cfg["seeds"]
    rng = random.Random(seed)
    out, long_out = [], []
    w = cfg["sources"]["wikipedia"]
    n_wiki, n_long = int(scfg["n_wikipedia"]), int(scfg["n_long"])
    for art in iter_rows(w["repo"], w["revision"], w["files"], ["id", "title", "text"], seed=seed + 21):
        if len(out) >= n_wiki and len(long_out) >= n_long:
            break
        text = art["text"] or ""
        if len(text) < 2000 or MAGIC_RE.search(text):
            continue
        clean = clean_article(text, 80)
        pars = [p for p in clean.split("\n") if p]
        if len(long_out) < n_long and len(clean) >= int(scfg["long_min_chars"]):
            long_out.append({"id": f"wl-{art['id']}", "text": clean[: int(scfg["long_max_chars"])],
                             "source": "wikipedia", "title": art["title"]})
        elif len(out) < n_wiki:
            win = _window(pars, rng, int(scfg["min_words"]), int(scfg["max_words"]))
            if win:
                out.append({"id": f"w-{art['id']}", "text": win, "source": "wikipedia", "title": art["title"]})
    f = cfg["sources"]["fineweb2"]
    k = 0
    for row in iter_rows(f["repo"], f["revision"], f["files"], ["id", "text", "url"], seed=seed + 22):
        if k >= int(scfg["n_fineweb2"]):
            break
        url = (row.get("url") or "").split("/")[2] if "//" in (row.get("url") or "") else ""
        if url.endswith(".pt") or MAGIC_RE.search(row["text"] or ""):  # mantém o português brasileiro
            continue
        pars = [norm_text(p) for p in (row["text"] or "").split("\n") if len(p.split()) >= 20]
        win = _window(pars, rng, int(scfg["min_words"]), int(scfg["max_words"]))
        if win:
            out.append({"id": f"f-{row['id']}", "text": win, "source": "fineweb2", "url_domain": url})
            k += 1
    rng.shuffle(out)
    return {"seeds": _write(data_dir("seeds", "seeds.jsonl"), out),
            "long_seeds": _write(data_dir("seeds", "long_seeds.jsonl"), long_out)}


def s0_text(cfg: dict, seed: int) -> dict:
    """Texto cru do MNTP: ~80 % PT (sementes) + ~20 % multidomínio em inglês (FineWeb-Edu)."""
    import json

    scfg = cfg["s0"]
    rng = random.Random(seed)
    pt = [json.loads(x)["text"] for x in data_dir("seeds", "seeds.jsonl").read_text(encoding="utf-8").splitlines()]
    rng.shuffle(pt)
    n_total = int(scfg["n_texts"])
    n_en = int(round(n_total * float(scfg["multidomain_fraction"])))
    rows = [{"text": t, "lang": "pt-BR"} for t in pt[: n_total - n_en]]
    e = cfg["sources"]["fineweb_edu"]
    for k, row in enumerate(iter_rows(e["repo"], e["revision"], e["files"], ["text"], seed=seed + 31)):
        if k >= n_en:
            break
        rows.append({"text": _truncate_words(norm_text(row["text"]), int(cfg["seeds"]["max_words"])), "lang": "en"})
    rng.shuffle(rows)
    return {"s0_texts": _write(data_dir("sets", "s0", "train.jsonl"), rows)}


SOURCES = {"wikipedia": wiki_pairs, "mmarco": mmarco_pt, "bitext": bitext, "english": english,
           "seeds": seeds, "s0": s0_text}


def run_sources(cfg: dict, names: list[str] | None = None) -> dict:
    seed = int(cfg.get("seed", 1234))
    stats = {}
    for name in names or list(SOURCES):
        log.info("Fonte: %s", name)
        stats[name] = SOURCES[name](cfg, seed)
        log.info("  %s", stats[name])
    write_json(data_dir("raw", f"stats_{'-'.join(names or ['all'])}.json"), stats)
    return stats
