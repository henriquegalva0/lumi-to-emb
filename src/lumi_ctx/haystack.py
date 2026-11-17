"""Palheiro (texto de fundo) em PT-BR — seção 3.5 da especificação.

Fluxo:
1. Streaming determinístico do parquet do Hugging Face (Wikipédia PT, revisão fixada):
   os *row groups* são lidos sob demanda (HfFileSystem + pyarrow) numa ordem
   embaralhada pela semente, e os artigos de cada *row group* também.
2. Limpeza: Unicode NFC, espaços normalizados, remoção de linhas curtas (títulos,
   listas, linhas de tabela) e de artefatos de predefinições ("( )").
3. Segmentação em frases (regex com lista de abreviações do português).
4. Acúmulo até `target_tokens` e gravação em `data/haystack_pt.parquet`
   (texto + offsets de início de frase), reutilizado entre execuções.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import Stage1Error, sha256_text

log = logging.getLogger(__name__)

POOL_FORMAT_VERSION = 1
DOC_SEPARATOR = "\n\n"

# Artigos que falam de "números mágicos" (ex.: física nuclear) confundiriam a tarefa.
MAGIC_RE = re.compile(r"n[úu]meros?\s+m[áa]gicos?", re.IGNORECASE)
EMPTY_PARENS = re.compile(r"\(\s*[,;:–—-]?\s*\)")
SPACES = re.compile(r"[ \t  -​  　]+")
SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?)])")
LIST_OR_TABLE = re.compile(r"^\s*(\{\||\|[-}+]?|!|\*|#|•|·|–\s|-\s|=)")
SENTENCE_END = re.compile(r"[.!?…][\"”’»)\]]*$")

# Fim de frase candidato: pontuação final + espaço + início provável de frase.
SPLIT_RE = re.compile(r"([.!?…][\"”’»)\]]*)(\s+)(?=[\"“«(\[]?[A-ZÀ-ÖØ-Þ0-9])")
ABBREVIATIONS = {
    "sr", "sra", "srs", "sras", "dr", "dra", "drs", "prof", "profa", "profs", "st", "sto", "sta",
    "jr", "sr", "nº", "n", "no", "nos", "p", "pp", "ed", "eds", "vol", "vols", "cap", "caps", "art",
    "arts", "fig", "figs", "ex", "av", "gen", "gal", "cel", "ten", "sgt", "maj", "cmdt", "dom", "d",
    "s", "a.c", "d.c", "km", "kg", "op", "cit", "inc", "ltda", "cia", "mr", "mrs", "ms", "vs", "séc",
    "sec", "aprox", "máx", "min", "eng", "arq", "pe", "fr", "rev", "dep", "sen", "gov", "pres",
    "exmo", "exma", "ilmo", "ilma", "v.ex", "a", "e", "c", "ca", "cf", "co", "jan", "fev", "mar",
    "abr", "jun", "jul", "ago", "set", "out", "nov", "dez", "pág", "págs", "trad", "org", "coord",
    "u.s", "e.u.a", "s.a", "lda", "mt", "sto", "bros", "esq", "dir", "pl", "obs", "tel",
}


# --------------------------------------------------------------------------- limpeza


def clean_article(text: str, min_line_chars: int = 80) -> str:
    """Limpa um artigo e devolve só os parágrafos de prosa, separados por "\\n"."""
    text = unicodedata.normalize("NFC", text or "")
    kept: list[str] = []
    for raw in text.split("\n"):
        line = SPACES.sub(" ", raw).strip()
        if len(line) < min_line_chars:
            continue  # títulos de seção, itens curtos de lista, legendas
        if LIST_OR_TABLE.match(line) or line.count("|") >= 2 or line.count("\t") >= 1:
            continue  # tabelas e listas
        letters = sum(ch.isalpha() for ch in line)
        if letters < 0.6 * len(line):
            continue  # linhas majoritariamente numéricas/símbolos
        if not SENTENCE_END.search(line):
            continue  # não termina como frase (provável fragmento)
        line = EMPTY_PARENS.sub("", line)
        line = SPACES.sub(" ", line)
        line = SPACE_BEFORE_PUNCT.sub(r"\1", line).strip()
        if len(line) >= min_line_chars:
            kept.append(line)
    return "\n".join(kept)


def _is_abbreviation(paragraph: str, end_of_word: int) -> bool:
    start = end_of_word
    while start > 0 and not paragraph[start - 1].isspace() and paragraph[start - 1] not in "(\"“«[":
        start -= 1
    word = paragraph[start:end_of_word].rstrip(".").lower()
    if not word:
        return False
    if word in ABBREVIATIONS:
        return True
    return len(word) == 1 and word.isalpha()  # iniciais: "J. R. R. Tolkien"


def sentence_starts(paragraph: str) -> list[int]:
    """Offsets (no parágrafo) onde começa cada frase. Sempre inclui 0."""
    starts = [0]
    for m in SPLIT_RE.finditer(paragraph):
        punct_start = m.start(1)
        if paragraph[punct_start] == "." and _is_abbreviation(paragraph, punct_start):
            continue
        nxt = m.end(2)
        if nxt < len(paragraph) and nxt - starts[-1] >= 2:
            starts.append(nxt)
    return starts


def article_sentence_starts(article: str) -> list[int]:
    """Inícios de frase de um artigo limpo (parágrafos separados por "\\n")."""
    starts: list[int] = []
    offset = 0
    for paragraph in article.split("\n"):
        starts.extend(offset + s for s in sentence_starts(paragraph))
        offset += len(paragraph) + 1
    return starts


# --------------------------------------------------------------------------- streaming


def _hf_paths(source: dict) -> tuple[object, list[str]]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    pattern = f"datasets/{source['repo']}@{source['revision']}/{source['files']}"
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise Stage1Error(f"Nenhum arquivo parquet em {pattern}")
    return fs, paths


def iter_articles(source: dict, seed: int):
    """Itera artigos em ordem determinística (permutação de row groups + de linhas)."""
    import pyarrow.parquet as pq

    fs, paths = _hf_paths(source)
    handles: dict[str, object] = {}
    units: list[tuple[str, int]] = []
    for path in paths:
        fh = fs.open(path, "rb", block_size=16 * 1024 * 1024)
        handles[path] = pq.ParquetFile(fh)
        units.extend((path, i) for i in range(handles[path].metadata.num_row_groups))
    log.info("Fonte %s@%s: %d arquivo(s), %d row groups", source["repo"], source["revision"][:10], len(paths), len(units))
    order = np.random.default_rng([seed, 99]).permutation(len(units))
    columns = [c for c in (source.get("id_field"), source.get("title_field"), source["text_field"]) if c]
    for k in order:
        path, rg = units[int(k)]
        table = handles[path].read_row_group(rg, columns=columns)
        rows = table.to_pylist()
        for j in np.random.default_rng([seed, 99, int(k)]).permutation(len(rows)):
            row = rows[int(j)]
            yield {
                "source_id": str(row.get(source.get("id_field") or "", "")),
                "title": str(row.get(source.get("title_field") or "", "") or ""),
                "text": row[source["text_field"]] or "",
                "file": path.split("/", 3)[-1],
                "row_group": rg,
            }


def get_backend_tokenizer(tok):
    backend = getattr(tok, "backend_tokenizer", None) or getattr(tok, "_tokenizer", None)
    if backend is None:
        raise Stage1Error(f"Tokenizer {type(tok).__name__} sem backend 'tokenizers' (fast).")
    return backend


def build_pool(cfg: dict, out_path: Path, count_tokenizer, seed: int, articles_iter_factory=None) -> dict:
    """Constrói o pool e grava em parquet. Retorna os metadados gravados.

    `articles_iter_factory(source, seed)` permite injetar uma fonte (testes locais).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    hcfg = cfg["haystack"]
    factory = articles_iter_factory or iter_articles
    backend = get_backend_tokenizer(count_tokenizer)
    errors: list[str] = []
    t0 = time.time()
    for source_idx, source in enumerate(hcfg["sources"]):
        docs: list[dict] = []
        total_tokens = 0
        stats = {"seen": 0, "too_short": 0, "magic": 0, "short_after_clean": 0}
        try:
            batch: list[dict] = []
            for art in factory(source, seed):
                stats["seen"] += 1
                if len(art["text"]) < hcfg["min_article_chars"]:
                    stats["too_short"] += 1
                    continue
                if MAGIC_RE.search(art["text"]):
                    stats["magic"] += 1
                    continue
                clean = clean_article(art["text"], hcfg["min_line_chars"])
                if len(clean) < hcfg["min_clean_chars"]:
                    stats["short_after_clean"] += 1
                    continue
                art["text"] = clean
                batch.append(art)
                if len(batch) >= 256:
                    total_tokens += _accept(batch, docs, backend)
                    batch = []
                    log.info("  pool: %d artigos, %.2fM tokens", len(docs), total_tokens / 1e6)
                    if total_tokens >= hcfg["target_tokens"]:
                        break
            if batch and total_tokens < hcfg["target_tokens"]:
                total_tokens += _accept(batch, docs, backend)
        except Exception as exc:  # noqa: BLE001 - qualquer falha de acesso leva ao fallback
            msg = f"{source['repo']}: {type(exc).__name__}: {exc}"
            log.warning("Falha ao ler a fonte do palheiro (%s). Tentando a próxima.", msg)
            errors.append(msg)
            continue
        if total_tokens < hcfg["target_tokens"]:
            msg = f"{source['repo']}: só {total_tokens} tokens (< {hcfg['target_tokens']})"
            errors.append(msg)
            log.warning("Fonte insuficiente: %s", msg)
            continue

        meta = {
            "format_version": POOL_FORMAT_VERSION,
            "source": {k: source.get(k) for k in ("repo", "revision", "files")},
            "source_index": source_idx,
            "fallback_used": source_idx > 0,
            "errors": errors,
            "seed": seed,
            "params": {k: hcfg[k] for k in ("target_tokens", "min_article_chars", "min_clean_chars", "min_line_chars")},
            "count_tokenizer": getattr(count_tokenizer, "name_or_path", "?"),
            "n_docs": len(docs),
            "n_tokens": int(total_tokens),
            "n_sentences": int(sum(len(d["sent_starts"]) for d in docs)),
            "n_chars": int(sum(len(d["text"]) for d in docs)),
            "filter_stats": stats,
            "build_seconds": round(time.time() - t0, 1),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        table = pa.table(
            {
                "doc": pa.array(range(len(docs)), pa.int32()),
                "source_id": [d["source_id"] for d in docs],
                "title": [d["title"] for d in docs],
                "text": [d["text"] for d in docs],
                "sent_starts": pa.array([d["sent_starts"] for d in docs], pa.list_(pa.int32())),
                "n_tokens": pa.array([d["n_tokens"] for d in docs], pa.int32()),
                "file": [d["file"] for d in docs],
                "row_group": pa.array([d["row_group"] for d in docs], pa.int32()),
            }
        )
        table = table.replace_schema_metadata({"lumi_ctx": json.dumps(meta, ensure_ascii=False)})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(out_path)
        log.info(
            "Pool salvo em %s: %d artigos, %.2fM tokens, %d frases (%.0fs)",
            out_path, meta["n_docs"], meta["n_tokens"] / 1e6, meta["n_sentences"], meta["build_seconds"],
        )
        return meta
    raise Stage1Error(
        "Não foi possível montar o palheiro com nenhuma fonte configurada:\n  - " + "\n  - ".join(errors)
    )


def _accept(batch: list[dict], docs: list[dict], backend) -> int:
    encodings = backend.encode_batch([a["text"] for a in batch], add_special_tokens=False)
    added = 0
    for art, enc in zip(batch, encodings):
        art["n_tokens"] = len(enc.ids)
        art["sent_starts"] = article_sentence_starts(art["text"])
        docs.append(art)
        added += art["n_tokens"]
    return added


# --------------------------------------------------------------------------- pool carregado


@dataclass
class HaystackPool:
    text: str
    sent_char: np.ndarray  # inícios globais de frase + sentinela len(text)
    doc_bounds: np.ndarray  # início de cada artigo no texto concatenado
    meta: dict
    text_hash: str = field(default="")

    @property
    def n_sentences(self) -> int:
        return len(self.sent_char) - 1

    def piece(self, i: int) -> str:
        return self.text[self.sent_char[i] : self.sent_char[i + 1]]


def load_pool(path: Path) -> HaystackPool:
    import pyarrow.parquet as pq

    if not path.exists():
        raise Stage1Error(f"Pool do palheiro não encontrado: {path}. Rode o modo 'prepare' antes.")
    table = pq.read_table(path, columns=["text", "sent_starts"])
    raw_meta = (table.schema.metadata or {}).get(b"lumi_ctx", b"{}")
    meta = json.loads(raw_meta.decode("utf-8"))
    texts = table.column("text").to_pylist()
    starts = table.column("sent_starts").to_pylist()
    pieces: list[str] = []
    sent_char: list[int] = []
    doc_bounds: list[int] = []
    offset = 0
    for i, (text, s) in enumerate(zip(texts, starts)):
        if i:
            pieces.append(DOC_SEPARATOR)
            offset += len(DOC_SEPARATOR)
        doc_bounds.append(offset)
        sent_char.extend(offset + int(x) for x in s)
        pieces.append(text)
        offset += len(text)
    full = "".join(pieces)
    sent_char.append(len(full))
    return HaystackPool(
        text=full,
        sent_char=np.asarray(sent_char, dtype=np.int64),
        doc_bounds=np.asarray(doc_bounds, dtype=np.int64),
        meta=meta,
        text_hash=sha256_text(full)[:16],
    )


def tokenizer_fingerprint(tok) -> str:
    backend = get_backend_tokenizer(tok)
    return sha256_text(backend.to_str())[:16]


def token_index(pool: HaystackPool, tok, cache_dir: Path | None = None) -> np.ndarray:
    """tok_cum[i] ≈ nº de tokens do pool antes da frase i (len = n_sentences + 1).

    Calculado tokenizando cada artigo com offsets; serve só para planejar o corte.
    O comprimento final de cada prompt é sempre medido na tokenização real.
    """
    key = f"{pool.text_hash}_{tokenizer_fingerprint(tok)}"
    cache = cache_dir / f"tokidx_{key}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        return np.load(cache)
    backend = get_backend_tokenizer(tok)
    sep_tokens = len(backend.encode(DOC_SEPARATOR, add_special_tokens=False).ids)
    bounds = list(pool.doc_bounds) + [len(pool.text) + len(DOC_SEPARATOR)]
    docs = [pool.text[bounds[i] : bounds[i + 1] - len(DOC_SEPARATOR)] for i in range(len(pool.doc_bounds))]
    tok_cum = np.zeros(pool.n_sentences + 1, dtype=np.int64)
    running = 0
    s_ptr = 0
    batch = 512
    for b0 in range(0, len(docs), batch):
        encs = backend.encode_batch(docs[b0 : b0 + batch], add_special_tokens=False)
        for d, enc in enumerate(encs):
            doc_start = bounds[b0 + d]
            doc_end = doc_start + len(docs[b0 + d])
            tok_starts = np.fromiter((o[0] for o in enc.offsets), dtype=np.int64, count=len(enc.ids))
            s_end = int(np.searchsorted(pool.sent_char[:-1], doc_end, side="left"))
            local = pool.sent_char[s_ptr:s_end] - doc_start
            tok_cum[s_ptr:s_end] = running + np.searchsorted(tok_starts, local, side="left")
            s_ptr = s_end
            running += len(enc.ids) + sep_tokens
    tok_cum[-1] = running - sep_tokens
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, tok_cum)
    return tok_cum
