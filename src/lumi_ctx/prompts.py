"""Construção dos prompts com comprimento exato (seções 3.3, 3.6 e 3.7).

Para cada (tarefa, L, profundidade, repetição):
1. Sorteia agulha(s) com um gerador próprio do prompt (independe do modelo).
2. O = tokens do template com o palheiro vazio + agulhas; B = L - O.
3. Sorteia um início de frase no pool e pega frases inteiras até somar ~B tokens.
4. Insere as agulhas em fronteiras de frase, na profundidade definida.
5. Renderiza com o chat template do modelo (enable_thinking=False) e mede.
   Se passar de L, remove frases do fim; se ficar abaixo de L·(1-2 %), acrescenta.
   As agulhas nunca são removidas. Se não houver como encaixar, sorteia outro início.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import Stage1Error, read_jsonl, sha256_text, stable_hash, write_json, write_jsonl
from .haystack import HaystackPool, load_pool, token_index, tokenizer_fingerprint
from .needles import (
    draw_needles,
    draw_value,
    expected_answers,
    needle_sentence,
    number_variants,
    prompt_rng,
    question_for,
    render_user_message,
)

log = logging.getLogger(__name__)

BUILDER_VERSION = 1
MAX_RESAMPLES = 25
MAX_FIT_ITERS = 10
TOKENIZER_FILES = ["*.json", "*.jinja", "*.model", "*.txt", "*.py", "*.tiktoken"]


# --------------------------------------------------------------------------- tokenizer


def tokenizer_dir(model: dict) -> str:
    """Baixa (ou acha no cache do HF) só os arquivos de tokenizer/config do modelo."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model["hf_id"], revision=model["revision"], allow_patterns=TOKENIZER_FILES)


def load_tokenizer(path: str):
    """Carrega o tokenizer; se a classe não for reconhecida, monta um fast tokenizer genérico."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tok._lumi_fallback = None
    except Exception as exc:  # noqa: BLE001
        from transformers import PreTrainedTokenizerFast

        log.warning("AutoTokenizer falhou (%s); usando PreTrainedTokenizerFast + chat_template.jinja", exc)
        cfg_file = Path(path) / "tokenizer_config.json"
        import json

        tcfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        tok = PreTrainedTokenizerFast(
            tokenizer_file=str(Path(path) / "tokenizer.json"),
            bos_token=tcfg.get("bos_token"),
            eos_token=tcfg.get("eos_token"),
            unk_token=tcfg.get("unk_token"),
        )
        tok._lumi_fallback = f"AutoTokenizer falhou: {type(exc).__name__}: {exc}"
    if not getattr(tok, "chat_template", None):
        jinja = Path(path) / "chat_template.jinja"
        if not jinja.exists():
            raise Stage1Error(f"Modelo sem chat template em {path}")
        tok.chat_template = jinja.read_text(encoding="utf-8")
    return tok


def render_ids(tok, user_text: str, enable_thinking: bool = False) -> list[int]:
    """Renderiza a mensagem única de usuário com o chat template e devolve os IDs."""
    msgs = [{"role": "user", "content": user_text}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=True)
    if hasattr(out, "keys"):  # transformers 5 devolve BatchEncoding
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


def render_text(tok, user_text: str, enable_thinking: bool = False) -> str:
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=False)


def check_template(tok) -> dict:
    """Confere que enable_thinking=False fecha o bloco de raciocínio no prompt."""
    off = render_text(tok, "Teste", enable_thinking=False)
    on = render_text(tok, "Teste", enable_thinking=True)
    last_open, last_close = off.rfind("<think>"), off.rfind("</think>")
    closed = last_open == -1 or last_close > last_open
    return {
        "ok": bool(closed and off != on),
        "thinking_closed_in_prompt": closed,
        "differs_from_thinking_on": off != on,
        "generation_suffix_off": off[-60:],
        "generation_suffix_on": on[-60:],
        "template_sha256": sha256_text(tok.chat_template or "")[:16],
    }


def think_token_ids(tok) -> list[int]:
    ids = []
    for t in ("<think>", "</think>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None):
            ids.append(i)
    return ids


def stop_token_ids(tok) -> list[int]:
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in ("<|im_end|>", "</s>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None):
            ids.add(i)
    return sorted(i for i in ids if i is not None)


# --------------------------------------------------------------------------- especificações


def depth_pm(depth: float | None) -> int:
    return -1 if depth is None else int(round(depth * 1000))


def make_uid(task: str, length: int, depth: float | None, rep: int) -> str:
    if task == "C0":
        return f"C0-r{rep:02d}"
    return f"{task}-L{length:06d}-d{depth_pm(depth):04d}-r{rep:02d}"


def enumerate_specs(cfg: dict) -> list[dict]:
    exp = cfg["experiment"]
    specs: list[dict] = []
    tasks = exp["tasks"]
    if "C0" in tasks:
        for r in range(tasks["C0"]["reps"]):
            specs.append({"task": "C0", "length": 0, "depth": None, "rep": r})
    for task in ("S", "MK", "MV"):
        if task not in tasks:
            continue
        for length in exp["lengths"]:
            for depth in tasks[task]["depths"]:
                for r in range(tasks[task]["reps"]):
                    specs.append({"task": task, "length": int(length), "depth": float(depth), "rep": r})
    for s in specs:
        s["uid"] = make_uid(s["task"], s["length"], s["depth"], s["rep"])
    if len({s["uid"] for s in specs}) != len(specs):
        raise Stage1Error("UIDs de prompt repetidos: confira profundidades/repetições na config")
    return specs


# --------------------------------------------------------------------------- construtor


@dataclass
class PromptBuilder:
    tok: object
    pool: HaystackPool
    tok_cum: np.ndarray
    cfg: dict

    def __post_init__(self):
        exp = self.cfg["experiment"]
        self.seed = int(exp["seed"])
        self.tol = float(exp["length_tolerance"])
        self.gap = float(exp.get("min_needle_gap", 0.02))
        tasks = exp["tasks"]
        self.n_distractors = int(tasks.get("MK", {}).get("n_distractors", 3))
        self.n_values = int(tasks.get("MV", {}).get("n_values", 4))
        self.enable_thinking = bool(self.cfg["generation"].get("enable_thinking", False))

    # -- helpers
    def _render(self, haystack: str, question: str) -> list[int]:
        return render_ids(self.tok, render_user_message(haystack, question), self.enable_thinking)

    def _boundary_offsets(self, s: int, e: int) -> np.ndarray:
        return self.tok_cum[s : e + 1] - self.tok_cum[s]

    def _place(self, offsets: np.ndarray, depth: float, n_needles: int, place_seed: int) -> tuple[list[int], bool]:
        """Fronteiras (índices 0..k) para cada agulha. A primeira é o alvo."""
        span = float(offsets[-1]) if len(offsets) else 0.0
        k = len(offsets) - 1

        def nearest(target_tok: float) -> int:
            j = int(np.searchsorted(offsets, target_tok))
            if j > k:
                return k
            if j > 0 and abs(offsets[j - 1] - target_tok) <= abs(offsets[j] - target_tok):
                return j - 1
            return j

        chosen = [nearest(depth * span)]
        rng = np.random.default_rng(place_seed)
        relaxed = False
        gap = self.gap * span
        for _ in range(n_needles - 1):
            for attempt in range(400):
                if attempt == 200:
                    gap, relaxed = gap / 2, True
                j = nearest(float(rng.random()) * span)
                if all(j != c and abs(offsets[j] - offsets[c]) >= gap for c in chosen):
                    chosen.append(j)
                    break
            else:
                raise Stage1Error("Não foi possível posicionar as agulhas (palheiro curto demais?)")
        return chosen, relaxed

    def _assemble(self, s: int, e: int, positions: list[int], sentences: list[str]) -> str:
        at: dict[int, str] = {j: sent for j, sent in zip(positions, sentences)}
        parts: list[str] = []
        for i in range(e - s):
            if i in at:
                parts.append(at[i] + " ")
            parts.append(self.pool.piece(s + i))
        if (e - s) in at:
            if parts and not parts[-1][-1:].isspace():
                parts.append(" ")
            parts.append(at[e - s])
        return "".join(parts).strip()

    # -- principal
    def build(self, task: str, length: int, depth: float | None, rep: int) -> tuple[dict, list[int]]:
        rng = prompt_rng(self.seed, task, length, depth, rep)
        draw = draw_needles(task, rng, self.n_distractors, self.n_values)
        city, needles = draw["city"], draw["needles"]
        question = question_for(task, city)
        meta = {
            "uid": make_uid(task, length, depth, rep),
            "task": task,
            "length": int(length),
            "depth": depth,
            "rep": int(rep),
            "city": city,
            "answers": None,
            "distractors": [n["value"] for n in needles if n["role"] == "distractor"],
            "needles": None,
            "value_redraws": 0,
            "resamples": 0,
            "gap_relaxed": False,
        }
        if task == "C0":
            ids = self._render(needle_sentence(city, needles[0]["value"]), question)
            meta.update(answers=expected_answers(task, needles), needles=needles, n_tokens=len(ids))
            meta.update(len_dev=None, within_tol=True, hay_start_sentence=None, hay_sentences=0)
            return meta, ids

        tol_low = int(np.ceil(length * (1 - self.tol)))
        tc = self.tok_cum
        total = int(tc[-1])
        used_values = {n["value"] for n in needles}

        for attempt in range(MAX_RESAMPLES):
            sentences = [needle_sentence(n["city"], n["value"]) for n in needles]
            overhead = len(self._render(" ".join(sentences), question))
            budget = length - overhead
            if budget <= 0:
                raise Stage1Error(f"L={length} é menor que o overhead do template ({overhead} tokens)")
            max_start_tok = total - int(budget * 1.1) - 256
            n_valid = int(np.searchsorted(tc, max_start_tok, side="right"))
            if n_valid <= 0:
                raise Stage1Error(
                    f"Pool pequeno demais para L={length}: {total} tokens. Aumente haystack.target_tokens."
                )
            s = int(rng.integers(0, n_valid))
            place_seed = int(rng.integers(0, 2**62))
            e = int(np.searchsorted(tc, tc[s] + budget, side="right")) - 1

            for _ in range(MAX_FIT_ITERS):
                if e <= s:
                    break
                offsets = self._boundary_offsets(s, e)
                try:
                    positions, relaxed = self._place(offsets, depth, len(needles), place_seed)
                except Stage1Error:
                    break  # palheiro curto demais para o espaçamento: sorteia outro início
                text = self._assemble(s, e, positions, sentences)
                # os números das agulhas não podem aparecer por acaso no palheiro
                hay_only = text
                for sent in sentences:
                    hay_only = hay_only.replace(sent, " ")
                clashes = [n for n in needles if any(v in hay_only for v in number_variants(n["value"]))]
                if clashes:
                    for n in clashes:
                        used_values.discard(n["value"])
                        n["value"] = draw_value(rng, used_values)
                        meta["value_redraws"] += 1
                    sentences = [needle_sentence(n["city"], n["value"]) for n in needles]
                    continue
                ids = self._render(text, question)
                n_tok = len(ids)
                if tol_low <= n_tok <= length:
                    span = float(offsets[-1]) or 1.0
                    for n, j in zip(needles, positions):
                        n["boundary"] = int(j)
                        n["depth_actual"] = round(float(offsets[j]) / span, 4)
                    for sent in sentences:
                        if text.count(sent) != 1:
                            raise Stage1Error(f"Agulha ausente/duplicada no prompt {meta['uid']}")
                    meta.update(
                        answers=expected_answers(task, needles),
                        needles=needles,
                        n_tokens=n_tok,
                        len_dev=round((n_tok - length) / length, 5),
                        within_tol=True,
                        resamples=attempt,
                        gap_relaxed=relaxed,
                        hay_start_sentence=s,
                        hay_sentences=e - s,
                        hay_tokens_est=int(offsets[-1]),
                        target_depth_actual=needles[0]["depth_actual"],
                    )
                    return meta, ids
                if n_tok > length:
                    excess = n_tok - length
                    k = int(np.searchsorted(tc[s : e + 1], tc[e] - excess, side="right")) - 1
                    e = min(e - 1, s + max(k, 0))
                else:
                    deficit = length - n_tok
                    new_e = int(np.searchsorted(tc, tc[e] + deficit, side="right")) - 1
                    if new_e <= e:
                        break  # a próxima frase não cabe: sorteia outro início
                    e = min(new_e, self.pool.n_sentences)
        raise Stage1Error(f"Não foi possível montar {meta['uid']} dentro da tolerância após {MAX_RESAMPLES} sorteios")


# --------------------------------------------------------------------------- construção em paralelo

_W: dict = {}


def _worker_init(tok_path: str, pool_path: str, tok_cum_path: str, cfg: dict) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = load_tokenizer(tok_path)
    pool = load_pool(Path(pool_path))
    tok_cum = np.load(tok_cum_path)
    _W["builder"] = PromptBuilder(tok, pool, tok_cum, cfg)


def _worker_build(spec: dict) -> tuple[dict, np.ndarray]:
    meta, ids = _W["builder"].build(spec["task"], spec["length"], spec["depth"], spec["rep"])
    return meta, np.asarray(ids, dtype=np.int32)


def prompt_set_key(cfg: dict, pool: HaystackPool, tok) -> str:
    analysis_only = ("bootstrap", "threshold", "c0_gate", "think_gate_max_length")
    exp = {k: v for k, v in cfg["experiment"].items() if k not in analysis_only}
    return stable_hash(
        {
            "builder": BUILDER_VERSION,
            "experiment": exp,
            "enable_thinking": cfg["generation"].get("enable_thinking", False),
            "pool": pool.text_hash,
            "tokenizer": tokenizer_fingerprint(tok),
            "template": sha256_text(tok.chat_template or ""),
        }
    )


@dataclass
class PromptSet:
    key: str
    directory: Path
    index: list[dict]

    @property
    def tokens_path(self) -> Path:
        return self.directory / "tokens.bin"

    def ids(self, row: dict, memmap: np.ndarray | None = None) -> list[int]:
        arr = memmap if memmap is not None else np.memmap(self.tokens_path, dtype=np.int32, mode="r")
        return arr[row["offset"] : row["offset"] + row["n_tokens"]].tolist()


def load_prompt_set(directory: Path) -> PromptSet:
    index = list(read_jsonl(directory / "index.jsonl"))
    return PromptSet(key=directory.name, directory=directory, index=index)


def build_prompt_set(
    cfg: dict, tok, tok_path: str, pool: HaystackPool, pool_path: Path, out_root: Path, n_workers: int
) -> PromptSet:
    """Monta (ou reaproveita) o conjunto de prompts de um tokenizer/template."""
    key = prompt_set_key(cfg, pool, tok)
    directory = out_root / key
    if (directory / "DONE").exists():
        log.info("Prompts já construídos em %s (reaproveitando)", directory)
        return load_prompt_set(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache_dir = pool_path.parent
    tok_cum = token_index(pool, tok, cache_dir)
    tok_cum_path = directory / "tok_cum.npy"
    np.save(tok_cum_path, tok_cum)

    specs = enumerate_specs(cfg)
    order = sorted(range(len(specs)), key=lambda i: -specs[i]["length"])  # longos primeiro
    t0 = time.time()
    index: list[dict | None] = [None] * len(specs)
    offset = 0
    log.info("Construindo %d prompts com %d processos...", len(specs), n_workers)
    initargs = (tok_path, str(pool_path), str(tok_cum_path), cfg)
    with open(directory / "tokens.bin.tmp", "wb") as fh:
        if n_workers <= 1:
            _worker_init(*initargs)
            results = (_worker_build(specs[i]) for i in order)
            pool_ctx = None
        else:
            pool_ctx = mp.get_context("spawn").Pool(n_workers, initializer=_worker_init, initargs=initargs)
            results = pool_ctx.imap(_worker_build, [specs[i] for i in order], chunksize=1)
        try:
            last_log = time.time()
            for done, (i, (meta, arr)) in enumerate(zip(order, results), start=1):
                meta["offset"] = offset
                fh.write(arr.tobytes())
                offset += len(arr)
                index[i] = meta
                if time.time() - last_log > 30 or done == len(specs):
                    log.info("  %d/%d prompts (%.0fs)", done, len(specs), time.time() - t0)
                    last_log = time.time()
        finally:
            if pool_ctx is not None:
                pool_ctx.close()
                pool_ctx.join()
    os.replace(directory / "tokens.bin.tmp", directory / "tokens.bin")
    write_jsonl(directory / "index.jsonl", index)
    stats = summarize_prompt_set(index, cfg)
    stats.update(key=key, seconds=round(time.time() - t0, 1), total_tokens=int(offset))
    write_json(directory / "stats.json", stats)
    (directory / "DONE").write_text("ok\n")
    log.info("Prompts prontos: %d (%.1fM tokens) em %.0fs", len(index), offset / 1e6, time.time() - t0)
    return load_prompt_set(directory)


def summarize_prompt_set(index: list[dict], cfg: dict) -> dict:
    tol = cfg["experiment"]["length_tolerance"]
    devs = [r["len_dev"] for r in index if r["len_dev"] is not None]
    over = [r["uid"] for r in index if r["task"] != "C0" and r["n_tokens"] > r["length"]]
    out_tol = [r["uid"] for r in index if r["len_dev"] is not None and abs(r["len_dev"]) > tol]
    return {
        "n_prompts": len(index),
        "max_abs_len_dev": max((abs(d) for d in devs), default=0.0),
        "mean_len_dev": float(np.mean(devs)) if devs else 0.0,
        "n_over_target": len(over),
        "n_outside_tol": len(out_tol),
        "outside_tol_uids": out_tol[:50],
        "n_resampled": sum(1 for r in index if r.get("resamples")),
        "n_value_redraws": sum(r.get("value_redraws", 0) for r in index),
        "n_gap_relaxed": sum(1 for r in index if r.get("gap_relaxed")),
        "c0_tokens_mean": float(np.mean([r["n_tokens"] for r in index if r["task"] == "C0"] or [0])),
    }
