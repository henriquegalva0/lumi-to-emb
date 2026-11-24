"""Mineração de negativos difíceis com um teacher forte (seção 2.4 do plano). Ambiente do vLLM.

Para cada família de tarefa (retrieval curto, retrieval longo, sts, bitext):
1. o teacher (Qwen3-Embedding, runner de pooling do vLLM) embute queries (com instrução)
   e documentos (positivos + negativos já existentes + corpus extra opcional);
2. pos_score = cos(q, positivo); nos sintéticos, descarta se pos_score ≤ 0,7 ou se o
   positivo não aparece no top-k do teacher (filtros do Qwen3-Embedding e proposta do plano);
3. busca os candidatos mais próximos (produto interno na GPU), sem o próprio positivo nem
   quase-duplicatas dele, e guarda:
   - negatives:       TopK-PercPos — só candidatos com score < 0,95 × pos_score [10][11]
   - naive_negatives: top-k sem filtro (condição da ablação A3)
"""

from __future__ import annotations

import json
import logging
import math
import re

from ..utils import EmbedError, data_dir, iter_jsonl, write_json
from .schema import norm_text

log = logging.getLogger("lumi_embed.mine")


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", norm_text(text).lower())


def word_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


def near_duplicate(a: str, b: str, threshold: float = 0.8) -> bool:
    """Igual ou quase igual (Jaccard de 3-gramas de palavras, ignorando pontuação e caixa)."""
    if _words(a) == _words(b):
        return True
    ga, gb = word_ngrams(a), word_ngrams(b)
    if not ga or not gb:
        return False
    return len(ga & gb) / len(ga | gb) >= threshold


class NearDup:
    """`near_duplicate` com a assinatura de cada texto calculada uma vez só.

    Mesma regra (palavras iguais, ou Jaccard de 3-gramas de palavras ≥ limiar), mas os 3-gramas
    viram um vetor ordenado de hashes de 64 bits guardado em cache. Sem isso, cada query refazia os
    conjuntos do positivo e de até 64 candidatos: em documentos longos, horas só nessa checagem.
    """

    def __init__(self, threshold: float = 0.8, n: int = 3):
        import numpy as np

        self.np, self.t, self.n, self._cache = np, threshold, n, {}

    def sig(self, text: str):
        s = self._cache.get(text)
        if s is None:
            w = _words(text)
            g = self.np.unique(self.np.fromiter(
                (hash(tuple(w[i : i + self.n])) for i in range(max(0, len(w) - self.n + 1))), dtype=self.np.int64))
            s = (hash(tuple(w)), len(w), g)
            self._cache[text] = s
        return s

    def __call__(self, a: str, b: str) -> bool:
        ha, na, ga = self.sig(a)
        hb, nb, gb = self.sig(b)
        if ha == hb and na == nb:
            return True
        la, lb = len(ga), len(gb)
        if not la or not lb or min(la, lb) / max(la, lb) < self.t:  # Jaccard ≤ min/max
            return False
        inter = len(self.np.intersect1d(ga, gb, assume_unique=True))
        return inter / (la + lb - inter) >= self.t


def select_negatives(cands: list[tuple[str, float]], positive: str, pos_score: float, k: int, perc: float,
                     dup=None) -> tuple[list[dict], list[dict]]:
    """(TopK-PercPos, top-k ingênuo) a partir de candidatos ordenados por score decrescente.

    Para assim que as duas listas têm k itens e só checa quase-duplicata nos candidatos que entrariam.
    """
    is_dup = dup or near_duplicate
    percpos: list[dict] = []
    naive: list[dict] = []
    for t, s in cands:
        if len(percpos) >= k and len(naive) >= k:
            break
        want_naive = len(naive) < k
        want_perc = len(percpos) < k and s < perc * pos_score
        if not (want_naive or want_perc) or is_dup(t, positive):
            continue
        if want_naive:
            naive.append({"text": t, "score": round(s, 4)})
        if want_perc:
            percpos.append({"text": t, "score": round(s, 4)})
    return percpos, naive


def family(ex: dict) -> str:
    if ex["task"] == "retrieval":
        return "retrieval_long" if ex.get("source", {}).get("long") else "retrieval"
    return ex["task"]


class Teacher:
    """Um único motor de pooling por processo (o vLLM não libera a GPU de forma confiável)."""

    def __init__(self, mcfg: dict):
        from vllm import LLM

        self.cfg = mcfg
        self.max_len = int(mcfg["max_len"])
        self.llm = LLM(model=mcfg["model"], revision=mcfg.get("revision"), runner="pooling",
                       max_model_len=int(mcfg["max_len_long"]),
                       gpu_memory_utilization=float(mcfg.get("gpu_memory_utilization", 0.5)), seed=0)

    def embed(self, texts: list[str]):
        import numpy as np
        import torch

        outs = self.llm.embed(texts, use_tqdm=False,
                              tokenization_kwargs={"truncation": True, "max_length": self.max_len})
        # O vLLM devolve listas Python; numpy converte bem mais rápido que torch.tensor(lista de listas)
        e = torch.from_numpy(np.asarray([o.outputs.embedding for o in outs], dtype=np.float32))
        e = torch.nn.functional.normalize(e, dim=-1)
        return e.half().cuda() if torch.cuda.is_available() else e

    def query_text(self, ex: dict) -> str:
        inst = ex.get("instruction") or self.cfg["default_instruction"]
        return self.cfg["query_template"].format(instruction=inst, query=ex["query"])


class HFTeacher:
    """O mesmo teacher pelo transformers, sem o vLLM, para a família de documentos longos.

    No vLLM 0.29, o pooling travou duas vezes no primeiro bloco de documentos de até 8k tokens, sem erro:
    8 h numa B200 inteira e 2,5 h na classe menor, com as queries da mesma família codificadas em segundos.
    Aqui: padding à esquerda e o estado do último token, normalizado (como no card do Qwen3-Embedding);
    lotes por comprimento, para pouco padding. Queries e documentos da família passam pelo mesmo caminho.
    """

    def __init__(self, mcfg: dict, max_len: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.cfg, self.max_len = mcfg, int(max_len)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rev = mcfg.get("revision")
        self.tok = AutoTokenizer.from_pretrained(mcfg["model"], revision=rev, padding_side="left")
        if self.tok.pad_token is None:  # o do Qwen3-Embedding tem (<|endoftext|>); outros podem não ter
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(mcfg["model"], revision=rev, dtype=dtype).to(self.device).eval()
        self.bs = int(mcfg.get("hf_batch_long", 8))

    def embed(self, texts: list[str]):
        import torch

        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out: list = [None] * len(texts)
        with torch.no_grad():
            for b in range(0, len(order), self.bs):
                idx = order[b : b + self.bs]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_len,
                               return_tensors="pt")
                h = self.model(input_ids=enc["input_ids"].to(self.device),
                               attention_mask=enc["attention_mask"].to(self.device)).last_hidden_state[:, -1]
                e = torch.nn.functional.normalize(h.float(), dim=-1)
                for j, i in enumerate(idx):
                    out[i] = e[j]
        e = torch.stack(out)
        return e.half() if self.device == "cuda" else e

    query_text = Teacher.query_text


def _embed_all(teacher: Teacher, texts: list[str], bs: int, what: str):
    """Embeddings em lotes, com progresso no log (a família longa leva horas)."""
    import time

    import torch

    out, t0 = [], time.time()
    for i in range(0, len(texts), bs):
        out.append(teacher.embed(texts[i : i + bs]))
        done = min(i + bs, len(texts))
        el = time.time() - t0
        log.info("    %s: %d/%d (%.0f min; ~%.0f min restantes)", what, done, len(texts), el / 60,
                 el / done * (len(texts) - done) / 60)
    return torch.cat(out)


def mine_group(teacher: Teacher, rows: list[dict], extra_docs: list[str], mcfg: dict) -> tuple[list[dict], dict]:
    import torch

    docs, index = [], {}

    def add(t: str) -> int:
        t = norm_text(t)
        if t not in index:
            index[t] = len(docs)
            docs.append(t)
        return index[t]

    pos_idx = [add(r["positive"]) for r in rows]
    for r in rows:
        for n in r.get("negatives") or []:
            add(n["text"])
    for t in extra_docs:
        add(t)
    symmetric = rows[0]["task"] in ("sts", "bitext")
    q_inputs = [teacher.query_text(r) for r in rows]
    d_inputs = docs
    if symmetric:  # nas tarefas simétricas o teacher também recebe a instrução nos documentos
        inst = rows[0].get("instruction") or mcfg["default_instruction"]
        d_inputs = [mcfg["query_template"].format(instruction=inst, query=d) for d in docs]
    long_docs = teacher.max_len > int(mcfg["max_len"])
    bs = int(mcfg.get("embed_chunk_long" if long_docs else "embed_chunk", 2048 if long_docs else 16384))
    Q = _embed_all(teacher, q_inputs, bs, "queries")
    D = _embed_all(teacher, d_inputs, bs, "documentos")
    k_store, top = int(mcfg["store_negatives"]), int(mcfg["search_top"])
    perc, thr, rank_k = float(mcfg["percpos"]), float(mcfg["synthetic_min_cos"]), int(mcfg["positive_rank_check"])
    if long_docs:  # o cosseno de uma pergunta sobre um trecho com o documento inteiro é naturalmente menor
        thr = float(mcfg.get("synthetic_min_cos_long", thr))
    kept, stats = [], {"in": len(rows), "drop_low_cos": 0, "drop_rank": 0, "few_negatives": 0}
    dup = NearDup()  # assinaturas em cache por família (cada documento é processado uma vez)
    pidx = torch.tensor(pos_idx, device=Q.device)
    for s in range(0, len(rows), 4096):
        q = Q[s : s + 4096]
        scores = (q @ D.T).float()
        pos_scores = scores[torch.arange(len(q)), pidx[s : s + 4096]]
        rank = (scores > pos_scores.unsqueeze(1)).sum(dim=1) + 1
        vals, ids = scores.topk(min(top, scores.shape[1]), dim=1)
        vals, ids, pos_scores, rank = vals.cpu(), ids.cpu(), pos_scores.cpu(), rank.cpu()
        for j in range(len(q)):
            r = rows[s + j]
            ps = float(pos_scores[j])
            synthetic = "gen" in (r.get("source") or {})
            if synthetic and ps <= thr:
                stats["drop_low_cos"] += 1
                continue
            if synthetic and int(rank[j]) > rank_k:
                stats["drop_rank"] += 1
                continue
            cands = [(docs[int(i)], float(v)) for v, i in zip(vals[j], ids[j]) if int(i) != pos_idx[s + j]]
            percpos, naive = select_negatives(cands, r["positive"], ps, k_store, perc, dup=dup)
            prior = [n for n in (r.get("negatives") or []) if n.get("kind") in ("nli", "contraste")]
            r = {**r, "pos_score": round(ps, 4), "negatives": prior + percpos, "naive_negatives": prior + naive,
                 "teacher": {"model": mcfg["model"], "rank_pos": int(rank[j])}}
            if len(r["negatives"]) < int(mcfg.get("min_negatives", 1)):
                stats["few_negatives"] += 1
            kept.append(r)
    stats["out"] = len(kept)
    stats["pos_score_mean"] = round(sum(r["pos_score"] for r in kept) / max(1, len(kept)), 4)
    return kept, stats


def run_mine(cfg: dict, inputs: list[str] | None = None) -> dict:
    mcfg = cfg["mining"]
    files = inputs or mcfg["inputs"]
    groups: dict[str, list[dict]] = {}
    for f in files:
        path = data_dir(f)
        if not path.exists():
            log.warning("Entrada ausente (pulando): %s", path)
            continue
        for ex in iter_jsonl(path):
            groups.setdefault(family(ex), []).append(ex)
    if not groups:
        raise EmbedError("Nada para minerar: rode as fontes, a síntese e a descontaminação antes.")
    all_stats = {}
    teacher = None
    deferred = set(mcfg.get("defer_families") or [])
    for fam, rows in sorted(groups.items()):
        if fam in deferred:
            log.warning("%s adiada (mining.defer_families): %d exemplos não minerados agora", fam, len(rows))
            all_stats[fam] = {"in": len(rows), "deferred": True}
            continue
        out = data_dir("mined", f"{fam}.jsonl")
        fam_stats = data_dir("mined", f"stats_{fam}.json")
        if out.exists() and fam_stats.exists():  # família já minerada (job anterior): não refaz
            all_stats[fam] = json.loads(fam_stats.read_text())
            log.info("%s já minerada (%d exemplos); pulando", fam, all_stats[fam].get("out", -1))
            continue
        if fam == "retrieval_long" and mcfg.get("long_backend", "vllm") == "hf":
            fam_teacher = HFTeacher(mcfg, int(mcfg["max_len_long"]))
            backend = "transformers"
        else:
            if teacher is None:
                teacher = Teacher(mcfg)
            teacher.max_len = int(mcfg["max_len_long"] if fam == "retrieval_long" else mcfg["max_len"])
            fam_teacher, backend = teacher, "vllm"
        log.info("Minerando %s: %d exemplos (max_len %d, %s)", fam, len(rows), fam_teacher.max_len, backend)
        kept, stats = mine_group(fam_teacher, rows, [], mcfg)
        stats["backend"] = backend
        del fam_teacher
        _free_gpu()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(out)
        write_json(fam_stats, stats)  # gravado por último: marca a família como concluída
        all_stats[fam] = stats
        log.info("  %s", stats)
    write_json(data_dir("mined", "stats.json"), all_stats)
    return all_stats


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def expected_pool_gb(n_docs: int, dim: int) -> float:
    return n_docs * dim * 2 / 1e9 + math.ceil(n_docs / 1e6)
