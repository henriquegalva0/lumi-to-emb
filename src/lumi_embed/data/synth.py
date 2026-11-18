"""Geração sintética de dados em PT-BR nativo (seção 2.3 do plano). Roda no ambiente do vLLM.

Para cada documento-semente sorteia-se uma configuração (persona, tipo de consulta,
dificuldade, registro e tamanho) e o gerador escreve a consulta em JSON; o positivo é o
próprio documento. Três geradores abertos (Apache 2.0) recebem sementes diferentes e as
saídas são misturadas depois (seção 2.1: misturar geradores é melhor que um só).
Também gera:
- STS: paráfrase + "contraste" (frase parecida com sentido diferente → negativo difícil);
- consultas sobre um trecho de documento longo (positivo = documento inteiro, S-long).

Filtros nesta etapa: JSON válido, tamanho, cópia literal (8-gramas do documento), idioma.
O filtro por cosseno do teacher (> 0,7) vem depois, na mineração.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re

from lumi_ctx.haystack import sentence_starts

from ..utils import EmbedError, data_dir, iter_jsonl
from .schema import make_example, norm_text

log = logging.getLogger("lumi_embed.synth")

QUERY_TYPES = {
    "keywords": ("palavras-chave de busca, sem formar uma frase completa", 6),
    "acquire_knowledge": ("uma pergunta de quem quer aprender algo que o documento explica", 20),
    "summary": ("um pedido de resumo sobre um tema tratado no documento", 18),
    "yes_or_no": ("uma pergunta de sim ou não cuja resposta está no documento", 16),
    "background": ("um pedido de contexto ou histórico sobre o assunto do documento", 20),
}
DIFFICULTY = {"high_school": "ensino médio", "university": "graduação", "phd": "doutorado"}
REGISTER = {"formal": "formal", "coloquial": "coloquial, como alguém digitaria numa busca"}

PROMPT_QUERY = (
    "Você é {persona}. Leia o documento e escreva UMA consulta do tipo \"{tipo}\", em português "
    "brasileiro {registro}, no nível \"{dificuldade}\", com no máximo {n} palavras, que o documento "
    "responda. Não copie trechos literais do documento. Responda apenas em JSON: {{\"query\": \"...\"}}\n\n"
    "DOCUMENTO:\n{documento}"
)
PROMPT_LONG = (
    "Você é {persona}. O TRECHO abaixo faz parte de um documento longo. Escreva UMA pergunta em português "
    "brasileiro, com no máximo {n} palavras, que só este trecho responda (não uma pergunta genérica sobre o "
    "documento todo). Não copie trechos literais. Responda apenas em JSON: {{\"query\": \"...\"}}\n\n"
    "TRECHO:\n{documento}"
)
PROMPT_STS = (
    "Reescreva a frase abaixo em português brasileiro com outras palavras, mantendo exatamente o mesmo "
    "sentido. Depois escreva uma segunda frase parecida na forma, mas com sentido diferente (mude um fato, "
    "um número, uma negação ou a relação entre as partes). Responda apenas em JSON: "
    "{{\"parafrase\": \"...\", \"contraste\": \"...\"}}\n\nFRASE: {frase}"
)

PT_HINTS = re.compile(r"\b(de|que|o|a|os|as|do|da|em|um|uma|para|com|não|por|qual|como|quais|é)\b", re.I)


def stable_bucket(key: str, n: int) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % n


def extract_json(text: str, key: str) -> dict | None:
    """Último objeto JSON com a chave pedida (tolera texto de raciocínio antes da resposta)."""
    text = text.split("assistantfinal")[-1]
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, flags=re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get(key), str) and obj[key].strip():
            return obj
    return None


def copies_document(query: str, doc: str, n: int = 8) -> bool:
    q = norm_text(query).lower().split()
    if len(q) < n:
        return norm_text(query).lower() in norm_text(doc).lower() and len(q) > 3
    d = " " + " ".join(norm_text(doc).lower().split()) + " "
    return any(" " + " ".join(q[i : i + n]) + " " in d for i in range(len(q) - n + 1))


def looks_portuguese(text: str) -> bool:
    words = text.split()
    return len(words) <= 4 or len(PT_HINTS.findall(text)) / max(1, len(words)) >= 0.08


def draw_config(rng: random.Random, cfg: dict) -> dict:
    t = rng.choices(list(QUERY_TYPES), weights=[cfg["type_weights"][k] for k in QUERY_TYPES])[0]
    return {"persona": rng.choice(cfg["personas"]), "tipo": t, "dificuldade": rng.choice(list(DIFFICULTY)),
            "registro": rng.choice(list(REGISTER)), "n": QUERY_TYPES[t][1]}


def pick_sentence(text: str, rng: random.Random) -> str | None:
    starts = sentence_starts(text)
    sents = [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])]
    good = [s for s in sents if 10 <= len(s.split()) <= 40]
    return rng.choice(good) if good else None


def build_requests(cfg: dict, gen_key: str, gen_index: int, n_gen: int, seed: int) -> list[dict]:
    """Pedidos (mensagens + metadados) deste gerador; cada gerador recebe sementes diferentes."""
    scfg = cfg["synth"]
    gcfg = scfg["generators"][gen_key]
    rng = random.Random(seed * 100 + gen_index)
    reqs = []
    seeds = [s for s in iter_jsonl(data_dir("seeds", "seeds.jsonl")) if stable_bucket(s["id"], n_gen) == gen_index]
    rng.shuffle(seeds)
    n_ret, n_sts = int(gcfg["n_retrieval"]), int(gcfg["n_sts"])
    for s in seeds[:n_ret]:
        c = draw_config(rng, scfg)
        prompt = PROMPT_QUERY.format(persona=c["persona"], tipo=QUERY_TYPES[c["tipo"]][0],
                                     registro=REGISTER[c["registro"]], dificuldade=DIFFICULTY[c["dificuldade"]],
                                     n=c["n"], documento=s["text"])
        reqs.append({"kind": "retrieval", "seed": s, "conf": c, "prompt": prompt})
    for s in seeds[n_ret : n_ret + n_sts]:
        frase = pick_sentence(s["text"], rng)
        if frase:
            reqs.append({"kind": "sts", "seed": s, "conf": {"frase": frase}, "prompt": PROMPT_STS.format(frase=frase)})
    longs = [s for s in iter_jsonl(data_dir("seeds", "long_seeds.jsonl")) if stable_bucket(s["id"], n_gen) == gen_index]
    for s in longs[: int(gcfg.get("n_long", 0))]:
        words = s["text"].split()
        w = int(scfg.get("long_window_words", 300))
        start = rng.randrange(max(1, len(words) - w))
        c = {**draw_config(rng, scfg), "n": 20, "window_start_word": start,
             "window_rel_pos": round(start / max(1, len(words)), 3)}
        prompt = PROMPT_LONG.format(persona=c["persona"], n=c["n"], documento=" ".join(words[start : start + w]))
        reqs.append({"kind": "long", "seed": s, "conf": c, "prompt": prompt})
    return reqs


def to_example(req: dict, text: str, cfg: dict, gen_key: str) -> dict | None:
    gcfg = cfg["synth"]["generators"][gen_key]
    src = {"gen": gcfg["model"], "gen_key": gen_key, "seed_id": req["seed"]["id"], "corpus": req["seed"]["source"],
           **{k: v for k, v in req["conf"].items() if k != "frase"}}
    inst = cfg["instructions"]
    if req["kind"] == "sts":
        obj = extract_json(text, "parafrase")
        if not obj:
            return None
        para, frase = norm_text(obj["parafrase"]), req["conf"]["frase"]
        if para.lower() == frase.lower() or not looks_portuguese(para):
            return None
        negs = []
        if isinstance(obj.get("contraste"), str) and obj["contraste"].strip():
            negs = [{"text": norm_text(obj["contraste"]), "score": None, "kind": "contraste"}]
        return make_example("sts", "pt-BR", inst["sts"], frase, para, src, negatives=negs)
    obj = extract_json(text, "query")
    if not obj:
        return None
    q = norm_text(obj["query"])
    n_max = req["conf"].get("n", 20) * 2
    if not (2 <= len(q.split()) <= n_max):
        return None
    doc = req["seed"]["text"]
    window = doc if req["kind"] != "long" else " ".join(
        doc.split()[req["conf"]["window_start_word"] : req["conf"]["window_start_word"] + 300])
    if copies_document(q, window) or (req["conf"].get("tipo") != "keywords" and not looks_portuguese(q)):
        return None
    if req["kind"] == "long":
        return make_example("retrieval", "pt-BR", inst["qa"], q, doc, {**src, "long": True})
    return make_example("retrieval", "pt-BR", inst["qa"], q, doc, src)


def run_synth(cfg: dict, gen_key: str) -> dict:
    """Gera com um gerador (vLLM). Retomável: pula sementes já processadas."""
    from vllm import LLM, SamplingParams

    scfg = cfg["synth"]
    keys = list(scfg["generators"])
    if gen_key not in keys:
        raise EmbedError(f"gerador desconhecido: {gen_key} (opções: {keys})")
    gcfg = scfg["generators"][gen_key]
    seed = int(cfg.get("seed", 1234))
    out_path = data_dir("synth", f"{gen_key}.jsonl")
    done = {r["source"]["seed_id"] + "|" + r["task"] for r in iter_jsonl(out_path)} if out_path.exists() else set()
    reqs = build_requests(cfg, gen_key, keys.index(gen_key), len(keys), seed)
    kind_task = {"retrieval": "retrieval", "long": "retrieval", "sts": "sts"}
    reqs = [r for r in reqs if f"{r['seed']['id']}|{kind_task[r['kind']]}" not in done]
    log.info("%s: %d pedidos novos (%d já feitos)", gen_key, len(reqs), len(done))
    if not reqs:
        return {"generator": gen_key, "new": 0}
    llm = LLM(model=gcfg["model"], revision=gcfg.get("revision"), seed=seed,
              max_model_len=int(gcfg.get("max_model_len", 8192)),
              gpu_memory_utilization=float(gcfg.get("gpu_memory_utilization", 0.9)),
              **gcfg.get("engine_kwargs", {}))
    sp = SamplingParams(temperature=float(gcfg.get("temperature", 0.8)), top_p=float(gcfg.get("top_p", 0.95)),
                        max_tokens=int(gcfg.get("max_tokens", 256)), seed=seed)
    system = gcfg.get("system")
    kept = rejected = 0
    chunk = int(scfg.get("chunk", 5000))
    for i in range(0, len(reqs), chunk):
        part = reqs[i : i + chunk]
        convs = [([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": r["prompt"]}]
                 for r in part]
        outs = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs=gcfg.get("chat_template_kwargs"))
        rows = []
        for r, o in zip(part, outs):
            ex = to_example(r, o.outputs[0].text, cfg, gen_key)
            if ex is None:
                rejected += 1
            else:
                rows.append(ex)
        kept += len(rows)
        with open(out_path, "a", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("%s: %d/%d pedidos | mantidos %d | rejeitados %d", gen_key, i + len(part), len(reqs), kept, rejected)
    return {"generator": gen_key, "new": kept, "rejected": rejected}
