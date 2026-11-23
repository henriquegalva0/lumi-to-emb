"""Teste de carregamento do modelo publicado, do jeito que o model card ensina.

1. Branches: cada branch do repositório tem só os pesos do próprio checkpoint (os shards
   presentes são exatamente os do índice) e o config.
2. Sentence-Transformers a partir do Hub (trecho do card) contra o checkpoint local.
3. Transformers a partir do Hub (trecho do card) contra o checkpoint local.
4. Kernels pelo trecho do card: ativos (o scan Mamba-2 em PyTorch puro não roda) e mesmo resultado.
5. Sanidade: a query do card fica mais perto do documento dela que de documentos sem relação.

Saída: results/embed/hubcheck.json. Falha (EmbedError) se algum item falhar, depois de gravar o JSON.
"""

from __future__ import annotations

import json
import logging

from .utils import EmbedError, ckpt_dir, results_dir, write_json

log = logging.getLogger("lumi_embed.hubcheck")

# Textos e instrução copiados do model card (docs/huggingface/README.md, seção "Uso").
CARD_QUERY = "quais as regras para dedução de despesas médicas no imposto de renda?"
CARD_DOC = "As despesas médicas podem ser deduzidas da base de cálculo do imposto de renda."
TF_QUERY = "o que é inflação?"
TF_DOC = "Inflação é o aumento generalizado e contínuo dos preços."
TF_INSTRUCTION = "Dada uma pergunta, recupere passagens que a respondam"
DISTRACTORS = [
    "O Rio Amazonas é o maior rio do mundo em volume de água.",
    "A seleção brasileira de futebol venceu a Copa do Mundo cinco vezes.",
    "Bolo de cenoura com cobertura de chocolate é uma receita tradicional.",
]
MIN_COS = 0.995  # bf16 do Hub (PyTorch puro ou kernels) contra o checkpoint local
INDEX = "model.safetensors.index.json"


def branch_hygiene(api, repo: str) -> dict:
    """Shards de cada branch = os do índice (ou um model.safetensors só), mais o config.json."""
    from huggingface_hub import hf_hub_download

    out = {}
    for b in api.list_repo_refs(repo, repo_type="model").branches:
        files = set(api.list_repo_files(repo, revision=b.name, repo_type="model"))
        shards = sorted(f for f in files if f.endswith(".safetensors"))
        if INDEX in files:
            with open(hf_hub_download(repo, INDEX, revision=b.name)) as fh:
                want = sorted(set(json.load(fh)["weight_map"].values()))
        else:
            want = ["model.safetensors"]
        out[b.name] = {"shards": shards, "expected": want, "config": "config.json" in files,
                       "meta": "meta.json" in files, "ok": shards == want and "config.json" in files}
    return out


def card_sentence_transformers(repo: str, device: str):
    """Trecho "Sentence-Transformers" do card (com o dispositivo explícito)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(repo, model_kwargs={"dtype": "bfloat16"}, device=device)
    q = model.encode([CARD_QUERY], prompt_name="query", convert_to_tensor=True)
    d = model.encode([CARD_DOC] + DISTRACTORS, convert_to_tensor=True)
    sim = model.similarity(q, d)
    return q.float().cpu(), d.float().cpu(), sim.float().cpu(), model.max_seq_length


def card_transformers(repo: str, device: str):
    """Trecho "Transformers" do card (com o dispositivo explícito)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to(device).eval()

    def embed(texts, instruction=None):
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
        batch = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**batch, use_cache=False).last_hidden_state
        last = batch["attention_mask"].sum(dim=1) - 1
        return F.normalize(h[torch.arange(h.shape[0], device=h.device), last].float(), dim=-1)

    return model, embed


def _cos(a, b) -> float:
    import torch.nn.functional as F

    return float(F.cosine_similarity(a.float().cpu(), b.float().cpu(), dim=-1).min())


def run_hubcheck(cfg: dict, repo: str | None = None, local: str | None = None, device: str | None = None,
                 api=None) -> dict:
    import gc

    import torch

    from .modeling.embedder import LumiEmbedder
    from .modeling.export import DEFAULT_QUERY_INSTRUCTION

    repo = repo or cfg["hub"]["repo"]
    local = local or str(ckpt_dir("lumi-emb") / "model")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    res: dict = {"repo": repo, "local": local, "device": device, "min_cos": MIN_COS}

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    res["branches"] = branch_hygiene(api, repo)
    log.info("branches: %s", {b: r["ok"] for b, r in res["branches"].items()})

    def free():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ref = LumiEmbedder.from_pretrained(local, device=device, use_hub_kernels=device == "cuda").eval()
    rq = ref.encode([CARD_QUERY], instruction=DEFAULT_QUERY_INSTRUCTION)
    rd = ref.encode([CARD_DOC] + DISTRACTORS)
    rq2 = ref.encode([TF_QUERY], instruction=TF_INSTRUCTION)
    rd2 = ref.encode([TF_DOC])
    del ref
    free()

    q, d, sim, max_len = card_sentence_transformers(repo, device)
    res["sentence_transformers"] = {
        "cos_query": _cos(q, rq), "cos_docs": _cos(d, rd), "max_seq_length": max_len,
        "similarity": [round(float(x), 4) for x in sim[0]],
        "relevant_first": bool(sim[0, 0] > sim[0, 1:].max()),
    }
    st = res["sentence_transformers"]
    st["ok"] = st["cos_query"] >= MIN_COS and st["cos_docs"] >= MIN_COS and max_len == 512
    log.info("Sentence-Transformers: %s", st)
    free()

    model, embed = card_transformers(repo, device)
    tq, td = embed([TF_QUERY], TF_INSTRUCTION), embed([TF_DOC])
    res["transformers"] = {"cos_query": _cos(tq, rq2), "cos_doc": _cos(td, rd2),
                           "similarity": round(float(tq @ td.T), 4)}
    tr = res["transformers"]
    tr["ok"] = tr["cos_query"] >= MIN_COS and tr["cos_doc"] >= MIN_COS
    log.info("Transformers: %s", tr)

    if device == "cuda":
        from kernels import Mode

        from .modeling.kernels import probe_mamba_path

        model.set_use_kernels(True, mode=Mode.TRAINING)  # trecho "Kernels" do card
        kq, kd = embed([TF_QUERY], TF_INSTRUCTION), embed([TF_DOC])
        probe = probe_mamba_path(model)
        res["kernels"] = {"cos_query": _cos(kq, tq), "cos_doc": _cos(kd, td), **probe}
        kr = res["kernels"]
        kr["ok"] = (kr["cos_query"] >= MIN_COS and kr["cos_doc"] >= MIN_COS
                    and probe["torch_scan_calls_eval"] == 0 and probe["torch_scan_calls_train"] == 0)
        log.info("Kernels: %s", kr)
    else:
        res["kernels"] = {"ok": None, "skipped": "sem CUDA"}
    del model
    free()

    checks = {"branches": all(r["ok"] for r in res["branches"].values()) and bool(res["branches"]),
              "sentence_transformers": st["ok"], "transformers": tr["ok"], "kernels": res["kernels"]["ok"],
              "sanity": st["relevant_first"]}
    res["checks"] = checks
    res["ok"] = all(v is not False for v in checks.values())
    write_json(results_dir("hubcheck.json"), res)
    log.info("hubcheck: %s", checks)
    if not res["ok"]:
        bad = [k for k, v in checks.items() if v is False]
        raise EmbedError(f"Teste de carregamento do Hub falhou em: {', '.join(bad)} (ver results/embed/hubcheck.json)")
    return res
