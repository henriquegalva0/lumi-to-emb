"""Execução das suítes de avaliação (seção 5.1 do plano) com o mteb.

Suítes (configs/embed/eval.yaml): smoke, mteb_br (22 tarefas nativas, arXiv:2607.04581),
mteb_pt (MTEB(por, v1), 14 datasets, arXiv:2607.04071), longembed, eng_v2, mmteb.
Os resultados ficam no cache do mteb em results/embed/mteb/ (um JSON por tarefa),
retomáveis (tarefas já avaliadas são puladas).
"""

from __future__ import annotations

import logging
import time

from ..utils import EmbedError, ckpt_dir, git_commit, results_dir, write_json
from .names import MODEL_PREFIX, model_name  # noqa: F401

log = logging.getLogger("lumi_embed.eval")


def suite_tasks(ecfg: dict, suite: str):
    import mteb

    s = ecfg["suites"].get(suite)
    if s is None:
        raise EmbedError(f"suíte desconhecida: {suite} (opções: {list(ecfg['suites'])})")
    tasks = []
    if s.get("mteb_br"):
        import mteb_pt
        import mteb_pt.register  # noqa: F401

        names = s["mteb_br"] if isinstance(s["mteb_br"], list) else mteb_pt.HEADLINE_TASKS
        tasks += list(mteb.get_tasks(tasks=names))
    if s.get("benchmark"):
        tasks += list(mteb.get_benchmark(s["benchmark"]).tasks)
    if s.get("tasks"):
        tasks += list(mteb.get_tasks(tasks=s["tasks"], languages=s.get("languages")))
    seen, out = set(), []
    for t in tasks:
        if t.metadata.name not in seen:
            seen.add(t.metadata.name)
            out.append(t)
    return out


def checkpoint_path(ckpt_id: str) -> str:
    base = "R0" if ckpt_id == "R0e" else ckpt_id  # R0e usa os pesos do R0 com echo na inferência
    path = ckpt_dir(base) / "model"
    if not (path / "config.json").exists():
        raise EmbedError(f"Checkpoint {base} não encontrado em {path}.")
    return str(path)


def suite_params(ecfg: dict, suite: str, model_batch: int | None = None) -> tuple[int, int]:
    """(comprimento máximo, lote) de uma suíte, iguais para todos os modelos.

    O comprimento vem da suíte (ex.: LongEmbed) ou do protocolo (`eval.max_length`), nunca do modelo:
    sem isso, um baseline com contexto de 32k codificaria documentos inteiros enquanto o nosso corta
    em 512. O lote é o do modelo, limitado pelo da suíte quando ela define um (ex.: 1 no LongEmbed).
    """
    s = ecfg["suites"][suite]
    max_len = int(s.get("max_length") or ecfg.get("max_length", 512))
    bs = int(model_batch or ecfg["batch_size"])
    if s.get("batch_size"):
        bs = min(bs, int(s["batch_size"]))
    return max_len, bs


def cache_name(base: str, ecfg: dict, suite: str) -> str:
    """Nome do modelo no cache do mteb para a suíte. Com `cache_tag` (ex.: MLDR-pt em 8k), o nome ganha
    um sufixo, para os resultados não se misturarem com os da mesma tarefa em 512 tokens."""
    tag = ecfg["suites"][suite].get("cache_tag")
    return f"{base}-{tag}" if tag else base


def cap_length(model, max_len: int) -> bool:
    """Limita o comprimento de um encoder do mteb baseado em Sentence-Transformers."""
    st = getattr(model, "model", None)
    if st is not None and hasattr(st, "max_seq_length"):
        st.max_seq_length = max_len
        return True
    return False


def evaluate_checkpoint(cfg: dict, ckpt_id: str, suites: list[str]) -> dict:
    import mteb
    from mteb.cache import ResultCache

    from .mteb_encoder import build_encoder, spec_override_for

    ecfg = cfg["eval"]
    cache = ResultCache(results_dir("mteb"))
    summary = {}
    for suite in suites:
        max_len, bs = suite_params(ecfg, suite)
        enc = build_encoder(checkpoint_path(ckpt_id), cache_name(model_name(ckpt_id), ecfg, suite), git_commit()["commit"][:12],
                            ecfg["instructions"], batch_size=bs, max_length=max_len,
                            spec_overrides=spec_override_for(ckpt_id))
        tasks = suite_tasks(ecfg, suite)
        write_json(results_dir("suites", f"{suite}.json"), [t.metadata.name for t in tasks])
        log.info("Avaliando %s em %s (%d tarefas)", ckpt_id, suite, len(tasks))
        t0 = time.time()
        mteb.evaluate(enc, tasks, cache=cache, overwrite_strategy="only-missing", show_progress_bar=False,
                      raise_error=bool(ecfg.get("raise_error", False)), encode_kwargs={"batch_size": bs})
        enc.save_instructions()
        summary[suite] = {"tasks": len(tasks), "max_length": max_len, "batch_size": bs,
                          "minutes": round((time.time() - t0) / 60, 1)}
        del enc
        _free()
    _write_run(results_dir("runs", f"{ckpt_id}.json"), summary)
    return summary


def evaluate_baseline(cfg: dict, hf_id: str, suites: list[str]) -> dict:
    """Modelos de comparação (Tabela 14). Modelos com instrução recebem as MESMAS instruções."""
    import mteb
    from mteb.cache import ResultCache

    ecfg = cfg["eval"]
    bcfg = ecfg["baselines"].get(hf_id, {})
    cache = ResultCache(results_dir("mteb"))
    summary = {}
    for suite in suites:
        tasks = suite_tasks(ecfg, suite)
        max_len, bs = suite_params(ecfg, suite, bcfg.get("batch_size"))
        if bcfg.get("instruct_template"):
            from mteb.models.instruct_wrapper import InstructSentenceTransformerModel

            prompts = {**ecfg["instructions"].get("by_type", {}), **ecfg["instructions"].get("by_task", {})}
            model = InstructSentenceTransformerModel(
                hf_id, revision=bcfg.get("revision"), instruction_template=bcfg["instruct_template"],
                apply_instruction_to_passages=False, prompts_dict=prompts,
                model_kwargs={"dtype": "bfloat16"}, max_seq_length=max_len,
            )
            model.mteb_model_meta = mteb.get_model_meta(hf_id)
        else:
            model = mteb.get_model(hf_id, revision=bcfg.get("revision"))
        meta = getattr(model, "mteb_model_meta", None)
        if meta is not None and cache_name(hf_id, ecfg, suite) != hf_id:
            model.mteb_model_meta = meta.model_copy(update={"name": cache_name(meta.name, ecfg, suite)})
        if not cap_length(model, max_len):
            raise EmbedError(f"Não foi possível limitar {hf_id} a {max_len} tokens (protocolo: mesmo comprimento "
                             "para todos os modelos). Adapte cap_length() ao wrapper deste modelo.")
        log.info("Baseline %s em %s: max_length %d, lote %d", hf_id, suite, max_len, bs)
        t0 = time.time()
        mteb.evaluate(model, tasks, cache=cache, overwrite_strategy="only-missing", show_progress_bar=False,
                      raise_error=bool(ecfg.get("raise_error", False)), encode_kwargs={"batch_size": bs})
        summary[suite] = {"tasks": len(tasks), "max_length": max_len, "batch_size": bs,
                          "minutes": round((time.time() - t0) / 60, 1)}
        del model
        _free()
    _write_run(results_dir("runs", f"baseline__{hf_id.replace('/', '__')}.json"), summary)
    return summary


def _write_run(path, summary: dict) -> None:
    """Acumula as suítes no resumo do modelo: avaliar outra suíte não apaga o registro das anteriores."""
    import json

    try:
        old = json.loads(path.read_text())
    except (OSError, ValueError):
        old = {}
    write_json(path, {**old, **summary})


def _free():
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
