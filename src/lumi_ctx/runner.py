"""Orquestrador da Etapa 1: check, prepare, run (smoke/full) e analyze."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import workqueue
from .common import (
    REPO_ROOT,
    Stage1Error,
    in_workdir,
    load_config,
    new_run_id,
    read_json,
    read_jsonl,
    results_dir,
    setup_logging,
    work_dir,
    workdir,
    write_json,
    write_jsonl,
    write_yaml,
)
from .engine import ENGINE_INIT_FAILED
from .envinfo import Deviations, collect_env, record_static_deviations, visible_gpu_ids
from .haystack import build_pool, load_pool
from .prompts import (
    build_prompt_set,
    check_template,
    load_prompt_set,
    load_tokenizer,
    render_text,
    stop_token_ids,
    think_token_ids,
    tokenizer_dir,
)
from .needles import needle_sentence, question_for, render_user_message
from .scoring import has_think_leak, score

log = logging.getLogger("lumi_ctx.runner")

MODE_CONFIGS = {"smoke": "smoke.yaml", "full": "stage1.yaml"}
POLL_SECONDS = 15
HANG_SECONDS = 600  # worker vivo sem nada na fila por mais que isso é encerrado


def ensure_hf_home() -> None:
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(workdir() / "hf_cache")
        os.environ.setdefault("LUMI_HF_HOME_SOURCE", "workdir")
    else:
        os.environ.setdefault("LUMI_HF_HOME_SOURCE", "external")


def build_workers(cfg: dict) -> int:
    value = cfg.get("build", {}).get("workers", "auto")
    if value == "auto":
        return max(1, min(os.cpu_count() or 1, 16))
    return max(1, int(value))


@contextlib.contextmanager
def phase(timing: dict, name: str):
    t0 = time.time()
    log.info("=== %s ===", name)
    try:
        yield
    finally:
        timing[name] = round(time.time() - t0, 1)
        log.info("=== %s: %.0fs ===", name, timing[name])


# --------------------------------------------------------------------------- preparação


def load_model_tokenizers(cfg: dict, dev: Deviations | None) -> dict:
    toks = {}
    for m in cfg["models"]:
        path = tokenizer_dir(m)
        tok = load_tokenizer(path)
        if getattr(tok, "_lumi_fallback", None) and dev is not None:
            dev.add(
                f"tokenizer-fallback-{m['key']}",
                f"Tokenizer de {m['key']} carregado como PreTrainedTokenizerFast genérico.",
                tok._lumi_fallback,
                "Mesmo vocabulário (tokenizer.json) e mesmo chat template do repositório.",
            )
        toks[m["key"]] = (tok, path)
    return toks


def template_report(cfg: dict, toks: dict, out_dir: Path | None) -> dict:
    """Confere o template e grava um prompt C0 completo, renderizado, para inspeção."""
    checks = {}
    for m in cfg["models"]:
        tok, _ = toks[m["key"]]
        chk = check_template(tok)
        example = render_text(
            tok,
            render_user_message(needle_sentence("Recife", "1234567"), question_for("C0", "Recife")),
            enable_thinking=cfg["generation"]["enable_thinking"],
        )
        chk["example"] = example
        checks[m["key"]] = chk
        log.info("Template de %s: %s\n----- exemplo renderizado -----\n%s\n-------------------------------",
                 m["key"], "OK (thinking desligado)" if chk["ok"] else "FALHOU", example)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"template_{m['key']}.txt").write_text(example, encoding="utf-8")
    return checks


def ensure_pool(cfg: dict, toks: dict, dev: Deviations | None):
    pool_path = in_workdir(cfg["haystack"]["file"])
    params = {k: cfg["haystack"][k] for k in ("target_tokens", "min_article_chars", "min_clean_chars", "min_line_chars")}
    if pool_path.exists():
        pool = load_pool(pool_path)
        same_source = pool.meta.get("source", {}).get("revision") in {s["revision"] for s in cfg["haystack"]["sources"]}
        if pool.meta.get("params") == params and pool.meta.get("seed") == cfg["experiment"]["seed"] and same_source:
            log.info("Pool do palheiro reaproveitado: %s (%d artigos, %.2fM tokens)",
                     pool_path, pool.meta.get("n_docs", -1), pool.meta.get("n_tokens", 0) / 1e6)
        else:
            log.warning("Pool existente foi feito com outros parâmetros; reconstruindo.")
            pool = None
    else:
        pool = None
    if pool is None:
        first = cfg["models"][0]["key"]
        build_pool(cfg, pool_path, toks[first][0], int(cfg["experiment"]["seed"]))
        pool = load_pool(pool_path)
    if pool.meta.get("fallback_used") and dev is not None:
        dev.add(
            "haystack-source",
            f"Palheiro montado com a fonte alternativa {pool.meta['source']['repo']}.",
            "A fonte principal (Wikipédia PT) não estava acessível: " + "; ".join(pool.meta.get("errors", [])),
            "Muda o estilo do texto de fundo; compare com cautela execuções com fontes diferentes.",
        )
    return pool, pool_path


def prepare_prompts(cfg: dict, toks: dict, pool, pool_path: Path) -> dict:
    out_root = workdir() / "work" / "prompts"
    sets: dict[str, dict] = {}
    for m in cfg["models"]:
        tok, path = toks[m["key"]]
        pset = build_prompt_set(cfg, tok, path, pool, pool_path, out_root, build_workers(cfg))
        sets[m["key"]] = {
            "prompt_key": pset.key,
            "prompt_dir": str(pset.directory.relative_to(workdir())),
            "tok_path": path,
            "stop_ids": stop_token_ids(tok),
            "think_ids": think_token_ids(tok),
            "stats": read_json(pset.directory / "stats.json"),
        }
    return sets


# --------------------------------------------------------------------------- geração


class WorkerHandle:
    def __init__(self, label: str, gpu: str):
        self.label = label
        self.gpu = gpu
        self.proc: subprocess.Popen | None = None
        self.profile = "engine"
        self.retries = 0
        self.done = False
        self.failed = False


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "(sem log)"


def run_model_workers(run_id: str, cfg: dict, model_key: str, gpus: list[str], backend: str, dev: Deviations) -> None:
    rdir, wdir = results_dir(run_id), work_dir(run_id)
    qdir = wdir / "queue" / model_key
    workqueue.reset_running(qdir)
    st = workqueue.status(qdir)
    if st["todo"] == 0:
        log.info("%s: nada pendente na fila (%d lotes concluídos)", model_key, st["done"])
        return
    perf_needed = not (wdir / "perf" / f"{model_key}.jsonl").exists()
    max_retries = int(cfg["vllm"].get("worker_retries", 1))
    env_base = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env_base["PYTHONPATH"] = src + (os.pathsep + env_base["PYTHONPATH"] if env_base.get("PYTHONPATH") else "")
    env_base.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    handles = [WorkerHandle(f"gpu{g}", g) for g in gpus]

    def launch(h: WorkerHandle, perf: bool) -> None:
        env = dict(env_base)
        if backend == "vllm":
            env["CUDA_VISIBLE_DEVICES"] = h.gpu
        cmd = [sys.executable, "-m", "lumi_ctx.worker", "--run-id", run_id, "--model", model_key,
               "--label", h.label, "--profile", h.profile]
        if perf:
            cmd.append("--perf")
        out = open(rdir / "logs" / f"worker_{model_key}_{h.label}.stdout.log", "a", encoding="utf-8")
        h.proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
        h.done = False
        log.info("Worker %s/%s lançado (pid %d, perfil %s%s)", model_key, h.label, h.proc.pid, h.profile,
                 ", com benchmark" if perf else "")

    for i, h in enumerate(handles):
        launch(h, perf=perf_needed and i == 0)

    t0 = time.time()
    tokens_at_start = st["tokens_done"]
    last_progress = 0.0
    spare_relaunches = 2 * len(handles)
    idle_since: float | None = None
    while True:
        time.sleep(POLL_SECONDS if backend == "vllm" else 1)
        for h in handles:
            if h.done or h.proc is None:
                continue
            rc = h.proc.poll()
            if rc is None:
                continue
            log_file = rdir / "logs" / f"worker_{model_key}_{h.label}.log"
            if rc == 0:
                h.done = True
                continue
            back = workqueue.reset_running(qdir, h.label)
            if rc == ENGINE_INIT_FAILED and h.profile == "engine":
                log.error("Motor não subiu em %s com as opções principais; tentando o perfil mínimo.\n%s",
                          h.label, _tail(log_file))
                dev.add(
                    f"engine-fallback-{model_key}",
                    f"vLLM iniciado com `vllm.fallback_engine` para {model_key} (as opções de `vllm.engine` falharam).",
                    "Erro na inicialização do motor; ver logs/worker_*.log.",
                    "Pode mudar o desempenho medido; a acurácia não depende dessas opções.",
                )
                h.profile = "fallback_engine"
                launch(h, perf=perf_needed and h is handles[0])
            elif rc != ENGINE_INIT_FAILED and h.retries < max_retries:
                h.retries += 1
                log.error("Worker %s caiu (código %s); %d lote(s) devolvido(s). Relançando (%d/%d).\n%s",
                          h.label, rc, back, h.retries, max_retries, _tail(log_file))
                launch(h, perf=False)
            else:
                h.done, h.failed = True, True
                log.error("Worker %s desistiu (código %s).\n%s", h.label, rc, _tail(log_file))

        st = workqueue.status(qdir)
        alive = [h for h in handles if not h.done]
        if alive and st["todo"] == 0 and st["run"] == 0:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > HANG_SECONDS:
                for h in alive:
                    log.warning("Worker %s não encerrou %ds depois de a fila esvaziar; encerrando à força.",
                                h.label, HANG_SECONDS)
                    h.proc.kill()
                    h.proc.wait(timeout=60)
                    h.done = True
                alive = []
        else:
            idle_since = None
        if not alive and st["todo"] > 0 and spare_relaunches > 0:
            healthy = [h for h in handles if not h.failed]
            if healthy:
                spare_relaunches -= 1
                log.warning("Restam %d lotes sem worker; relançando em %s", st["todo"], healthy[0].label)
                launch(healthy[0], perf=False)
                continue
        if not alive:
            if st["todo"] or st["run"]:
                raise Stage1Error(
                    f"Todos os workers de {model_key} pararam com {st['todo'] + st['run']} lote(s) pendente(s). "
                    f"Veja {rdir / 'logs'} (worker_{model_key}_*.log) e rode './run_stage1.sh diagnose'. "
                    f"Reenviar o job com o mesmo RUN_ID retoma de onde parou."
                )
            break
        if time.time() - last_progress > (60 if backend == "vllm" else 5):
            elapsed = time.time() - t0
            rate = (st["tokens_done"] - tokens_at_start) / max(elapsed, 1e-9)
            remaining = st["tokens_total"] - st["tokens_done"]
            eta = f"{remaining / rate / 60:.0f} min" if rate > 0 else "?"
            log.info("%s: %d/%d prompts (%.0f%% dos tokens), %d worker(s) ativo(s), ETA ~%s",
                     model_key, st["prompts_done"], st["prompts_total"],
                     100 * st["tokens_done"] / max(st["tokens_total"], 1), len(alive), eta)
            last_progress = time.time()


def download_models(cfg: dict) -> None:
    from huggingface_hub import snapshot_download

    for m in cfg["models"]:
        log.info("Baixando/conferindo pesos de %s@%s ...", m["hf_id"], m["revision"][:12])
        snapshot_download(repo_id=m["hf_id"], revision=m["revision"])


# --------------------------------------------------------------------------- junção e pontuação


def collect_results(run_id: str, cfg: dict, plan: dict) -> list[dict]:
    wdir = work_dir(run_id)
    rows: list[dict] = []
    for m in cfg["models"]:
        mplan = plan["models"][m["key"]]
        pset = load_prompt_set(in_workdir(mplan["prompt_dir"]))
        outputs: dict[str, dict] = {}
        odir = wdir / "outputs" / m["key"]
        for f in sorted(odir.glob("*.jsonl")) if odir.exists() else []:
            for rec in read_jsonl(f):
                outputs[rec["uid"]] = rec
        think_ids = set(mplan.get("think_ids", []))
        for p in pset.index:
            row = {
                "model": m["key"], "model_label": m["label"], "role": m["role"], "uid": p["uid"],
                "task": p["task"], "length": p["length"], "n_tokens": p["n_tokens"], "len_dev": p["len_dev"],
                "depth": p["depth"], "rep": p["rep"], "city": p["city"], "answers": p["answers"],
                "distractors": p["distractors"], "target_depth_actual": p.get("target_depth_actual"),
                "needle_depths": [n.get("depth_actual") for n in p["needles"]],
                "resamples": p.get("resamples", 0), "gap_relaxed": bool(p.get("gap_relaxed", False)),
            }
            out = outputs.get(p["uid"])
            if out is None:
                row.update(status="missing")
            else:
                sc = score(p["task"], out["text"], p["answers"], p["distractors"])
                row.update(
                    status="ok", output=out["text"], finish_reason=out["finish_reason"], n_out=out["n_out"],
                    think_leak=has_think_leak(out["text"], out["token_ids"], think_ids), worker=out["worker"], **sc,
                )
            rows.append(row)
    return rows


def collect_throughput(run_id: str, cfg: dict) -> list[dict]:
    rows = []
    for m in cfg["models"]:
        odir = work_dir(run_id) / "outputs" / m["key"]
        for f in sorted(odir.glob("*.meta.json")) if odir.exists() else []:
            meta = read_json(f)
            meta["model"] = m["key"]
            meta["tok_per_s"] = meta["prompt_tokens"] / max(meta["seconds"], 1e-9)
            rows.append(meta)
    return rows


def collect_perf(run_id: str, cfg: dict) -> list[dict]:
    rows = []
    for m in cfg["models"]:
        f = work_dir(run_id) / "perf" / f"{m['key']}.jsonl"
        if f.exists():
            rows.extend(read_jsonl(f))
    return rows


# --------------------------------------------------------------------------- modos


def cmd_check(config: str) -> int:
    """Confere ambiente, acesso ao Hub, tokenizers, template e config dos modelos (sem GPU)."""
    ensure_hf_home()
    cfg = load_config(config)
    out_dir = results_dir(new_run_id("check"))
    setup_logging(out_dir / "logs" / "main.log")
    env = collect_env()
    problems: list[str] = []
    warnings: list[str] = []
    log.info("Ambiente: python %s | vllm %s | torch %s (CUDA %s) | transformers %s",
             env["python"], env["packages"].get("vllm"), env["packages"].get("torch"), env["torch_cuda"],
             env["packages"].get("transformers"))
    log.info("GPUs: %s | driver %s | CUDA do driver %s", [g["name"] for g in env["gpus"]] or "nenhuma",
             env["driver_version"], env["driver_cuda_version"])
    if not env["packages"].get("vllm"):
        warnings.append("vLLM não instalado: só os modos sem GPU (testes, prepare, analyze, backend fake) funcionam aqui")
    if not env["gpus"]:
        warnings.append("nenhuma GPU visível (nvidia-smi)")
    if env.get("image_tag_env") and env["image_tag_env"] != cfg["vllm"]["image"]:
        warnings.append(f"imagem {env['image_tag_env']} != config {cfg['vllm']['image']}")
    toks = load_model_tokenizers(cfg, None)
    checks = template_report(cfg, toks, out_dir)
    for key, chk in checks.items():
        if not chk["ok"]:
            problems.append(f"template de {key} não desliga o thinking")
    from .analysis import model_memory

    mem = {}
    for m in cfg["models"]:
        try:
            mem[m["key"]] = model_memory(Path(toks[m["key"]][1]) / "config.json")
            log.info("Memória teórica %s: %s", m["key"], mem[m["key"]])
            tsl = mem[m["key"]].get("time_step_limit")
            if tsl is not None and not all(isinstance(x, (int, float)) for x in tsl):
                warnings.append(f"{m['key']}: time_step_limit não numérico ({tsl})")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"config de {m['key']}: {exc}")
    try:
        from .haystack import _hf_paths

        _, paths = _hf_paths(cfg["haystack"]["sources"][0])
        log.info("Fonte do palheiro acessível: %d arquivo(s)", len(paths))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"fonte do palheiro inacessível: {exc}")
    free_gb = shutil.disk_usage(workdir()).free / 1e9
    if free_gb < 30:
        warnings.append(f"pouco espaço livre em {workdir()}: {free_gb:.0f} GB")
    report = {"env": env, "templates": checks, "memory": mem, "problems": problems, "warnings": warnings,
              "disk_free_gb": round(free_gb, 1)}
    write_json(out_dir / "check.json", report)
    for w in warnings:
        log.warning("AVISO: %s", w)
    for p in problems:
        log.error("PROBLEMA: %s", p)
    log.info("Relatório do check: %s", out_dir / "check.json")
    return 1 if problems else 0


def cmd_prepare(config: str) -> int:
    ensure_hf_home()
    cfg = load_config(config)
    setup_logging(workdir() / "logs" / f"prepare_{time.strftime('%Y%m%d-%H%M%S')}.log")
    toks = load_model_tokenizers(cfg, None)
    pool, pool_path = ensure_pool(cfg, toks, None)
    sets = prepare_prompts(cfg, toks, pool, pool_path)
    for key, s in sets.items():
        log.info("%s: prompts em %s | %s", key, s["prompt_dir"], s["stats"])
    return 0


def latest_incomplete_run(mode: str) -> str | None:
    """Execução mais recente deste modo que começou e não gerou REPORT.md."""
    root = workdir() / "results"
    if not root.exists():
        return None
    cands = sorted((d for d in root.iterdir() if d.is_dir() and d.name.endswith(f"_{mode}")), reverse=True)
    for d in cands:
        if (d / "run_info.json").exists() and not (d / "REPORT.md").exists():
            return d.name
        return None  # só a mais recente conta
    return None


def cmd_run(mode: str, config: str | None, run_id: str | None, backend: str, fake_gpus: int,
            resume_latest: bool = False) -> int:
    ensure_hf_home()
    cfg = load_config(config or MODE_CONFIGS[mode])
    if not run_id and resume_latest:
        run_id = latest_incomplete_run(mode)
        if run_id:
            print(f"Retomando a execução incompleta {run_id} (use FRESH=1 para começar outra).", flush=True)
    resumed = bool(run_id and results_dir(run_id).exists())
    run_id = run_id or new_run_id(mode)
    rdir, wdir = results_dir(run_id), work_dir(run_id)
    (rdir / "logs").mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)
    setup_logging(rdir / "logs" / "main.log")
    log.info("Execução %s (%s, backend=%s%s). Diretório de trabalho: %s",
             run_id, mode, backend, ", retomando" if resumed else "", workdir())
    t_start = time.time()
    info_path = rdir / "run_info.json"
    info = read_json(info_path) if resumed and info_path.exists() else {}
    timing: dict = {}
    info.update(run_id=run_id, mode=mode, backend=backend, config_file=cfg["_meta"]["config_file"],
                started_at=info.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                resumed=resumed or info.get("resumed", False))
    dev = Deviations(rdir / "deviations.jsonl")

    with phase(timing, "ambiente"):
        env = collect_env()
        write_json(rdir / "env.json", env)
        freeze = os.environ.get("LUMI_PIP_FREEZE")
        if freeze and Path(freeze).exists():
            shutil.copy(freeze, rdir / "pip_freeze.txt")
        record_static_deviations(dev, cfg, env, backend)
        cfg_used = dict(cfg)
        cfg_used["_runtime"] = {"run_id": run_id, "mode": mode, "backend": backend, "workdir": str(workdir()),
                                "git": env["git"], "image_tag_env": env.get("image_tag_env")}
        write_yaml(rdir / "config_used.yaml", cfg_used)
        if backend == "vllm":
            gpus = visible_gpu_ids(cfg["vllm"].get("gpus", "auto"))
            if not gpus:
                raise Stage1Error("Nenhuma GPU visível (nvidia-smi/CUDA_VISIBLE_DEVICES). O job pediu GPU?")
        else:
            gpus = [f"f{i}" for i in range(max(1, fake_gpus))]
        info["gpus"] = gpus
        log.info("GPUs usadas: %s", gpus)

    with phase(timing, "tokenizers_e_template"):
        toks = load_model_tokenizers(cfg, dev)
        info["template_checks"] = template_report(cfg, toks, rdir)
        from .analysis import model_memory

        info["model_memory"] = {}
        for key, (_, path) in toks.items():
            try:
                info["model_memory"][key] = model_memory(Path(path) / "config.json")
            except Exception as exc:  # noqa: BLE001
                log.warning("Memória teórica de %s indisponível: %s", key, exc)
        bad = [k for k, c in info["template_checks"].items() if not c["ok"]]
        if bad:
            raise Stage1Error(f"O chat template não desligou o thinking para: {bad}. Abortando (gate G1).")

    with phase(timing, "palheiro"):
        pool, pool_path = ensure_pool(cfg, toks, dev)
        info["pool"] = pool.meta

    with phase(timing, "prompts"):
        sets = prepare_prompts(cfg, toks, pool, pool_path)
        plan = {"run_id": run_id, "backend": backend, "models": sets}
        write_json(wdir / "plan.json", plan)
        keys = {s["prompt_key"] for s in sets.values()}
        info["prompt_sets"] = {k: {kk: v for kk, v in s.items() if kk != "tok_path"} for k, s in sets.items()}
        info["paired_identical"] = len(keys) == 1
        c0_mean = next(iter(sets.values()))["stats"].get("c0_tokens_mean", 0)
        if c0_mean and abs(c0_mean - 150) / 150 > 0.2:
            dev.add(
                "c0-length",
                f"Prompts C0 com ~{c0_mean:.0f} tokens (o plano previa cerca de 150).",
                "É o tamanho real de template + agulha + pergunta com este tokenizer.",
                "Nenhum: o C0 continua sendo o controle sem palheiro.",
            )
        for key, s in sets.items():
            if s["stats"]["n_outside_tol"] or s["stats"]["n_over_target"]:
                raise Stage1Error(f"{key}: prompts fora da tolerância de comprimento: {s['stats']}")
        write_json(info_path, info)

    if backend == "vllm":
        with phase(timing, "download_modelos"):
            download_models(cfg)

    for m in cfg["models"]:
        with phase(timing, f"geracao_{m['key']}"):
            pset = load_prompt_set(in_workdir(sets[m["key"]]["prompt_dir"]))
            qdir = wdir / "queue" / m["key"]
            key_file = qdir / "PROMPT_KEY"
            if key_file.exists() and key_file.read_text().strip() != pset.key:
                raise Stage1Error(
                    f"A fila de {m['key']} em {run_id} foi criada com outro conjunto de prompts "
                    f"({key_file.read_text().strip()} ≠ {pset.key}); a config ou o código mudou. "
                    "Comece uma execução nova (FRESH=1 ./run_stage1.sh ...)."
                )
            workqueue.create_chunks(qdir, pset.index, int(cfg["vllm"]["chunk_tokens"]), int(cfg["vllm"]["chunk_max_prompts"]))
            key_file.write_text(pset.key + "\n")
            run_model_workers(run_id, cfg, m["key"], gpus, backend, dev)
        info.setdefault("engine_info", {})[m["key"]] = [
            read_json(f) for f in sorted(wdir.glob(f"engine_info_{m['key']}_*.json"))
        ]

    with phase(timing, "pontuacao"):
        rows = collect_results(run_id, cfg, plan)
        write_jsonl(rdir / "raw" / "results.jsonl", rows)
        write_jsonl(rdir / "raw" / "throughput.jsonl", collect_throughput(run_id, cfg))
        write_jsonl(rdir / "raw" / "perf.jsonl", collect_perf(run_id, cfg))
        missing = sum(1 for r in rows if r["status"] != "ok")
        log.info("%d resultados (%d faltando)", len(rows), missing)

    info["timing"] = {**info.get("timing", {}), **timing}
    info["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    info["duration_s"] = round(info.get("duration_s", 0) + time.time() - t_start, 1)
    write_json(info_path, info)

    from .analysis import analyze_run

    analyze_run(run_id)
    log.info("Pronto. Resultados em %s", rdir)
    return 0


def cmd_analyze(run_id: str) -> int:
    rdir = results_dir(run_id)
    if not rdir.exists():
        raise Stage1Error(f"Execução não encontrada: {rdir}")
    setup_logging(rdir / "logs" / "analyze.log")
    from .analysis import analyze_run

    analyze_run(run_id)
    return 0
