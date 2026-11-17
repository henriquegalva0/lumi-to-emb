"""Registro do ambiente (versões, GPUs, imagem) e dos desvios em relação à especificação."""

from __future__ import annotations

import importlib.metadata as md
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .common import REPO_ROOT, git_commit, workdir

PACKAGES = (
    "vllm", "torch", "transformers", "tokenizers", "huggingface_hub", "numpy",
    "pandas", "pyarrow", "matplotlib", "PyYAML", "flashinfer-python",
)


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_inventory() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": parts[0], "name": parts[1], "memory_mib": parts[2], "driver": parts[3]})
    return gpus


def driver_cuda_version() -> str | None:
    out = _run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([\d.]+)", out or "")
    return m.group(1) if m else None


def visible_gpu_ids(limit: str | int = "auto") -> list[str]:
    """IDs de GPU para lançar um worker em cada (respeita CUDA_VISIBLE_DEVICES)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() and cvd.strip() not in ("NoDevFiles", "-1"):
        ids = [x.strip() for x in cvd.split(",") if x.strip()]
    else:
        ids = [g["index"] for g in gpu_inventory()]
    if isinstance(limit, int) or (isinstance(limit, str) and limit.isdigit()):
        ids = ids[: int(limit)]
    return ids


def available_cpus() -> int:
    """CPUs que o job pode usar de fato (afinidade e cota do cgroup), não as do host inteiro."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    quotas = []
    try:  # cgroup v2
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            quotas.append(int(quota) / int(period))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            quotas.append(quota / period)
    except (OSError, ValueError):
        pass
    for q in quotas:
        n = min(n, max(1, math.ceil(q)))
    return max(1, n)


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def torch_cuda() -> str | None:
    try:
        import torch

        return torch.version.cuda
    except Exception:  # noqa: BLE001
        return None


def memory_total_gb() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text()
        kb = int(re.search(r"MemTotal:\s+(\d+)", text).group(1))
        return round(kb / 1024**2, 1)
    except (OSError, AttributeError):
        return None


def collect_env() -> dict:
    gpus = gpu_inventory()
    return {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lumi_ctx_version": __version__,
        "git": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch_cuda": torch_cuda(),
        "image_tag_env": os.environ.get("VLLM_IMAGE_TAG"),
        "image_vllm_build_commit": os.environ.get("VLLM_BUILD_COMMIT"),
        "image_cuda_version": os.environ.get("CUDA_VERSION"),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "driver_version": gpus[0]["driver"] if gpus else None,
        "driver_cuda_version": driver_cuda_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_count": os.cpu_count(),
        "cpu_available": available_cpus(),
        "memory_total_gb": memory_total_gb(),
        "workdir": str(workdir()),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_home_source": os.environ.get("LUMI_HF_HOME_SOURCE", "desconhecida"),
    }


class Deviations:
    """Lista de desvios em relação à especificação, gravada em jsonl e depois em DESVIOS.md."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._seen.add(json.loads(line)["code"])

    def add(self, code: str, what: str, why: str, impact: str = "") -> None:
        if code in self._seen:
            return
        self._seen.add(code)
        row = {"code": code, "what": what, "why": why, "impact": impact, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]


def record_static_deviations(dev: Deviations, cfg: dict, env: dict, backend: str) -> None:
    """Desvios conhecidos no início da execução."""
    expected_image = cfg["vllm"].get("image")
    if backend == "vllm" and env.get("image_tag_env") and env["image_tag_env"] != expected_image:
        dev.add(
            "image-mismatch",
            f"Imagem em uso ({env['image_tag_env']}) difere da fixada na config ({expected_image}).",
            "A imagem do job foi trocada.",
            "Versões registradas em env.json; compare antes de juntar resultados.",
        )
    names = {g["name"] for g in env.get("gpus", [])}
    if backend == "vllm" and names and not any("H100" in n for n in names):
        dev.add(
            "hardware",
            f"Hardware diferente do previsto na especificação (8× H100): {env['gpu_count']}× {', '.join(sorted(names))}.",
            "Cluster disponível para o experimento.",
            "Afeta só as medidas de desempenho (perf.csv); a acurácia não depende do hardware.",
        )
    for m in cfg["models"]:
        if m["key"] == "lumi" and m["hf_id"] != "ceia-br/lumi-4b":
            dev.add(
                "lumi-repo-id",
                f"Lumi carregado de `{m['hf_id']}` (revisão `{m['revision'][:12]}`) em vez de `ceia-br/lumi-4b`.",
                "O repositório foi renomeado; o id antigo redireciona para este.",
                "Nenhum: é o mesmo checkpoint, agora com revisão fixada.",
            )
    if env.get("hf_home_source") == "external":
        dev.add(
            "hf-cache",
            f"Cache do Hugging Face em `{env.get('hf_home')}` (definido pelo ambiente), não dentro do diretório de trabalho.",
            "HF_HOME já vinha definido no ambiente (no cluster usado, o gerenciador de jobs fornece um cache "
            "compartilhado e pede para não trocá-lo).",
            "Nenhum sobre os resultados.",
        )
    if Path(env.get("workdir", "")).resolve() == REPO_ROOT.resolve():
        dev.add(
            "workdir",
            "Saídas gravadas em subpastas do próprio clone (results/, data/, work/), todas ignoradas pelo Git.",
            "No servidor o clone é a própria pasta de trabalho do job; LUMI_WORKDIR permite separar.",
            "Nenhum: nada gerado entra no controle de versão.",
        )
    dev.add(
        "haystack-streaming",
        "Palheiro lido em streaming direto dos parquets do Hub (HfFileSystem + pyarrow), sem a biblioteca `datasets`.",
        "A versão atual de `datasets` exige um `huggingface_hub` mais novo que o da imagem do vLLM; "
        "instalá-la trocaria pacotes da imagem.",
        "Nenhum: mesma fonte, mesma revisão fixada e ordem determinística pela semente.",
    )
