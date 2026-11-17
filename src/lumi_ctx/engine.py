"""Backends de geração.

- `VLLMEngine`: vLLM offline (`LLM.generate`) recebendo IDs de token prontos.
- `FakeEngine`: simulador determinístico, sem GPU, usado nos testes locais para
  exercitar todo o pipeline (fila, workers, pontuação, análise e relatório).
  Resultados com backend "fake" são marcados como SIMULADOS no relatório.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

ENGINE_INIT_FAILED = 3  # código de saída do worker quando o motor não sobe


@dataclass
class GenResult:
    text: str
    token_ids: list[int]
    finish_reason: str | None
    n_prompt: int
    metrics: dict = field(default_factory=dict)


def _extract_metrics(out) -> dict:
    """Coleta, se existirem, métricas por requisição do vLLM (variam entre versões)."""
    m = getattr(out, "metrics", None)
    if m is None:
        return {}
    keys = (
        "arrival_time", "first_scheduled_time", "first_token_time", "last_token_time",
        "time_in_queue", "first_token_latency", "num_generation_tokens", "scheduler_time",
    )
    res = {}
    for k in keys:
        v = getattr(m, k, None)
        if isinstance(v, (int, float)):
            res[k] = float(v)
    return res


class VLLMEngine:
    name = "vllm"

    def __init__(self, model: dict, cfg: dict, stop_ids: list[int], profile: str = "engine"):
        from vllm import LLM

        gen = cfg["generation"]
        engine_args = dict(cfg["vllm"][profile])
        self.args = dict(
            model=model["hf_id"],
            revision=model["revision"],
            tokenizer=model["hf_id"],
            tokenizer_revision=model["revision"],
            seed=int(gen["seed"]),
            **engine_args,
        )
        log.info("Iniciando vLLM (%s) com %s", profile, self.args)
        t0 = time.time()
        self.llm = LLM(**self.args)
        self.load_seconds = time.time() - t0
        self.stop_ids = stop_ids
        self.seed = int(gen["seed"])
        self.temperature = float(gen["temperature"])
        log.info("vLLM pronto em %.0fs", self.load_seconds)

    def _params(self, max_tokens: int, ignore_eos: bool = False):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=self.temperature,
            max_tokens=int(max_tokens),
            seed=self.seed,
            stop_token_ids=self.stop_ids,
            skip_special_tokens=False,
            ignore_eos=ignore_eos,
        )

    @staticmethod
    def _inputs(prompts: list[list[int]]):
        try:
            from vllm.inputs import TokensPrompt

            return [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
        except ImportError:  # pragma: no cover - versões antigas
            return [{"prompt_token_ids": ids} for ids in prompts]

    def generate(self, prompts: list[list[int]], max_tokens: list[int], ignore_eos: bool = False) -> list[GenResult]:
        params = [self._params(m, ignore_eos) for m in max_tokens]
        outs = self.llm.generate(self._inputs(prompts), sampling_params=params, use_tqdm=False)
        results = []
        for o in outs:
            c = o.outputs[0]
            results.append(
                GenResult(
                    text=c.text,
                    token_ids=[int(t) for t in c.token_ids],
                    finish_reason=c.finish_reason,
                    n_prompt=len(o.prompt_token_ids or []),
                    metrics=_extract_metrics(o),
                )
            )
        return results

    def close(self) -> None:
        """Libera o motor (e o processo EngineCore) antes de o worker sair."""
        import gc

        engine = getattr(self.llm, "llm_engine", None)
        for target in (getattr(engine, "engine_core", None), engine):  # v1: engine_core.shutdown()
            shutdown = getattr(target, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:  # noqa: BLE001
                    log.warning("shutdown do vLLM falhou: %s", exc)
                break
        del self.llm
        gc.collect()

    def info(self) -> dict:
        import vllm

        return {"backend": "vllm", "vllm_version": vllm.__version__, "args": self.args, "load_seconds": self.load_seconds}


# --------------------------------------------------------------------------- simulador

_Q_RE = re.compile(r"Pergunta: (?:Qual é o número mágico|Quais são todos os números mágicos) de (.+?)\?")
_N_RE = re.compile(r"O número mágico de (.+?) é (\d{7})\.")

# Perfis de "acurácia" do simulador: (comprimento a partir do qual cai, piso)
FAKE_PROFILES = {"lumi": (16384, 0.15), "nemotron": (131072, 0.6)}


class FakeEngine:
    """Responde lendo as agulhas do próprio prompt e erra com probabilidade que
    cresce com o comprimento (perfil por modelo). Determinístico."""

    name = "fake"

    def __init__(self, model: dict, cfg: dict, tok, stop_ids: list[int] | None = None, profile: str = "engine"):
        self.model_key = model["key"]
        self.tok = tok
        self.knee, self.floor = FAKE_PROFILES.get(self.model_key, (65536, 0.5))
        self.load_seconds = 0.0
        self.args = {"backend": "fake", "profile": profile}

    def _p_correct(self, n_tokens: int) -> float:
        if n_tokens <= self.knee:
            return 0.99
        # queda log-linear até o piso em 4x o joelho
        import math

        frac = min(1.0, math.log2(n_tokens / self.knee) / 2)
        return 0.99 - (0.99 - self.floor) * frac

    def _u(self, *parts) -> float:
        h = hashlib.sha256("|".join(map(str, (self.model_key,) + parts)).encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def generate(self, prompts: list[list[int]], max_tokens: list[int], ignore_eos: bool = False) -> list[GenResult]:
        out = []
        for ids in prompts:
            text = self.tok.decode(ids, skip_special_tokens=False)
            q = _Q_RE.search(text)
            city = q.group(1) if q else ""
            needles = _N_RE.findall(text)
            values = [v for c, v in needles if c == city]
            others = [v for c, v in needles if c != city]
            p = self._p_correct(len(ids))
            key = (len(ids), sum(ids[:: max(1, len(ids) // 64)]))
            if "Quais são" in (q.group(0) if q else ""):
                kept = [v for i, v in enumerate(values) if self._u(*key, i) < p]
                answer = ", ".join(kept) if kept else "Não encontrei."
            elif values and self._u(*key) < p:
                answer = values[0]
            elif others and self._u(*key, "d") < 0.5:
                answer = others[0]  # erro "realista": devolve o número de outra cidade
            else:
                answer = str(1_000_000 + int(self._u(*key, "x") * 8_999_999))
            out.append(GenResult(text=answer, token_ids=[], finish_reason="stop", n_prompt=len(ids)))
        return out

    def close(self) -> None:
        pass

    def info(self) -> dict:
        return {"backend": "fake", "args": self.args, "load_seconds": 0.0}
