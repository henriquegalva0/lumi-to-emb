"""Nomes dos modelos no cache do mteb (sem dependências pesadas)."""

MODEL_PREFIX = "lumi-embed"


def model_name(ckpt_id: str) -> str:
    return f"{MODEL_PREFIX}/{ckpt_id}"
