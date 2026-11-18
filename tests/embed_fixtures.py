"""Modelo NemotronH minúsculo (CPU, float32) para os testes do Lumi-Embed."""

from __future__ import annotations

from pathlib import Path

LAYERS = ["linear_attention", "mlp", "full_attention", "linear_attention", "mlp", "full_attention", "linear_attention"]


def tiny_config(vocab_size: int):
    from transformers import NemotronHConfig

    return NemotronHConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        layers_block_type=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        mamba_num_heads=8,
        mamba_head_dim=16,
        ssm_state_size=16,
        n_groups=2,
        chunk_size=8,
        conv_kernel=4,
        max_position_embeddings=4096,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_mamba_kernels=False,
    )


def build_tiny_backbone_dir(out: Path, tok_dir: Path, seed: int = 0) -> Path:
    """Salva um backbone aleatório + o tokenizer minúsculo num diretório carregável."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    torch.manual_seed(seed)
    model = AutoModel.from_config(tiny_config(len(tok)), dtype=torch.float32)
    # inicialização com escala maior, para diferenças entre textos ficarem bem acima do ruído numérico
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.08)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out
