"""Verificações obrigatórias por checkpoint (seção 3.6 do plano), num NemotronH minúsculo.

T1 invariância a padding e a lote · T2 causalidade por trilho · T3 posição do pooling
· T4 salvar e recarregar (lumi_embed e Sentence-Transformers).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")  # só no ambiente de treino (requirements-embed.txt)

import torch.nn.functional as F  # noqa: E402

from embed_fixtures import build_tiny_backbone_dir  # noqa: E402
from lumi_embed.modeling.embedder import EmbedSpec, LumiEmbedder  # noqa: E402
from lumi_embed.modeling.export import save_embedder  # noqa: E402

TEXTS = [
    "A cidade foi fundada no primeiro século e tem uma universidade antiga.",
    "Rio.",
    "O governo da província construiu uma ferrovia entre a capital e o porto durante a década.",
    "Museu de arte.",
]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, tiny_tok_dir):
    return build_tiny_backbone_dir(tmp_path_factory.mktemp("tiny_lumi"), tiny_tok_dir)


def load(tiny_dir, **spec):
    return LumiEmbedder.from_pretrained(str(tiny_dir), spec=spec, dtype=torch.float32, use_hub_kernels=False).eval()


@pytest.mark.parametrize("direction,pooling", [("C", "eos"), ("Ba", "mean"), ("Bf", "mean"), ("Ba", "eos")])
def test_T1_padding_and_batch_invariance(tiny_dir, direction, pooling):
    m = load(tiny_dir, direction=direction, pooling=pooling)
    alone = torch.cat([m.encode([t]) for t in TEXTS])
    batched = m.encode(TEXTS, batch_size=4)
    shuffled = m.encode(TEXTS[::-1], batch_size=3).flip(0)
    for other in (batched, shuffled):
        cos = F.cosine_similarity(alone, other, dim=-1)
        assert cos.min() >= 0.9999, (direction, cos)
    assert m.tokenizer.padding_side == "right"


def test_eos_is_appended_even_when_truncated(tiny_dir):
    m = load(tiny_dir)
    enc = m.tokenize(["palavra " * 200], max_length=16)
    assert enc["input_ids"].shape[1] == 16
    assert enc["input_ids"][0, -1].item() == m.tokenizer.convert_tokens_to_ids("</s>")


def _states(m, ids):
    mask = torch.ones_like(ids)
    out = m.hidden_states(ids, mask, output_hidden_states=True)
    return out.hidden_states


@pytest.mark.parametrize("direction", ["C", "Ba", "Bf"])
def test_T2_causality(tiny_dir, direction):
    m = load(tiny_dir, direction=direction)
    ids = m.tokenize([TEXTS[0]])["input_ids"]
    ids2 = ids.clone()
    ids2[0, -2] = (ids2[0, -2] + 7) % len(m.tokenizer)  # troca o último token antes do EOS
    a, b = _states(m, ids), _states(m, ids2)
    prefix = slice(0, ids.shape[1] - 2)
    layer_types = [layer.block_type for layer in m.backbone.layers]
    # hidden_states[i] = saída da camada i-1 (0 = embeddings); a última entrada passa pela norm_f
    changed = [not torch.allclose(x[0, prefix], y[0, prefix], atol=1e-6) for x, y in zip(a, b)]
    if direction == "C":
        assert not any(changed)
    elif direction == "Ba":
        first_attn = layer_types.index("full_attention") + 1
        assert not any(changed[:first_attn]) and changed[first_attn]
    else:  # Bf: muda já na primeira camada Mamba
        assert changed[1]


@pytest.mark.parametrize("pooling", ["eos", "mean"])
def test_T3_pooling_position(tiny_dir, pooling):
    m = load(tiny_dir, pooling=pooling)
    enc = m.tokenize(TEXTS)
    with torch.no_grad():
        h = m.hidden_states(enc["input_ids"], enc["attention_mask"]).last_hidden_state
        got = m(enc["input_ids"], enc["attention_mask"])
    mask = enc["attention_mask"]
    manual = []
    for i in range(len(TEXTS)):
        n = int(mask[i].sum())
        manual.append(h[i, n - 1] if pooling == "eos" else h[i, :n].mean(0))
    manual = F.normalize(torch.stack(manual).float(), dim=-1)
    assert torch.allclose(got, manual, atol=1e-5)


def test_T4_roundtrip_lumi_and_sentence_transformers(tiny_dir, tmp_path):
    m = load(tiny_dir)
    ref = m.encode(TEXTS)
    out = save_embedder(m, tmp_path / "ckpt", meta={"seed": 0}, dtype=torch.float32)
    again = LumiEmbedder.from_pretrained(str(out), dtype=torch.float32, use_hub_kernels=False)
    assert again.spec.direction == "C" and again.spec.pooling == "eos"
    assert torch.allclose(again.encode(TEXTS), ref, atol=1e-5)

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(out), device="cpu", model_kwargs={"dtype": torch.float32})
    st_emb = torch.tensor(st.encode(TEXTS, convert_to_numpy=True, batch_size=4))
    assert F.cosine_similarity(st_emb, ref, dim=-1).min() >= 0.9999
    q = st.encode(["o que é inflação?"], prompt_name="query", convert_to_tensor=True).cpu()
    ours = again.encode(["o que é inflação?"], instruction="Dada uma pergunta, recupere passagens que a respondam")
    assert F.cosine_similarity(q.float(), ours, dim=-1).item() >= 0.9999
