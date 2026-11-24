"""Perdas, GradCache, laço de treino (LoRA e completo), retomada, MNTP e merges — no NemotronH minúsculo."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import torch.nn.functional as F  # noqa: E402

from embed_fixtures import build_tiny_backbone_dir, tiny_config  # noqa: E402
from lumi_embed.eval.dev import borda, dev_eval  # noqa: E402
from lumi_embed.merge import merge as mg  # noqa: E402
from lumi_embed.modeling.embedder import LumiEmbedder  # noqa: E402
from lumi_embed.modeling.export import save_embedder  # noqa: E402
from lumi_embed.train.gradcache import gradcache_step  # noqa: E402
from lumi_embed.train.losses import LossConfig, contrastive_loss, info_nce  # noqa: E402

TINY_LORA = {"r": 4, "alpha": 8, "target_modules": ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj",
                                                       "down_proj"]}


def _unit(n, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_hn_loss_matches_manual_with_ragged_negatives():
    q, p, n = _unit(3), _unit(3, seed=1), _unit(4, seed=2)
    owner = torch.tensor([0, 0, 2, 1])  # query 0 tem 2 negativos, 1 e 2 têm 1
    cfg = LossConfig(mode="hn", temperature=0.05)
    got = info_nce(q, p, n, owner, cfg)
    manual = []
    for i in range(3):
        logits = [float(q[i] @ p[i])] + [float(q[i] @ n[m]) for m in range(4) if owner[m] == i]
        lt = torch.tensor(logits) / 0.05
        manual.append(-torch.log_softmax(lt, 0)[0])
    assert torch.allclose(got, torch.stack(manual).mean(), atol=1e-5)


def test_inbatch_false_negative_mask():
    q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    p = F.normalize(torch.tensor([[0.6, 0.8], [0.0, 1.0]]), dim=-1)  # o positivo de q0 é pior que p1 para q0? não
    p_dup = p.clone()
    p_dup[1] = p[0]  # documento repetido: positivo de q1 == positivo de q0
    cfg = LossConfig(mode="inbatch", temperature=1.0, fn_margin=0.1)
    ids = torch.tensor([7, 7])  # mesmo documento
    with_mask = info_nce(q, p_dup, None, torch.empty(0, dtype=torch.long), cfg, pos_ids=ids)
    no_dup_ids = torch.tensor([7, 8])
    without = info_nce(q, p_dup, None, torch.empty(0, dtype=torch.long), cfg, pos_ids=no_dup_ids)
    assert with_mask < without  # a cópia do positivo não é tratada como negativo


def test_mrl_averages_dimensions():
    q, p = _unit(4, 16), _unit(4, 16, seed=3)
    base = contrastive_loss(q, p, None, torch.empty(0, dtype=torch.long), LossConfig(mode="inbatch"))
    mrl = contrastive_loss(q, p, None, torch.empty(0, dtype=torch.long), LossConfig(mode="inbatch", mrl_dims=[4, 8]))
    assert torch.isfinite(mrl) and not torch.allclose(base, mrl)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, tiny_tok_dir):
    return build_tiny_backbone_dir(tmp_path_factory.mktemp("tiny_train"), tiny_tok_dir)


def test_gradcache_equals_full_batch_backward(tiny_dir):
    texts_q = ["a cidade de rio", "o museu de arte foi", "uma universidade antiga", "estrada de ferro"]
    texts_p = ["rio grande e porto", "museu importante", "escola e universidade", "ferrovia da capital"]
    cfg = LossConfig(mode="inbatch", temperature=0.05)

    def run(use_gc):
        m = LumiEmbedder.from_pretrained(str(tiny_dir), dtype=torch.float32, use_hub_kernels=False)
        m.train()
        encs = [m.tokenize(texts_q), m.tokenize(texts_p)]
        owner = torch.empty(0, dtype=torch.long)
        if use_gc:
            gradcache_step(m, encs + [None], lambda lv: contrastive_loss(lv[0], lv[1], None, owner, cfg), chunk=1)
        else:
            q = m(encs[0]["input_ids"], encs[0]["attention_mask"])
            p = m(encs[1]["input_ids"], encs[1]["attention_mask"])
            contrastive_loss(q, p, None, owner, cfg).backward()
        return {n: prm.grad.clone() for n, prm in m.backbone.named_parameters() if prm.grad is not None}

    a, b = run(True), run(False)
    assert a.keys() == b.keys() and len(a) > 10
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5, rtol=1e-3), k


def _write_set(path, n, seed=0):
    import random

    words = "a cidade rio porto museu arte escola governo estado capital ferrovia guerra paz povo".split()
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        q = " ".join(rng.choice(words) for _ in range(4))
        rows.append({"id": f"x{i}", "task": "retrieval" if i % 3 else "sts", "lang": "pt-BR",
                     "instruction": "Dada uma pergunta, recupere passagens que a respondam",
                     "query": q, "positive": q + " " + " ".join(rng.choice(words) for _ in range(6)),
                     "negatives": [{"text": " ".join(rng.choice(words) for _ in range(8)), "score": 0.3}
                                   for _ in range(4)],
                     "split": "train"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return rows


def test_hn_micro_batch_does_not_change_gradient(tiny_dir):
    # Com a perda hn, cada query só concorre com o próprio positivo e os próprios negativos: o micro-lote
    # muda a memória, não o gradiente (é o que permite micro-lote 1 na fase de 8k do S-long).
    from lumi_embed.modeling.embedder import EmbedSpec
    from lumi_embed.train.contrastive import _step, example_texts

    rows_raw = _write_set(tiny_dir.parent / "hn_micro" / "x.jsonl", 6, seed=11)
    spec = EmbedSpec.from_dict({"direction": "C", "pooling": "eos", "max_length": 32})
    rows = [example_texts(r, spec, 2) for r in rows_raw]
    m = LumiEmbedder.from_pretrained(str(tiny_dir), dtype=torch.float32, use_hub_kernels=False)
    grads = {}
    for micro in (1, 2, 6):
        m.zero_grad(set_to_none=True)
        loss = _step(m, rows, LossConfig.from_dict({"mode": "hn", "temperature": 0.05}), micro, spec,
                     torch.device("cpu"))
        grads[micro] = (loss, [p.grad.clone() for p in m.parameters() if p.grad is not None])
    for micro in (2, 6):
        assert abs(grads[micro][0] - grads[1][0]) < 1e-5
        assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(grads[micro][1], grads[1][1]))


@pytest.mark.parametrize("method,mode", [("lora", "hn"), ("lora", "inbatch"), ("full", "hn")])
def test_train_loop_saves_loadable_embedder(tiny_dir, tmp_path, monkeypatch, method, mode):
    from lumi_embed.train.contrastive import train

    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    _write_set(tmp_path / "data/embed/sets/t/train.jsonl", 24)
    dev_rows = _write_set(tmp_path / "data/embed/sets/t/dev.jsonl", 9, seed=5)
    cfg = {
        "seed": 3, "init": str(tiny_dir),
        "model": {"use_hub_kernels": False},
        "embedding": {"direction": "C", "pooling": "eos", "max_length": 32},
        "data": {"train": "sets/t/train.jsonl", "dev": "sets/t/dev.jsonl", "negatives_per_example": 2},
        "loss": {"mode": mode, "temperature": 0.05},
        "train": {"method": method, "lr": 1e-3, "batch_size": 8, "micro_batch": 4, "epochs": 1,
                  "warmup_steps": 1, "eval_every": 2, "lora": TINY_LORA, "gradient_checkpointing": False},
    }
    out = train(cfg, f"T-{method}-{mode}", dev_eval_fn=dev_eval)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["seed"] == 3 and meta["steps"] == 3 and "dev" in meta and meta["git"]["commit"]
    m = LumiEmbedder.from_pretrained(str(out), dtype=torch.float32, use_hub_kernels=False)
    res = dev_eval(m, dev_rows)
    assert 0 <= res["mean_ndcg@10"] <= 1 and set(res["tasks"]) == {"retrieval", "sts"}
    log = [json.loads(x) for x in (out.parent / "train_log.jsonl").read_text().splitlines()]
    assert len(log) == 3 and all(torch.isfinite(torch.tensor(r["loss"])) for r in log)
    base = LumiEmbedder.from_pretrained(str(tiny_dir), dtype=torch.float32, use_hub_kernels=False)
    assert not torch.allclose(base.encode(["a cidade"]), m.encode(["a cidade"]), atol=1e-4)  # os pesos mudaram


def test_train_resumes_from_saved_state(tiny_dir, tmp_path, monkeypatch):
    from lumi_embed.train.contrastive import train

    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    _write_set(tmp_path / "data/embed/sets/t/train.jsonl", 32)
    cfg = {"seed": 1, "init": str(tiny_dir), "model": {"use_hub_kernels": False},
           "embedding": {"max_length": 32}, "data": {"train": "sets/t/train.jsonl", "negatives_per_example": 1},
           "loss": {"mode": "hn"}, "train": {"method": "lora", "lr": 1e-3, "batch_size": 8, "micro_batch": 8,
                                             "epochs": 1, "warmup_steps": 1, "lora": TINY_LORA,
                                             "gradient_checkpointing": False}}
    out = train(cfg, "T-resume", max_minutes=0.0)  # salva o estado depois do 1º passo e sai
    assert (out / "train_state" / "state.json").exists() and not (out / "model").exists()
    final = train(cfg, "T-resume")
    assert json.loads((final / "meta.json").read_text())["steps"] == 4
    assert len((out / "train_log.jsonl").read_text().splitlines()) == 4


def test_mntp_exports_bidirectional_embedder(tiny_tok_dir, tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lumi_embed.train.mntp import train_mntp

    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    tok = AutoTokenizer.from_pretrained(str(tiny_tok_dir))
    lm = AutoModelForCausalLM.from_config(tiny_config(len(tok)), dtype=torch.float32)
    lm.save_pretrained(tmp_path / "lm")
    tok.save_pretrained(tmp_path / "lm")
    (tmp_path / "data/embed/sets/s0").mkdir(parents=True)
    (tmp_path / "data/embed/sets/s0/train.jsonl").write_text(
        "\n".join(json.dumps({"text": "a cidade de rio grande foi capital do estado durante a guerra"}) for _ in range(8)))
    cfg = {"seed": 0, "init": str(tmp_path / "lm"), "model": {"use_hub_kernels": False},
           "embedding": {"direction": "Ba", "pooling": "mean", "max_length": 32},
           "data": {"train": "sets/s0/train.jsonl"}, "mntp": {"mask_ratio": 0.3, "mask_token": "<unk>"},
           "train": {"lr": 1e-3, "batch_size": 4, "max_steps": 2, "lora": TINY_LORA}}
    out = train_mntp(cfg, "B0a-test")
    m = LumiEmbedder.from_pretrained(str(out), dtype=torch.float32, use_hub_kernels=False)
    assert m.spec.direction == "Ba" and m.spec.pooling == "mean"
    assert m.encode(["a cidade"]).shape == (1, 64)


def test_merges(tiny_tok_dir, tmp_path):
    dirs = []
    for s in range(3):
        d = build_tiny_backbone_dir(tmp_path / f"raw{s}", tiny_tok_dir, seed=s)
        m = LumiEmbedder.from_pretrained(str(d), dtype=torch.float32, use_hub_kernels=False)
        dirs.append(save_embedder(m, tmp_path / f"c{s}", dtype=torch.float32))
    from safetensors.torch import load_file

    w = [load_file(str(d / "model.safetensors")) for d in dirs]
    avg = mg.linear(dirs, tmp_path / "avg")
    got = load_file(str(avg / "model.safetensors"))
    k = "layers.0.mixer.in_proj.weight"
    assert torch.allclose(got[k].float(), ((w[0][k] + w[1][k] + w[2][k]) / 3).bfloat16().float())
    s0 = load_file(str(mg.slerp_merge(dirs[0], dirs[1], 0.0, tmp_path / "s0") / "model.safetensors"))
    assert torch.allclose(s0[k].float(), w[0][k].bfloat16().float())
    b0 = load_file(str(mg.base_interp(dirs[0], dirs[1], 1.0, tmp_path / "b1") / "model.safetensors"))
    assert torch.allclose(b0[k].float(), w[1][k].bfloat16().float())
    m = LumiEmbedder.from_pretrained(str(avg), dtype=torch.float32, use_hub_kernels=False)
    assert m.encode(["a cidade"]).shape == (1, 64)
    assert json.loads((avg / "meta.json").read_text())["method"] == "linear"


def test_slerp_tensor_properties():
    a, b = torch.randn(10, 10), torch.randn(10, 10)
    assert torch.allclose(mg.slerp(0.0, a, b), a, atol=1e-5)
    assert torch.allclose(mg.slerp(1.0, a, b), b, atol=1e-5)
    assert torch.allclose(mg.slerp(0.5, a, a), a, atol=1e-5)  # colineares: cai no linear


def test_borda():
    res = {"A": {"tasks": {"r": {"ndcg@10": 0.5}, "s": {"ndcg@10": 0.9}}},
           "B": {"tasks": {"r": {"ndcg@10": 0.6}, "s": {"ndcg@10": 0.8}}},
           "C": {"tasks": {"r": {"ndcg@10": 0.4}, "s": {"ndcg@10": 0.7}}}}
    assert borda(res) == {"A": 3.0, "B": 3.0, "C": 0.0}


def test_warmup_is_proportional_to_training_length():
    from lumi_embed.train.contrastive import lr_lambda, warmup_steps

    assert warmup_steps({"warmup_ratio": 0.1}, 195) == 20      # S1 na escala reduzida
    assert warmup_steps({"warmup_ratio": 0.1}, 24) == 3        # varredura de LR
    assert warmup_steps({}, 937) == 94                         # padrão 10 %
    assert warmup_steps({"warmup_steps": 50, "warmup_ratio": 0.1}, 195) == 50  # explícito vence
    f = lr_lambda(3, 24)
    assert f(2) == 1.0 and f(0) < f(1) < f(2)                   # a taxa chega ao valor testado


def test_dev_distractors_fixed_and_enlarge_the_pool(tiny_dir, tmp_path):
    # O dev ganha distratores fixos (positivos do treino, nunca do dev), determinísticos e por tarefa.
    from lumi_embed.eval.dev import DISTRACTORS_FILE, dev_distractors, load_dev_set

    sets = tmp_path / "sets"
    train = _write_set(sets / "S1" / "train.jsonl", 60, seed=1)
    _write_set(sets / "S2" / "train.jsonl", 30, seed=2)
    dev_rows = _write_set(sets / "S1" / "dev.jsonl", 9, seed=5)
    # um positivo do dev também aparece no treino: não pode virar distrator
    (sets / "S1" / "train.jsonl").write_text((sets / "S1" / "train.jsonl").read_text()
                                             + json.dumps({**train[0], "id": "dup", "positive": dev_rows[0]["positive"]}) + "\n")
    d = dev_distractors(sets, {"retrieval": 15, "sts": 5})
    assert len(d["retrieval"]) == 15 and len(d["sts"]) == 5
    assert dev_rows[0]["positive"] not in d["retrieval"] + d["sts"]
    first = (sets / DISTRACTORS_FILE).read_text()
    (sets / DISTRACTORS_FILE).unlink()
    dev_distractors(sets, {"retrieval": 15, "sts": 5})
    assert (sets / DISTRACTORS_FILE).read_text() == first          # determinístico

    m = LumiEmbedder.from_pretrained(str(tiny_dir), dtype=torch.float32, use_hub_kernels=False)
    plain = dev_eval(m, dev_rows)
    hard = dev_eval(m, load_dev_set(sets / "S1" / "dev.jsonl"))
    for task in plain["tasks"]:
        assert hard["tasks"][task]["pool"] > plain["tasks"][task]["pool"]


def test_long_dev_has_own_task_distractors_and_length(tiny_dir, tmp_path):
    # Dev do estágio longo: tarefa retrieval_long, distratores longos próprios (positivos longos do treino,
    # sem o replay curto), documentos codificados com até 8k tokens; o dev curto não muda.
    from lumi_embed.eval import dev as devmod

    sets = tmp_path / "sets"
    _write_set(sets / "S2" / "train.jsonl", 30, seed=2)
    short_dev = _write_set(sets / "S2" / "dev.jsonl", 6, seed=5)
    longs = _write_set(sets / "long" / "train.jsonl", 20, seed=7)
    rows = [{**r, "source": {"long": i % 4 != 0}} for i, r in enumerate(longs)]   # 1 em 4 = replay curto
    (sets / "long" / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ldev = _write_set(sets / "long" / "dev.jsonl", 4, seed=9)
    (sets / "long" / "dev.jsonl").write_text("\n".join(json.dumps({**r, "task": devmod.LONG_TASK}) for r in ldev) + "\n")
    d = devmod.load_dev_set([sets / "S2" / "dev.jsonl", sets / "long" / "dev.jsonl"])
    assert len(d.rows) == len(short_dev) + len(ldev)
    longd = d.distractors[devmod.LONG_TASK]
    assert set(longd) <= {r["positive"] for r in rows if r["source"]["long"]} and len(longd) == 15
    assert (sets / devmod.LONG_DISTRACTORS_FILE).exists() and (sets / devmod.DISTRACTORS_FILE).exists()

    m = LumiEmbedder.from_pretrained(str(tiny_dir), dtype=torch.float32, use_hub_kernels=False)
    seen = []
    enc = m.encode
    m.encode = lambda texts, **kw: (seen.append(kw.get("max_length")), enc(texts, **kw))[1]
    res = dev_eval(m, d)
    assert devmod.LONG_TASK in res["tasks"] and "retrieval" in res["tasks"]
    assert devmod.LONG_MAX_LENGTH in seen


def test_long_stage_then_merge_with_short_and_long_dev(tiny_dir, tmp_path, monkeypatch):
    # Bloco de contexto longo em miniatura: treino com currículo de comprimento e dev curto + longo a partir
    # de um checkpoint (ckpt:lumi-emb), depois o merge M-L (dev em lista, sem âncora) e o spec do longo.
    from lumi_embed.eval import dev as devmod
    from lumi_embed.merge.run import run_merge_grid
    from lumi_embed.train.contrastive import train
    from lumi_embed.utils import ckpt_dir

    monkeypatch.setenv("LUMI_WORKDIR", str(tmp_path))
    sets = tmp_path / "data/embed/sets"
    _write_set(sets / "S2" / "train.jsonl", 20, seed=2)
    _write_set(sets / "S2" / "dev.jsonl", 6, seed=5)
    rows = _write_set(sets / "long" / "train.jsonl", 16, seed=7)
    (sets / "long" / "train.jsonl").write_text("\n".join(
        json.dumps({**r, "source": {"long": True}, "positive": r["positive"] * 6}) for r in rows) + "\n")
    ldev = _write_set(sets / "long" / "dev.jsonl", 4, seed=9)
    (sets / "long" / "dev.jsonl").write_text("\n".join(
        json.dumps({**r, "task": devmod.LONG_TASK, "positive": r["positive"] * 6}) for r in ldev) + "\n")
    base = LumiEmbedder.from_pretrained(str(tiny_dir), dtype=torch.float32, use_hub_kernels=False)
    save_embedder(base, ckpt_dir("lumi-emb") / "model", meta={"seed": 0}, dtype=torch.float32)

    cfg = {
        "seed": 3, "init": "ckpt:lumi-emb", "model": {"use_hub_kernels": False},
        "embedding": {"direction": "C", "pooling": "eos", "max_length": 8192},
        "data": {"train": "sets/long/train.jsonl", "dev": ["sets/S2/dev.jsonl", "sets/long/dev.jsonl"],
                 "negatives_per_example": 2},
        "loss": {"mode": "hn", "temperature": 0.05},
        "train": {"method": "lora", "lr": 1e-3, "batch_size": 8, "micro_batch": 2, "epochs": 1, "warmup_steps": 1,
                  "eval_every": 1, "lora": TINY_LORA, "gradient_checkpointing": False,
                  "length_schedule": [[0.0, 16], [0.5, 64, 1]]},
    }
    import lumi_embed.train.contrastive as ct

    phases, real_step = [], ct._step
    monkeypatch.setattr(ct, "_step", lambda m, rows, lc, micro, spec, dev: (
        phases.append((spec.max_length, micro)), real_step(m, rows, lc, micro, spec, dev))[1])
    out = train(cfg, "C3-long", dev_eval_fn=dev_eval)
    assert phases == [(16, 2), (64, 1)]   # 2 passos; o terceiro item da fase troca o micro-lote
    log = [json.loads(x) for x in (out.parent / "train_log.jsonl").read_text().splitlines()]
    assert all(devmod.LONG_TASK in r["dev"]["tasks"] and "retrieval" in r["dev"]["tasks"] for r in log)
    assert json.loads((out / "lumi_embed.json").read_text())["max_length"] == 8192

    rep = run_merge_grid({"name": "M-L", "candidates": ["C3-long", "lumi-emb"],
                          "slerp_pairs": [["C3-long", "lumi-emb"]], "slerp_t": [0.5],
                          "dev": ["sets/S2/dev.jsonl", "sets/long/dev.jsonl"]})
    assert set(rep["dev"]) == {"C3-long", "lumi-emb", "M-L-avg", "M-L-slerp-C3-long-lumi-emb-t0.5"}
    assert all(devmod.LONG_TASK in r["tasks"] for r in rep["dev"].values())
    spec = json.loads((ckpt_dir("M-L") / "model" / "lumi_embed.json").read_text())
    assert spec["max_length"] == 8192   # o spec do resultado vem do primeiro candidato (o longo)
