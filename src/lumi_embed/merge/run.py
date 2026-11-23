"""Grade de merges e escolha no dev (seção 4.5 do plano).

1. Conjunto candidato fixado na config (antes de olhar qualquer teste).
2. Grade: média linear (pesos iguais); SLERP em pares com t ∈ {0,25; 0,5; 0,75};
   interpolação da média com a âncora, λ ∈ {0,25; 0,5; 0,75}.
3. Escolha pelo dev (nDCG@10 médio por tarefa; desempate por votos Borda).
4. O escolhido é copiado para checkpoints/<nome>/model; todos os merges ficam salvos com a receita.
"""

from __future__ import annotations

import json
import logging
import shutil

from ..eval.dev import borda, dev_eval, load_dev_set
from ..utils import EmbedError, ckpt_dir, data_dir, run_metadata, write_json
from . import merge as mg

log = logging.getLogger("lumi_embed.merge")


def _model(ckpt: str):
    p = ckpt_dir(ckpt) / "model"
    if not (p / "config.json").exists():
        raise EmbedError(f"Candidato ao merge não encontrado: {p} (treine {ckpt} antes).")
    return p


def run_merge_grid(cfg: dict) -> dict:
    import torch

    from ..modeling.embedder import LumiEmbedder

    name = cfg["name"]
    cands = [c for c in cfg["candidates"]]
    paths = {c: _model(c) for c in cands}
    variants: dict[str, object] = {}
    variants[f"{name}-avg"] = mg.linear(list(paths.values()), ckpt_dir(f"{name}-avg") / "model",
                                       recipe={"candidates": cands})
    for a, b in cfg.get("slerp_pairs", []):
        for t in cfg.get("slerp_t", []):
            vid = f"{name}-slerp-{a}-{b}-t{t}"
            variants[vid] = mg.slerp_merge(paths[a], paths[b], float(t), ckpt_dir(vid) / "model",
                                           recipe={"pair": [a, b]})
    anchor = cfg.get("anchor")
    if anchor and cfg.get("base_lambdas"):
        for lam in cfg.get("base_lambdas", []):
            vid = f"{name}-base{lam}"
            variants[vid] = mg.base_interp(_model(anchor), ckpt_dir(f"{name}-avg") / "model", float(lam),
                                           ckpt_dir(vid) / "model", recipe={"anchor": anchor, "of": f"{name}-avg"})
    dv = cfg["dev"]
    dev_rows = load_dev_set([data_dir(x) for x in (dv if isinstance(dv, list) else [dv])])
    scored = {}
    for vid, path in {**{c: p for c, p in paths.items()}, **variants}.items():
        m = LumiEmbedder.from_pretrained(str(path), device="cuda" if torch.cuda.is_available() else "cpu")
        scored[vid] = dev_eval(m, dev_rows)
        log.info("dev %s: %s", vid, scored[vid].get("mean_ndcg@10"))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    votes = borda(scored)
    best_individual = max(cands, key=lambda c: (scored[c]["mean_ndcg@10"], votes[c]))
    best = max(variants, key=lambda v: (scored[v]["mean_ndcg@10"], votes[v]))
    out = ckpt_dir(name) / "model"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(variants[best], out)
    meta = json.loads((out / "meta.json").read_text())
    meta.update(run_metadata(cfg, selected=best, selected_by="dev mean nDCG@10 (desempate Borda)"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    report = {"name": name, "selected": best, "best_individual": best_individual,
              "merge_beats_best_individual_on_dev": scored[best]["mean_ndcg@10"] >= scored[best_individual]["mean_ndcg@10"],
              "dev": scored, "borda": votes}
    write_json(ckpt_dir(name) / "merge_report.json", report)
    write_json(ckpt_dir(name) / "summary.json", {"dev": scored[best], "selected": best})
    log.info("%s = %s (dev %.4f; melhor individual %s %.4f)", name, best, scored[best]["mean_ndcg@10"],
             best_individual, scored[best_individual]["mean_ndcg@10"])
    return report
