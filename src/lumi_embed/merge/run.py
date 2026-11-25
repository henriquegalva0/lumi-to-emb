"""Grade de merges e escolha no dev (seção 4.5 do plano).

1. Conjunto candidato fixado na config (antes de olhar qualquer teste).
2. Grade: média linear (pesos iguais); SLERP em pares com t ∈ {0,25; 0,5; 0,75};
   interpolação da média com a âncora, λ ∈ {0,25; 0,5; 0,75}.
3. Escolha pelo dev (nDCG@10 médio por tarefa; desempate por votos Borda).
4. O escolhido vai para checkpoints/<nome>/model por hard links (sem gastar disco; só o meta.json é
   copiado, porque é reescrito); todos os merges ficam salvos com a receita. As versões descartadas podem
   ser apagadas depois com jobs/embed/prune_merge.py (as notas no dev ficam no merge_report.json).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from ..eval.dev import borda, dev_eval, load_dev_set
from ..utils import EmbedError, ckpt_dir, data_dir, ensure_free_space, run_metadata, write_json
from . import merge as mg

log = logging.getLogger("lumi_embed.merge")


def _model(ckpt: str):
    p = ckpt_dir(ckpt) / "model"
    if not (p / "config.json").exists():
        raise EmbedError(f"Candidato ao merge não encontrado: {p} (treine {ckpt} antes).")
    return p


def _weights_bytes(model_dir: Path) -> int:
    return sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors"))


def _link_tree(src: Path, dst: Path) -> None:
    """Cópia de um checkpoint feita de hard links (não gasta disco). O meta.json é copiado de verdade,
    porque é reescrito depois; os pesos nunca são alterados no lugar (os merges gravam em .tmp e renomeiam)."""
    for f in sorted(Path(src).rglob("*")):
        d = dst / f.relative_to(src)
        if f.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if f.name != "meta.json":
            try:
                os.link(f, d)
                continue
            except OSError:  # outro sistema de arquivos ou sem suporte: copia
                pass
        shutil.copy2(f, d)


def run_merge_grid(cfg: dict) -> dict:
    import torch

    from ..modeling.embedder import LumiEmbedder

    name = cfg["name"]
    cands = [c for c in cfg["candidates"]]
    paths = {c: _model(c) for c in cands}
    plan = [(f"{name}-avg", lambda: mg.linear(list(paths.values()), ckpt_dir(f"{name}-avg") / "model",
                                              recipe={"candidates": cands}))]
    for a, b in cfg.get("slerp_pairs", []):
        for t in cfg.get("slerp_t", []):
            vid = f"{name}-slerp-{a}-{b}-t{t}"
            plan.append((vid, lambda a=a, b=b, t=t, vid=vid: mg.slerp_merge(
                paths[a], paths[b], float(t), ckpt_dir(vid) / "model", recipe={"pair": [a, b]})))
    anchor = cfg.get("anchor")
    if anchor and cfg.get("base_lambdas"):
        for lam in cfg.get("base_lambdas", []):
            vid = f"{name}-base{lam}"
            plan.append((vid, lambda lam=lam, vid=vid: mg.base_interp(
                _model(anchor), ckpt_dir(f"{name}-avg") / "model", float(lam), ckpt_dir(vid) / "model",
                recipe={"anchor": anchor, "of": f"{name}-avg"})))
    # Espaço: cada versão nova grava um modelo inteiro; refazer uma que já existe usa um a mais, de passagem
    # (.tmp). O escolhido entra por hard links e não conta.
    size = _weights_bytes(paths[cands[0]])
    n_new = sum(not (ckpt_dir(vid) / "model" / "config.json").exists() for vid, _ in plan)
    ensure_free_space(ckpt_dir(name).parent, (n_new + 1) * size * 1.02, f"o merge {name} ({len(plan)} versões)")
    variants: dict[str, object] = {vid: build() for vid, build in plan}
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
    _link_tree(Path(variants[best]), out)
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
