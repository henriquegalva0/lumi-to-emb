"""Figuras do README e do model card, desenhadas só a partir das notas que o relatório já carregou.

Mesma regra das tabelas: nenhum número é digitado aqui; cada figura recebe os valores prontos do
`report` e só é desenhada se eles existirem. Cada uma sai em duas versões (fundo claro e escuro), em
docs/embed/figures/<nome>-<tema>.png. Cores: azul = este trabalho, laranja e verde-água = referências
(paleta categórica validada para daltonismo, nesta ordem), cinza = contexto.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

log = logging.getLogger("lumi_embed.figures")

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "s1": "#2a78d6", "s1b": "#86b6ef", "s2": "#eb6834", "s3": "#1baf7a",
              "band": "#f0efec"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "s1": "#3987e5", "s1b": "#1c5cab", "s2": "#d95926", "s3": "#199e70",
             "band": "#262624"},
}
DPI = 200


def _setup(plt, th: dict, w: float, h: float, ncols: int = 1):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.edgecolor": th["axis"], "axes.labelcolor": th["ink2"], "xtick.color": th["ink2"],
                         "ytick.color": th["ink2"], "text.color": th["ink"]})
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=th["surface"], squeeze=False)
    for ax in axes[0]:
        ax.set_facecolor(th["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(th["axis"])
        ax.spines["bottom"].set_color(th["axis"])
        ax.tick_params(length=0)
    return fig, axes[0]


def _title(fig, th: dict, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=th["ink"])
    fig.text(0.012, 0.905, subtitle, ha="left", va="top", fontsize=9.5, color=th["ink2"])


def _save(fig, plt, path: Path, th: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=th["surface"])
    plt.close(fig)
    return path


def _legend(ax, th: dict, handles, labels, **kw):
    leg = ax.legend(handles, labels, frameon=False, fontsize=9, labelcolor=th["ink2"], handlelength=1.6, **kw)
    return leg


def _dot(ax, x, y, color, th, size=8, z=4):
    ax.plot([x], [y], "o", ms=size, color=color, mec=th["surface"], mew=2, zorder=z)


# ---------------------------------------------------------------- figuras
def mldr_slope(plt, th: dict, data: list[tuple[str, str, float, float]]):
    """MLDR-pt em 512 tokens contra o documento inteiro (até 8k). data: [(rótulo, papel, v512, v8k)]."""
    fig, (ax,) = _setup(plt, th, 8.0, 4.6)
    fig.subplots_adjust(left=0.08, right=0.70, top=0.72, bottom=0.12)
    _title(fig, th, "Documentos longos em português (MLDR-pt)",
           "nDCG@10 lendo só os primeiros 512 tokens de cada documento, e lendo o documento inteiro (até 8k)")
    lo = min(min(v1, v2) for _, _, v1, v2 in data)
    hi = max(max(v1, v2) for _, _, v1, v2 in data)
    ax.set_ylim(math.floor((lo - 5) / 10) * 10, math.ceil((hi + 3) / 10) * 10)
    ax.set_xlim(-0.12, 1.12)
    ax.set_xticks([0, 1], ["512 tokens", "até 8k tokens"])
    ax.yaxis.grid(True, color=th["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    handles, labels = [], []
    for label, role, v1, v2 in data:
        c = th[role]
        (ln,) = ax.plot([0, 1], [v1, v2], lw=2.6 if role == "s1" else 2, color=c, solid_capstyle="round", zorder=3)
        _dot(ax, 0, v1, c, th)
        _dot(ax, 1, v2, c, th)
        weight = "bold" if role == "s1" else "normal"
        ax.annotate(f"{label}  {v2:.1f}", (1, v2), xytext=(12, 0), textcoords="offset points", va="center",
                    fontsize=10, color=th["ink"], fontweight=weight, annotation_clip=False)
        if role == "s1":
            ax.annotate(f"{v1:.1f}", (0, v1), xytext=(-12, 0), textcoords="offset points", va="center",
                        ha="right", fontsize=10, color=th["ink"], fontweight=weight, annotation_clip=False)
        handles.append(ln)
        labels.append(label)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 0.86), ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=th["ink2"], handlelength=1.6)
    return fig


def length_curves(plt, th: dict, data: dict[str, dict[str, dict[int, float]]], roles: dict[str, str],
                  trained: int):
    """Passkey e Needle (LongEmbed) por comprimento. data: {tarefa: {rótulo: {comprimento: nota}}}."""
    fig, axes = _setup(plt, th, 9.0, 4.6, ncols=len(data))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.68, bottom=0.12, wspace=0.12)
    _title(fig, th, "Encontrar um trecho escondido em textos longos (LongEmbed)",
           f"Acerto (nDCG@10) por comprimento do documento; o estágio longo treinou até {trained // 1024}k tokens")
    handles, labels = {}, []
    for ax, (task, series) in zip(axes, data.items()):
        lengths = sorted({n for s in series.values() for n in s})
        xs = {n: math.log2(n) for n in lengths}
        ax.axvspan(math.log2(trained) + 0.5, math.log2(max(lengths)) + 0.4, color=th["band"], lw=0, zorder=0)
        ax.text(math.log2(trained) + 0.55, 1.015, "além do treino", fontsize=8, color=th["muted"], va="bottom",
                transform=ax.get_xaxis_transform())
        for label, s in series.items():
            c = th[roles[label]]
            pts = sorted(s.items())
            (ln,) = ax.plot([xs[n] for n, _ in pts], [v for _, v in pts], lw=2.6 if roles[label] == "s1" else 2,
                            color=c, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            for n, v in pts:
                ax.plot([xs[n]], [v], "o", ms=5, color=c, mec=th["surface"], mew=1.5, zorder=4)
            if label not in handles:
                handles[label] = ln
                labels.append(label)
        ax.set_xticks([xs[n] for n in lengths], [f"{n // 1024}k" if n >= 1024 else str(n) for n in lengths])
        ax.set_xlim(min(xs.values()) - 0.4, max(xs.values()) + 0.4)
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.grid(True, color=th["grid"], lw=1)
        ax.set_axisbelow(True)
        ax.set_title(task, loc="left", fontsize=10.5, color=th["ink"], pad=6, fontweight="bold")
        ax.spines["bottom"].set_visible(False)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
        ax.spines["left"].set_visible(False)
    fig.legend([handles[k] for k in labels], labels, loc="upper left", bbox_to_anchor=(0.06, 0.86), ncol=len(labels),
               frameon=False, fontsize=9, labelcolor=th["ink2"], handlelength=1.6)
    return fig


def category_dots(plt, th: dict, cats: list[str], series: list[tuple[str, str, list[float]]]):
    """Média por categoria do MTEB-BR (pontos), uma linha por categoria. series: [(rótulo, papel, notas)]."""
    h_in = 0.55 * len(cats) + 1.9
    fig, (ax,) = _setup(plt, th, 8.0, h_in)
    fig.subplots_adjust(left=0.22, right=0.97, top=1 - 1.3 / h_in, bottom=0.5 / h_in)
    _title(fig, th, "MTEB-BR por categoria",
           "Média das tarefas de cada categoria; categorias ordenadas pela vantagem sobre o Qwen3-Embedding-4B")
    ys = list(range(len(cats)))[::-1]
    allv = [v for _, _, vs in series for v in vs]
    # Cada modelo numa faixa da linha (o nosso no meio): valores quase iguais não se escondem.
    offs = {0: 0.0, 1: -0.24, 2: 0.24}
    for i, y in enumerate(ys):
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=th["band"], lw=0, zorder=0)
    handles = []
    for k, (label, role, vs) in enumerate(series):
        for y, v in zip(ys, vs):
            _dot(ax, v, y + offs.get(k, 0), th[role], th, size=8.5, z=5 if role == "s1" else 4)
        handles.append(ax.plot([], [], "o", ms=8, color=th[role], mec=th["surface"], mew=2)[0])
    for y, v in zip(ys, series[0][2]):
        ax.annotate(f"{v:.1f}", (v, y), xytext=(9, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=th["ink"], fontweight="bold")
    ax.set_yticks(ys, cats)
    ax.tick_params(axis="y", labelcolor=th["ink"], labelsize=9.5)
    ax.set_xlim(math.floor((min(allv) - 4) / 10) * 10, math.ceil((max(allv) + 3) / 10) * 10)
    ax.set_ylim(-0.5, len(cats) - 0.5)
    ax.xaxis.grid(True, color=th["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    _legend(ax, th, handles, [s[0] for s in series], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
            borderaxespad=0.3)
    return fig


def ranking_bars(plt, th: dict, rows: list[tuple[str, str, float]], title: str, subtitle: str):
    """Barras horizontais ordenadas. rows: [(rótulo, papel, nota)]; papel s1/s1b = este trabalho, muted = outros."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    rows = sorted(rows, key=lambda r: -r[2])
    h_in = 0.38 * len(rows) + 1.5
    fig, (ax,) = _setup(plt, th, 8.0, h_in)
    fig.subplots_adjust(left=0.30, right=0.93, top=1 - 1.2 / h_in, bottom=0.35 / h_in)
    _title(fig, th, title, subtitle)
    xmax = math.ceil(max(v for *_, v in rows) / 10) * 10 + 5
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ys = list(range(len(rows)))[::-1]
    fig.canvas.draw()
    px_x = ax.bbox.width / xmax
    px_y = ax.bbox.height / len(rows)
    bar_h = min(20 / px_y, 0.62)  # <= 24 px de espessura
    r = 4 / px_x
    for y, (label, role, v) in zip(ys, rows):
        c = th[role]
        ax.add_patch(FancyBboxPatch((0, y - bar_h / 2), v, bar_h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    mutation_aspect=px_x / px_y, lw=0, color=c, zorder=3))
        ax.add_patch(Rectangle((0, y - bar_h / 2), min(v, 2 * r), bar_h, lw=0, color=c, zorder=3))
        bold = "bold" if role == "s1" else "normal"
        ax.annotate(f"{v:.2f}", (v, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5,
                    color=th["ink"], fontweight=bold)
    ax.set_yticks(ys, [lab for lab, *_ in rows])
    for tick, (_, role, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(th["ink"] if role in ("s1", "s1b") else th["ink2"])
        if role == "s1":
            tick.set_fontweight("bold")
    ax.xaxis.grid(True, color=th["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    return fig


# ---------------------------------------------------------------- ponto de entrada
def render(spec: dict, out_dir: Path) -> list[Path]:
    """spec: {nome: (função, kwargs)} com os dados já prontos. Devolve os PNGs gravados."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib ausente: figuras não geradas (as tabelas saem normalmente).")
        return []
    funcs = {"mldr_slope": mldr_slope, "length_curves": length_curves, "category_dots": category_dots,
             "ranking_bars": ranking_bars}
    out = []
    for name, (fn, kw) in spec.items():
        for theme, th in THEMES.items():
            fig = funcs[fn](plt, th, **kw)
            out.append(_save(fig, plt, Path(out_dir) / f"{name}-{theme}.png", th))
    return out
