"""Figuras (matplotlib, PNG estático).

Cores: paleta categórica de referência (azul = 1º modelo, laranja = 2º); a tarefa
é codificada por marcador e tracejado, para a identidade nunca depender só da cor.
Heatmap: rampa sequencial de um único matiz (azul), com o valor escrito na célula.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TASK_STYLE = {"S": ("o", "-"), "MK": ("s", "--"), "MV": ("^", ":")}
TASK_NAME = {"S": "S (1 agulha)", "MK": "MK (multi-chave)", "MV": "MV (multi-valor)"}


def fmt_len(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if n >= 1000:
        return f"{n / 1024:.0f}k"
    return str(n)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def heatmap_s(cells: dict, lengths: list[int], depths: list[float], title: str, out: Path) -> None:
    """cells[(depth, length)] = acurácia estrita média."""
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * len(lengths), 0.9 + 0.42 * len(depths)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    grid = [[cells.get((d, L), math.nan) for L in lengths] for d in depths]
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, d in enumerate(depths):
        for j, L in enumerate(lengths):
            v = grid[i][j]
            label = "–" if math.isnan(v) else f"{v:.2f}"
            color = "white" if (not math.isnan(v) and v >= 0.55) else TEXT
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(lengths)), [fmt_len(L) for L in lengths])
    ax.set_yticks(range(len(depths)), [f"{round(d * 100)}%" for d in depths])
    ax.set_xlabel("Comprimento do prompt (tokens)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("Profundidade da agulha", color=TEXT_2, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # fronteira de 2px entre células
    ax.set_xticks([x - 0.5 for x in range(1, len(lengths))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(depths))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Acurácia estrita", color=TEXT_2, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_2, labelsize=8)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _setup_len_axis(ax, lengths: list[int]):
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, [fmt_len(L) for L in lengths])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])


def acc_vs_len(curves: list[dict], lengths: list[int], threshold: float, title: str, out: Path) -> None:
    """curves: [{"model_label", "model_idx", "task", "x", "y", "lo", "hi"}]. Um único gráfico."""
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    _setup_len_axis(ax, lengths)
    ax.axhline(threshold, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(lengths[0], threshold + 0.012, f"limiar {threshold:.2f}", color=TEXT_2, fontsize=8, va="bottom")
    for c in curves:
        color = SERIES[c["model_idx"] % len(SERIES)]
        marker, ls = TASK_STYLE[c["task"]]
        ax.fill_between(c["x"], c["lo"], c["hi"], color=color, alpha=0.10, linewidth=0)
        ax.plot(c["x"], c["y"], color=color, linestyle=ls, marker=marker, markersize=6, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=f"{c['model_label']} · {TASK_NAME[c['task']]}")
    ax.set_xlabel("Comprimento do prompt (tokens, escala log2)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("Acurácia estrita (média, IC 95%)", color=TEXT_2, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def acc_vs_len_panels(curves: list[dict], lengths: list[int], threshold: float, out: Path) -> None:
    """Mesma informação em pequenos múltiplos (um painel por tarefa)."""
    tasks = [t for t in ("S", "MK", "MV") if any(c["task"] == t for c in curves)]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = list(axes) if len(tasks) > 1 else [axes]
    for ax, task in zip(axes, tasks):
        _style(ax)
        _setup_len_axis(ax, lengths)
        ax.axhline(threshold, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        for c in [c for c in curves if c["task"] == task]:
            color = SERIES[c["model_idx"] % len(SERIES)]
            marker, _ = TASK_STYLE[task]
            ax.fill_between(c["x"], c["lo"], c["hi"], color=color, alpha=0.12, linewidth=0)
            ax.plot(c["x"], c["y"], color=color, marker=marker, markersize=5, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=c["model_label"])
        ax.set_title(TASK_NAME[task], color=TEXT, fontsize=10, loc="left")
    axes[0].set_ylabel("Acurácia estrita", color=TEXT_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(labels), loc="lower center", labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def perf_plot(series: list[dict], lengths: list[int], out: Path) -> None:
    """series: [{"model_label", "model_idx", "x", "prefill_s"}]."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lengths, [fmt_len(L) for L in lengths])
    ax.minorticks_off()
    for s in series:
        ax.plot(s["x"], s["prefill_s"], color=SERIES[s["model_idx"] % len(SERIES)], marker="o", markersize=5,
                linewidth=2, markeredgecolor=SURFACE, label=s["model_label"])
    ax.set_xlabel("Comprimento do prompt (tokens)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("Prefill isolado (s, log)", color=TEXT_2, fontsize=9)
    ax.set_title("Latência de prefill (lote de 1)", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
