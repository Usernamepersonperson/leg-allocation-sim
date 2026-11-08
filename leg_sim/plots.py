"""Figures: docs/figure.png (1600x900) and docs/post-figure.png (960x700, dark, phone-sized)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from leg_sim.billing import AllocationResult  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"
METHOD_COLOR = {"static": AMBER, "dynamic": ACCENT, "fair": GREEN}
METHOD_LABEL = {"static": "static shares", "dynamic": "dynamic (by consumption)", "fair": "fair (weighted max-min)"}


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"


def _style(ax, size: float) -> None:
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=size)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _week(res: AllocationResult, start: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    sl = slice(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=7) - pd.Timedelta(minutes=15))
    load = res.loads.loc[sl].sum(axis=1).to_numpy() * 4 / 1000  # Wh per quarter -> kW
    used = res.alloc.loc[sl].sum(axis=1).to_numpy() * 4 / 1000
    pv = res.production.loc[sl].to_numpy() * 4 / 1000
    return res.loads.loc[sl].index, load, used, pv


def _draw_week(ax, res: AllocationResult, start: str, size: float, lw: float) -> None:
    idx, load, used, pv = _week(res, start)
    ax.stackplot(idx, used, load - used, colors=[GREEN, GREY], labels=["served by community PV", "from grid"], alpha=0.95)
    ax.plot(idx, pv, color=AMBER, lw=lw, label="PV production")
    ax.set_ylabel("kW", color=MUTED, fontsize=size)
    ax.set_xlim(idx[0], idx[-1])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a"))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    _style(ax, size)


def _legend(ax, size: float, **kw) -> None:
    leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=size, **kw)
    leg.get_frame().set_alpha(0.9)


def main_figure(results: dict[str, AllocationResult], metrics: dict[str, dict], household_savings: dict[str, np.ndarray],
                path: Path, week_start: str, source_line: str) -> None:
    plt.rcParams["font.family"] = _font()
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.18, left=0.06, right=0.98, top=0.85, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, :])
    _draw_week(ax1, results["dynamic"], week_start, 12, 2.0)
    _legend(ax1, 12, loc="upper left", ncol=3)
    ax1.set_title("One summer week: community consumption vs PV (dynamic allocation)", color=TEXT, fontsize=14, loc="left")

    ax2 = fig.add_subplot(gs[1, 0])
    names = list(metrics)
    x = np.arange(len(names))
    sc = [metrics[m]["self_consumption_pct"] for m in names]
    ss = [metrics[m]["self_sufficiency_pct"] for m in names]
    ax2.bar(x - 0.2, sc, 0.38, color=ACCENT, label="self-consumption (of PV)")
    ax2.bar(x + 0.2, ss, 0.38, color=GREEN, label="self-sufficiency (of load)")
    for i, (a, b) in enumerate(zip(sc, ss, strict=True)):
        ax2.text(i - 0.2, a + 1, f"{a:.1f}", ha="center", color=TEXT, fontsize=11)
        ax2.text(i + 0.2, b + 1, f"{b:.1f}", ha="center", color=TEXT, fontsize=11)
    ax2.set_xticks(x, [METHOD_LABEL[m] for m in names])
    ax2.set_ylim(0, max(sc) * 1.2)
    ax2.set_ylabel("%", color=MUTED, fontsize=12)
    _style(ax2, 12)
    _legend(ax2, 11, loc="upper left")
    ax2.set_title("Allocation method: community energy shares", color=TEXT, fontsize=14, loc="left")

    ax3 = fig.add_subplot(gs[1, 1])
    data = [household_savings[m] for m in names]
    bp = ax3.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color=TEXT, lw=2),
                     whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED), flierprops=dict(markeredgecolor=MUTED))  # fmt: skip
    for patch, m in zip(bp["boxes"], names, strict=True):
        patch.set(facecolor=METHOD_COLOR[m], alpha=0.85, edgecolor=TEXT)
    ax3.set_xticks(range(1, len(names) + 1), [METHOD_LABEL[m] for m in names])
    ax3.set_ylabel("CHF per household and year", color=MUTED, fontsize=12)
    ax3.axhline(0, color=GREY, lw=1)
    _style(ax3, 12)
    ax3.set_title("Saving vs no community, 26 households", color=TEXT, fontsize=14, loc="left")

    fig.text(0.06, 0.945, "Local electricity community: who gets the solar, and what it saves", color=TEXT, fontsize=22, fontweight="bold")
    fig.text(0.06, 0.905, source_line, color=MUTED, fontsize=12)  # fmt: skip
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def post_figure(res: AllocationResult, hh_saving_chf: np.ndarray, path: Path, week_start: str, source_line: str,
                self_consumption_pct: float) -> None:
    plt.rcParams["font.family"] = _font()
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.55, left=0.09, right=0.97, top=0.80, bottom=0.09)
    ax1 = fig.add_subplot(gs[0])
    _draw_week(ax1, res, week_start, 11, 2.4)
    _legend(ax1, 11, loc="upper left", ncol=3, columnspacing=0.9, handlelength=1.2)
    ax1.set_title("Summer week: the building PV serves the neighbours", color=TEXT, fontsize=13, loc="left")
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.22)

    ax2 = fig.add_subplot(gs[1])
    s = np.sort(hh_saving_chf)[::-1]
    cols = [GREEN if v >= 0 else RED for v in s]
    ax2.bar(np.arange(len(s)), s, color=cols, width=0.8)
    ax2.axhline(0, color=GREY, lw=1.2)
    ax2.set_xlim(-0.8, len(s) - 0.2)
    ax2.set_xticks([])
    ax2.set_ylabel("CHF / year", color=MUTED, fontsize=11)
    ax2.set_xlabel(f"26 households, sorted | median CHF {np.median(hh_saving_chf):.0f}", color=MUTED, fontsize=11)
    _style(ax2, 11)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Yearly saving per household vs no community", color=TEXT, fontsize=13, loc="left")

    fig.text(0.04, 0.935, f"Neighbour solar sharing: {self_consumption_pct:.0f}% of PV used locally", color=TEXT, fontsize=19, fontweight="bold")
    fig.text(0.04, 0.885, source_line, color=MUTED, fontsize=12)  # fmt: skip
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
