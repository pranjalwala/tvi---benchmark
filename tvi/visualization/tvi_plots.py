"""
tvi/visualization/tvi_plots.py
--------------------------------
All TVI benchmark plots.

Functions
---------
plot_tvi_curves()           — full TVI(a_nom) curves per method/channel with CI bands
plot_tvi_scalar()           — bar chart of TVI at 50% nominal (primary scalar)
plot_transfer_functions()   — a_nom → a_print curves per method/channel
plot_simulation_comparison()— side-by-side original vs simulated halftone images
save_figure()               — consistent DPI/format save helper

LIBRARY: matplotlib only (open-source)
All other imports are from the tvi package.

Style notes
-----------
* Figures use a clean publication style (no gridlines by default).
* Colours cycle through a colorblind-friendly palette.
* CI bands are shaded, not error bars, to avoid clutter on dense curves.
* All labels, titles and axis ranges are derived from data — nothing hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for scripts/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tvi.aggregation.statistics import AggregatedResult
from tvi.calibration.transfer_function import TransferFunction


# Colorblind-safe palette (Wong 2011)
_PALETTE = [
    "#0072B2",   # blue
    "#E69F00",   # orange
    "#009E73",   # green
    "#CC79A7",   # pink
    "#56B4E9",   # sky blue
    "#F0E442",   # yellow
    "#D55E00",   # vermillion
    "#000000",   # black
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_tvi_curves(
    results: list[AggregatedResult],
    title: str = "TVI Curves",
    show_ci: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Plot full TVI(a_nom) curves for each (method, channel) with CI shading.

    Parameters
    ----------
    results  : list of AggregatedResult (one per method/channel combination)
    title    : figure title
    show_ci  : whether to shade 95% CI bands
    ax       : existing Axes to draw on (None → create new figure)
    figsize  : figure size in inches

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = _get_axes(ax, figsize)

    for i, r in enumerate(results):
        color = _PALETTE[i % len(_PALETTE)]
        label = _result_label(r)
        x = r.a_nom * 100          # percent
        y = r.mean_tvi * 100       # percentage points

        ax.plot(x, y, color=color, linewidth=1.8, label=label)

        if show_ci and np.any(r.ci_tvi > 0):
            ci = r.ci_tvi * 100
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.18)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Nominal Coverage (%)", fontsize=11)
    ax.set_ylabel("TVI (pp)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    _clean_spines(ax)
    fig.tight_layout()
    return fig


def plot_tvi_scalar(
    results: list[AggregatedResult],
    title: str = "TVI at 50% Nominal Coverage",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Grouped bar chart of the primary TVI scalar (TVI at a_nom = 0.50).

    One group per halftoning method; bars coloured by channel.

    Parameters
    ----------
    results : list of AggregatedResult
    title   : figure title
    ax      : existing Axes (None → new figure)
    figsize : figure size

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = _get_axes(ax, figsize)

    methods = _unique_ordered([r.method for r in results])
    channels = _unique_ordered([r.channel for r in results])

    n_methods = len(methods)
    n_channels = len(channels)
    group_width = 0.7
    bar_width = group_width / max(n_channels, 1)
    x_base = np.arange(n_methods)

    for ci, ch in enumerate(channels):
        color = _PALETTE[ci % len(_PALETTE)]
        vals, errs, xs = [], [], []
        for mi, method in enumerate(methods):
            matched = [r for r in results if r.method == method and r.channel == ch]
            if matched:
                r = matched[0]
                vals.append(r.tvi_at_50 * 100)
                errs.append(r.ci_at_50 * 100)
                xs.append(x_base[mi] + (ci - n_channels / 2.0 + 0.5) * bar_width)

        if xs:
            ax.bar(xs, vals, width=bar_width * 0.85, color=color, label=ch,
                   yerr=errs, capsize=3, error_kw={"linewidth": 1.0})

    ax.set_xticks(x_base)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("TVI at 50% (pp)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(title="Channel", fontsize=8, framealpha=0.8)
    _clean_spines(ax)
    fig.tight_layout()
    return fig


def plot_transfer_functions(
    transfer_functions: list[TransferFunction],
    title: str = "Printer Transfer Functions",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
) -> plt.Figure:
    """
    Plot a_nom → a_print for each (method, channel) transfer function.

    The ideal (identity) line is always drawn for reference.

    Parameters
    ----------
    transfer_functions : list of TransferFunction
    title  : figure title
    ax     : existing Axes (None → new figure)
    figsize: figure size

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = _get_axes(ax, figsize)

    a_grid = np.linspace(0.0, 1.0, 200)

    # Identity reference
    ax.plot(a_grid * 100, a_grid * 100, color="gray", linewidth=1.0,
            linestyle=":", label="Ideal (no gain)")

    for i, tf in enumerate(transfer_functions):
        color = _PALETTE[i % len(_PALETTE)]
        label = f"{tf.method} / {tf.channel}" if tf.method else tf.channel
        a_print = tf.evaluate(a_grid)
        ax.plot(a_grid * 100, a_print * 100, color=color,
                linewidth=1.8, label=label)
        if not tf.is_monotone:
            ax.annotate(
                "⚠ non-monotone",
                xy=(50, float(tf.evaluate(0.5)) * 100),
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Nominal Coverage (%)", fontsize=11)
    ax.set_ylabel("Printed Coverage (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, framealpha=0.8)
    _clean_spines(ax)
    fig.tight_layout()
    return fig


def plot_simulation_comparison(
    original: np.ndarray,
    simulated: np.ndarray,
    title: str = "Original vs Simulated",
    channel_label: str = "",
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Side-by-side comparison: original binary halftone vs simulated output.

    Adds a difference panel showing regions of dot growth.

    Parameters
    ----------
    original  : 2-D uint8 or float halftone (H, W)
    simulated : 2-D uint8 or float simulated image (same shape)
    title     : figure title
    channel_label : ink channel label for annotation
    figsize   : figure size

    Returns
    -------
    matplotlib Figure
    """
    orig_f = _to_float(original)
    sim_f = _to_float(simulated)
    diff = sim_f - orig_f   # positive = darkened by dot gain

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(orig_f, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original halftone", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(sim_f, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Simulated (after dot gain)", fontsize=10)
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Difference (sim − orig)", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    sup = title
    if channel_label:
        sup += f"  [{channel_label}]"
    fig.suptitle(sup, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 150,
    fmt: str = "png",
) -> Path:
    """
    Save a matplotlib Figure to disk.

    Parameters
    ----------
    fig  : matplotlib Figure
    path : output file path (extension overrides fmt if present)
    dpi  : output resolution
    fmt  : file format ("png", "pdf", "svg")

    Returns
    -------
    Path of saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_axes(
    ax: plt.Axes | None,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _clean_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _result_label(r: AggregatedResult) -> str:
    parts = []
    if r.method:
        parts.append(r.method)
    if r.channel:
        parts.append(r.channel)
    if r.printer_label:
        parts.append(f"[{r.printer_label}]")
    return " / ".join(parts) if parts else "result"


def _unique_ordered(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _to_float(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)
