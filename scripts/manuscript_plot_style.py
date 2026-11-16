"""Shared plotting style for figures embedded in the manuscript."""

import os
from pathlib import Path

import matplotlib.pyplot as plt


TRUE_COLOR = "#1F5A94"
ESTIMATE_COLOR = "#C4473A"
SECONDARY_COLOR = "#E69F00"
TERTIARY_COLOR = "#009E73"
REFERENCE_COLOR = "#666666"
GRID_COLOR = "#D9D9D9"

FULL_WIDTH = 7.15
THREE_PANEL_SIZE = (FULL_WIDTH, 2.45)
TWO_PANEL_SIZE = (FULL_WIDTH, 2.85)


def apply_manuscript_style():
    """Match the manuscript fonts, with a TeX-free smoke-test fallback."""
    use_tex = os.environ.get("WMFKI_PLOT_USETEX", "1") != "0"
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "text.latex.preamble": (
                r"\usepackage[T1]{fontenc}\usepackage{times}" if use_tex else ""
            ),
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.8,
            "axes.prop_cycle": plt.cycler(
                color=[TRUE_COLOR, ESTIMATE_COLOR, SECONDARY_COLOR, TERTIARY_COLOR]
            ),
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "legend.fontsize": 9.0,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": GRID_COLOR,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def finish_axes(ax):
    ax.spines["top"].set_color(REFERENCE_COLOR)
    ax.spines["right"].set_color(REFERENCE_COLOR)
    ax.spines["bottom"].set_color(REFERENCE_COLOR)
    ax.spines["left"].set_color(REFERENCE_COLOR)


def save_pdf(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
