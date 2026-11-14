"""Shared figure style. Copy this file into each project as ``src/<pkg>/plotstyle.py``
and copy ``_shared/fonts`` to ``<repo>/assets/fonts``.

Usage
-----
    from <pkg>.plotstyle import apply_style, save, SERIES, INK, MUTED
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, y, color=SERIES[0], label="Model")
    title(ax, "Heston fits the 3-month smile to within 0.4 vol points",
          "SPY options, 13 Nov 2026 close. Implied volatility by strike.")
    save(fig, "smile-fit-3m")

Rules the style cannot enforce (follow them):
  * The title states the finding; the subtitle says what is plotted and the units.
  * One y-axis per plot. Two measures on different scales -> two panels.
  * Two or more series -> a legend (``legend(ax)``). One series -> no legend.
  * Assign SERIES colours in order (0, 1, 2 ...) and keep one colour per entity
    across every figure in the project.
  * Label sparingly: an endpoint or an extreme, never every point.
  * Magnitude heatmaps use SEQ_CMAP; signed quantities use DIV_CMAP centred on 0.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# --- tokens (shared with the website's stylesheet) ---------------------------
SURFACE = "#fcfcfb"
INK = "#0f1b33"        # primary text
INK_2 = "#52514e"      # secondary text
MUTED = "#898781"      # axis ticks, de-emphasised series
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HIGHLIGHT = "#ffe566"  # annotation underlay only, never a series

# categorical, fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a3", "#c93a3a"])

FIGSIZE = (7.2, 4.2)        # single panel
FIGSIZE_WIDE = (10.0, 4.2)  # two panels side by side
FIGSIZE_TALL = (7.2, 6.4)   # two stacked panels

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _register_fonts() -> str:
    """Use the bundled Fira Sans if present, else fall back to matplotlib's default."""
    font_dir = _REPO_ROOT / "assets" / "fonts"
    found = False
    if font_dir.is_dir():
        for ttf in font_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(str(ttf))
            found = True
    return "Fira Sans" if found else "DejaVu Sans"


def apply_style() -> None:
    family = _register_fonts()
    mpl.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "figure.dpi": 100,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9.5,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "patch.linewidth": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    })


def title(ax, headline: str, subtitle: str | None = None) -> None:
    """Left-aligned finding as the title, with an optional muted subtitle beneath."""
    if subtitle:
        ax.set_title(headline, pad=22)
        ax.text(0, 1.0, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color=INK_2, linespacing=1.3,
                bbox=dict(facecolor="none", edgecolor="none", pad=4))
    else:
        ax.set_title(headline, pad=10)


def legend(ax, **kw):
    kw.setdefault("loc", "best")
    kw.setdefault("handlelength", 1.6)
    return ax.legend(**kw)


def end_label(ax, x, y, text: str, dx: float = 4, color: str = INK_2) -> None:
    """Direct label at a line's endpoint. Text wears ink, not the series colour."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=color, annotation_clip=False)


def save(fig, name: str, directory: str | Path = "figures") -> Path:
    out = _REPO_ROOT / directory
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
