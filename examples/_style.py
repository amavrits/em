"""Shared chart styling for the examples.

The three series colours are categorical slots validated for colourblind
separation against a light surface (worst all-pairs CVD dE 9.2, normal-vision
24.0), so they stay distinguishable in the scatter plots where every pair of
series is adjacent.
"""

from __future__ import annotations

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e2"
SURFACE = "#fcfcfb"
HAIRLINE = "#c9c8c4"


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid and spines, so the data carries the ink."""
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)


def colour_by(order) -> dict[int, str]:
    """Map component indices to hues in the given display order."""
    return {int(k): SERIES[i % len(SERIES)] for i, k in enumerate(order)}
