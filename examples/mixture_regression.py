"""Two regression lines crossing in an X, and why one regression cannot see them.

Run with ``python examples/mixture_regression.py`` (add ``--show`` to open a
window). The figure is written next to this script.

The data is an X: half the points follow ``y = 3x``, half follow ``y = -3x``,
and nothing marks which is which. Ordinary least squares has to answer with a
single line, so it returns the average of the two — a flat line through the
middle that describes no point in the dataset. The mixture fits both arms and
the assignment at the same time.

Panels, clockwise from top left:
    1. one regression through everything, which is what OLS must return;
    2. the two-component fit, recovering both arms;
    3. AIC/BIC against the number of lines;
    4. assignment certainty, which collapses where the arms cross.
"""

from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from _style import GRID, HAIRLINE, INK, INK_SOFT, SEQUENTIAL, SERIES, SURFACE, colour_by, style_axes
from em import EM

# (weight, intercept, slope, sigma) — two arms of an X.
TRUE_LINES = [
    (0.5, 0.0, 3.0, 0.7),
    (0.5, 0.0, -3.0, 0.7),
]
N_SAMPLES = 3_000
X_RANGE = (-3.0, 3.0)
CANDIDATE_GROUPS = range(1, 7)


def make_data(seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the X. Returns (x, y, true label)."""
    rng = np.random.default_rng(seed)
    weights = np.array([w for w, _, _, _ in TRUE_LINES])
    labels = rng.choice(len(TRUE_LINES), size=N_SAMPLES, p=weights)
    x = rng.uniform(*X_RANGE, size=N_SAMPLES)
    y = np.empty(N_SAMPLES)
    for k, (_, intercept, slope, sigma) in enumerate(TRUE_LINES):
        mask = labels == k
        y[mask] = intercept + slope * x[mask] + rng.normal(0, sigma, int(mask.sum()))
    return x, y, labels


def line_label(intercept: float, slope: float, sigma: float) -> str:
    return f"y = {intercept:+.2f} {'+' if slope >= 0 else '-'} {abs(slope):.2f}x\n$\\sigma$={sigma:.2f}"


def plot_single_fit(ax, x, y, single) -> None:
    """What one regression must answer: the average of two opposite slopes."""
    grid = np.linspace(*X_RANGE, 200)
    intercept, slope, sigma = single.params_[0]
    ax.scatter(x, y, s=6, color="#b9b8b3", alpha=0.45, linewidths=0)
    ax.plot(grid, intercept + slope * grid, color=INK, linewidth=2.5)
    ax.annotate(
        line_label(intercept, slope, sigma),
        xy=(grid[-1], intercept + slope * grid[-1]), xytext=(-8, 14),
        textcoords="offset points", ha="right", fontsize=8, color=INK_SOFT,
        bbox={"facecolor": SURFACE, "edgecolor": HAIRLINE, "linewidth": 0.6, "pad": 3},
    )
    ax.annotate(
        "the fitted line passes through\nalmost none of the data",
        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8, color=INK_SOFT,
    )
    style_axes(ax, "One regression through everything (OLS)", "x", "y")


def plot_mixture_fit(ax, x, y, model, colours) -> None:
    """The same data, fitted as a mixture: both arms and the assignment."""
    assignment = model.classify()
    grid = np.linspace(*X_RANGE, 200)
    for k in np.argsort(-model.params_[:, 1]):
        colour = colours[int(k)]
        mask = assignment == k
        ax.scatter(x[mask], y[mask], s=6, color=colour, alpha=0.4, linewidths=0)
        intercept, slope, sigma = model.params_[k]
        ax.plot(grid, intercept + slope * grid, color=colour, linewidth=2.0, label=f"line {k}")
        ax.annotate(
            line_label(intercept, slope, sigma) + f", w={model.weights_[k]:.2f}",
            xy=(grid[-1], intercept + slope * grid[-1]), xytext=(8, 0),
            textcoords="offset points", ha="left", va="center", fontsize=8, color=INK_SOFT,
        )
    span = X_RANGE[1] - X_RANGE[0]
    ax.set_xlim(X_RANGE[0] - 0.05 * span, X_RANGE[1] + 0.45 * span)
    style_axes(ax, "Two regressions, fitted jointly", "x", "y")
    # The arms leave the middle-left empty; the corners are taken by labels.
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT, loc="center left")


def plot_model_selection(ax, scores) -> None:
    groups = sorted(scores)
    styles = (("AIC", 0, SERIES[0], (0, (5, 2))), ("BIC", 1, SERIES[1], "solid"))
    for name, index, colour, dashes in styles:
        ax.plot(
            groups, [scores[k][index] for k in groups], color=colour, linewidth=2.0,
            linestyle=dashes, marker="o", markersize=6, label=name,
        )
    best = min(groups, key=lambda k: scores[k][1])
    ax.annotate(
        f"BIC picks {best} lines",
        xy=(best, scores[best][1]), xytext=(16, 22), textcoords="offset points",
        fontsize=8, color=INK_SOFT,
        arrowprops={"arrowstyle": "-", "color": HAIRLINE, "linewidth": 1.0},
    )
    style_axes(ax, "Choosing the number of lines", "number of lines", "criterion (lower is better)")
    ax.set_xticks(groups)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT)


def plot_certainty(ax, fig, x, y, model) -> None:
    """At the crossing the arms are genuinely indistinguishable."""
    certainty = model.predict_proba().max(axis=1)
    ramp = LinearSegmentedColormap.from_list("blues", SEQUENTIAL)
    dots = ax.scatter(x, y, c=certainty, cmap=ramp, s=6, alpha=0.85, linewidths=0,
                      vmin=1.0 / len(TRUE_LINES), vmax=1.0)
    bar = fig.colorbar(dots, ax=ax, pad=0.02)
    bar.set_label("posterior of the assigned line", fontsize=8, color=INK_SOFT)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)
    bar.outline.set_edgecolor(HAIRLINE)
    ambiguous = (certainty < 0.8).mean()
    ax.annotate(
        f"{ambiguous:.1%} of points are ambiguous\n(posterior < 0.8, at the crossing)",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=8, color=INK_SOFT,
        bbox={"facecolor": SURFACE, "edgecolor": HAIRLINE, "linewidth": 0.6, "pad": 4},
    )
    style_axes(ax, "Assignment certainty", "x", "y")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure in a window")
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    x, y, truth = make_data(args.seed)

    scores = {}
    for k in CANDIDATE_GROUPS:
        candidate = EM("linear-regression", seed=0).train(x, n_groups=k, n_init=8, y=y)
        scores[k] = (candidate.aic(), candidate.bic())

    # n_groups=1 is ordinary least squares: one line, every point weighted equally.
    single = EM("linear-regression", seed=0).train(x, n_groups=1, y=y)
    n_groups = min(scores, key=lambda k: scores[k][1])
    model = EM("linear-regression", seed=0).train(x, n_groups=n_groups, n_init=12, y=y)

    print("single regression (n_groups=1):")
    print(f"   intercept={single.params_[0][0]:+.3f}, slope={single.params_[0][1]:+.3f}, "
          f"sigma={single.params_[0][2]:.3f}")
    print("\n" + model.summary())
    print("\ntrue lines (weight, intercept, slope, sigma):")
    for line in TRUE_LINES:
        print("  ", line)
    print(f"\nresidual spread: {single.params_[0][2]:.2f} for one line, "
          f"{model.params_[:, 2].mean():.2f} for the mixture")

    labels = model.classify()
    agreement = (
        max((labels == np.take(perm, truth)).mean() for perm in itertools.permutations(range(n_groups)))
        if n_groups == len(TRUE_LINES) else float("nan")
    )
    print(f"assignment accuracy: {agreement:.3f}")

    colours = colour_by(np.argsort(-model.params_[:, 1]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
    plot_single_fit(axes[0, 0], x, y, single)
    plot_mixture_fit(axes[0, 1], x, y, model, colours)
    plot_model_selection(axes[1, 0], scores)
    plot_certainty(axes[1, 1], fig, x, y, model)
    fig.suptitle(
        f"Mixture of regressions on X-shaped data ({N_SAMPLES:,} points, labels unknown)",
        fontsize=13, color=INK, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
