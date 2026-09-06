"""Fit a three-component normal mixture and plot what EM did.

Run with ``python examples/mixture_demo.py`` (add ``--show`` to open a window).
The figure is written next to this script.

Panels, clockwise from top left:
    1. the data, the fitted mixture density, and the weighted components;
    2. AIC/BIC against the number of groups, i.e. how you would pick that number;
    3. the log-likelihood climbing to convergence;
    4. the responsibilities — the soft assignment EM actually produces.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from _style import GRID, HAIRLINE, INK, INK_SOFT, SERIES, SURFACE, colour_by, style_axes
from em import EM

TRUE_COMPONENTS = [(0.45, -4.0, 1.0), (0.35, 1.0, 1.4), (0.20, 6.5, 0.8)]
N_SAMPLES = 6_000
CANDIDATE_GROUPS = range(1, 7)


def make_data(seed: int = 20240905) -> np.ndarray:
    """Draw from a known three-component normal mixture."""
    rng = np.random.default_rng(seed)
    weights = np.array([w for w, _, _ in TRUE_COMPONENTS])
    labels = rng.choice(len(TRUE_COMPONENTS), size=N_SAMPLES, p=weights)
    x = np.empty(N_SAMPLES)
    for k, (_, mu, sigma) in enumerate(TRUE_COMPONENTS):
        mask = labels == k
        x[mask] = rng.normal(mu, sigma, int(mask.sum()))
    return x


def colour_by_position(model: EM) -> dict[int, str]:
    """Map each component to a hue by its position on the x-axis.

    The labels keep the model's own component indices, but the palette runs
    left to right, and every panel shares this mapping so a colour always means
    the same group.
    """
    return colour_by(np.argsort(model.params_[:, 0]))


def plot_fit(ax, x: np.ndarray, model: EM, colours: dict[int, str]) -> None:
    """Data histogram, the fitted mixture, and each weighted component."""
    grid = np.linspace(x.min(), x.max(), 800)
    ax.hist(x, bins=70, density=True, color="#d9d8d4", edgecolor=SURFACE, linewidth=0.4)

    components = np.exp(model.loglike(grid, with_priors=False)) * model.weights_
    mixture = components.sum(axis=1)
    ax.plot(grid, mixture, color=INK, linewidth=2.0, label="fitted mixture")
    for k in np.argsort(model.params_[:, 0]):
        curve = components[:, k]
        ax.plot(grid, curve, color=colours[int(k)], linewidth=2.0, label=f"group {k}")
        peak = grid[np.argmax(curve)]
        # Direct labels: the relief the aqua slot needs, and easier to read anyway.
        ax.annotate(
            f"group {k}\nw={model.weights_[k]:.2f}",
            xy=(peak, curve.max()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_SOFT,
        )

    # Headroom for the direct labels, and a legend clear of the tallest peak.
    ax.set_ylim(top=mixture.max() * 1.32)
    style_axes(ax, "Data and fitted mixture", "x", "density")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT, loc="upper right")


def plot_model_selection(ax, x: np.ndarray, scores: dict[int, tuple[float, float]]) -> None:
    """AIC and BIC share units, so they share one axis."""
    groups = sorted(scores)
    # AIC and BIC nearly coincide, so the dash pattern keeps both readable where
    # colour alone would hide one under the other.
    styles = (("AIC", 0, SERIES[0], (0, (5, 2))), ("BIC", 1, SERIES[1], "solid"))
    for name, index, color, dashes in styles:
        values = [scores[k][index] for k in groups]
        ax.plot(
            groups, values, color=color, linewidth=2.0, linestyle=dashes,
            marker="o", markersize=6, label=name,
        )

    best = min(groups, key=lambda k: scores[k][1])
    ax.annotate(
        f"BIC picks {best} groups",
        xy=(best, scores[best][1]),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=8,
        color=INK_SOFT,
        arrowprops={"arrowstyle": "-", "color": HAIRLINE, "linewidth": 1.0},
    )
    style_axes(ax, "Choosing the number of groups", "number of groups", "criterion (lower is better)")
    ax.set_xticks(groups)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT)


def plot_convergence(ax, model: EM) -> None:
    """A single series: the title names it, so no legend box."""
    history = np.asarray(model.loglike_history_) / model.X_.size
    ax.plot(np.arange(1, history.size + 1), history, color=SERIES[0], linewidth=2.0)
    ax.annotate(
        f"converged in {model.n_iter_} iterations",
        xy=(history.size, history[-1]),
        xytext=(-8, -16),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_SOFT,
    )
    style_axes(ax, "Log-likelihood per iteration", "EM iteration", "mean log-likelihood")


def plot_responsibilities(ax, x: np.ndarray, model: EM, colours: dict[int, str]) -> None:
    """The posterior over groups, which is what EM produces before argmax."""
    grid = np.linspace(x.min(), x.max(), 800)
    resp = model.predict_proba(grid)
    for k in np.argsort(model.params_[:, 0]):
        ax.plot(grid, resp[:, k], color=colours[int(k)], linewidth=2.0, label=f"group {k}")
        peak = grid[np.argmax(resp[:, k])]
        ax.annotate(
            f"group {k}", xy=(peak, 1.0), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=8, color=INK_SOFT,
        )
    ax.set_ylim(-0.05, 1.18)
    style_axes(ax, "Responsibilities P(group | x)", "x", "posterior probability")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure in a window")
    parser.add_argument("--seed", type=int, default=20240905)
    args = parser.parse_args()

    x = make_data(args.seed)

    scores = {}
    for k in CANDIDATE_GROUPS:
        candidate = EM("normal", seed=0).train(x, n_groups=k, n_init=6)
        scores[k] = (candidate.aic(), candidate.bic())

    n_groups = min(scores, key=lambda k: scores[k][1])
    model = EM("normal", seed=0).train(x, n_groups=n_groups, n_init=8)
    print(model.summary())
    print(f"\ntrue components: {TRUE_COMPONENTS}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
    colours = colour_by_position(model)
    plot_fit(axes[0, 0], x, model, colours)
    plot_model_selection(axes[0, 1], x, scores)
    plot_convergence(axes[1, 0], model)
    plot_responsibilities(axes[1, 1], x, model, colours)
    fig.suptitle(
        f"EM on a {len(TRUE_COMPONENTS)}-component normal mixture ({N_SAMPLES:,} samples)",
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
