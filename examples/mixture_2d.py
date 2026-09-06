"""Fit a three-component mixture of correlated 2-D Gaussians.

Run with ``python examples/mixture_2d.py`` (add ``--show`` to open a window).
The figure is written next to this script.

The 1-D example separates components along a line; here each component has its
own orientation and spread, so what EM recovers is a full covariance per group,
not just a width.

Panels, clockwise from top left:
    1. the points, coloured by fitted component, with 2-sigma covariance ellipses;
    2. AIC/BIC against the number of groups;
    3. the log-likelihood climbing to convergence;
    4. the fitted mixture density over the plane.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from _style import HAIRLINE, INK, INK_SOFT, SEQUENTIAL, SERIES, SURFACE, colour_by, style_axes
from em import EM

# (weight, mean, covariance)
TRUE_COMPONENTS = [
    (0.40, [-2.5, 1.0], [[1.0, 0.75], [0.75, 1.0]]),
    (0.35, [3.0, 2.5], [[2.2, -1.3], [-1.3, 1.4]]),
    (0.25, [0.5, -3.5], [[0.6, 0.0], [0.0, 2.4]]),
]
N_SAMPLES = 4_000
CANDIDATE_GROUPS = range(1, 7)


def make_data(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Draw from a known 2-D mixture. Returns (points, true label)."""
    rng = np.random.default_rng(seed)
    weights = np.array([w for w, _, _ in TRUE_COMPONENTS])
    labels = rng.choice(len(TRUE_COMPONENTS), size=N_SAMPLES, p=weights)
    points = np.empty((N_SAMPLES, 2))
    for k, (_, mean, cov) in enumerate(TRUE_COMPONENTS):
        mask = labels == k
        points[mask] = rng.multivariate_normal(mean, cov, size=int(mask.sum()))
    return points, labels


def unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat parameter vector -> (mean, covariance). Only the triangle is stored."""
    c11, c21, c22 = p[2:]
    return np.asarray(p[:2]), np.array([[c11, c21], [c21, c22]])


def plot_components(ax, points, model, colours) -> None:
    """Points coloured by fitted component, with 2-sigma covariance ellipses."""
    assignment = model.classify()
    for k in np.argsort(model.params_[:, 0]):
        colour = colours[int(k)]
        mask = assignment == k
        ax.scatter(points[mask, 0], points[mask, 1], s=6, color=colour, alpha=0.3, linewidths=0)

        mean, cov = unpack(model.params_[k])
        spread, axes_ = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(axes_[1, -1], axes_[0, -1]))
        ax.add_patch(
            Ellipse(
                mean, *(2 * 2.0 * np.sqrt(spread[::-1])), angle=angle,
                facecolor="none", edgecolor=colour, linewidth=2.0, label=f"group {k}",
            )
        )
        # Direct labels, so identity never rests on colour alone.
        ax.annotate(
            f"group {k}\nw={model.weights_[k]:.2f}",
            xy=mean, xytext=(0, 0), textcoords="offset points", ha="center", va="center",
            fontsize=8, color=INK,
            bbox={"facecolor": SURFACE, "edgecolor": HAIRLINE, "linewidth": 0.6, "pad": 3},
        )
    style_axes(ax, "Points and fitted components (2$\\sigma$)", "$x_1$", "$x_2$")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT, loc="upper left")


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
        f"BIC picks {best} groups",
        xy=(best, scores[best][1]), xytext=(14, 20), textcoords="offset points",
        fontsize=8, color=INK_SOFT,
        arrowprops={"arrowstyle": "-", "color": HAIRLINE, "linewidth": 1.0},
    )
    style_axes(ax, "Choosing the number of groups", "number of groups", "criterion (lower is better)")
    ax.set_xticks(groups)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT)


def plot_convergence(ax, model) -> None:
    history = np.asarray(model.loglike_history_) / model.X_.shape[0]
    ax.plot(np.arange(1, history.size + 1), history, color=SERIES[0], linewidth=2.0)
    ax.annotate(
        f"converged in {model.n_iter_} iterations",
        xy=(history.size, history[-1]), xytext=(-8, -16), textcoords="offset points",
        ha="right", fontsize=8, color=INK_SOFT,
    )
    style_axes(ax, "Log-likelihood per iteration", "EM iteration", "mean log-likelihood")


def plot_density(ax, fig, points, model) -> None:
    """The fitted mixture density over the plane: one magnitude, one hue ramp."""
    pad = 1.0
    grid_x = np.linspace(points[:, 0].min() - pad, points[:, 0].max() + pad, 300)
    grid_y = np.linspace(points[:, 1].min() - pad, points[:, 1].max() + pad, 300)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    flat = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    density = np.exp(model.loglike(flat)).reshape(mesh_x.shape)

    ramp = LinearSegmentedColormap.from_list("blues", ["#ffffff"] + SEQUENTIAL)
    filled = ax.contourf(mesh_x, mesh_y, density, levels=14, cmap=ramp)
    bar = fig.colorbar(filled, ax=ax, pad=0.02)
    bar.set_label("fitted density", fontsize=8, color=INK_SOFT)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)
    bar.outline.set_edgecolor(HAIRLINE)
    for k in range(model.weights_.size):
        mean, _ = unpack(model.params_[k])
        ax.plot(*mean, marker="x", markersize=8, markeredgewidth=2.0, color=INK)
    style_axes(ax, "Fitted mixture density", "$x_1$", "$x_2$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure in a window")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    points, truth = make_data(args.seed)

    scores = {}
    for k in CANDIDATE_GROUPS:
        candidate = EM("multivariate-normal", seed=0).train(points, n_groups=k, n_init=6)
        scores[k] = (candidate.aic(), candidate.bic())

    n_groups = min(scores, key=lambda k: scores[k][1])
    model = EM("multivariate-normal", seed=0).train(points, n_groups=n_groups, n_init=8)
    print(model.summary())
    print("\ntrue components (weight, mean, covariance):")
    for weight, mean, cov in TRUE_COMPONENTS:
        print(f"   {weight}, {mean}, {cov}")

    colours = colour_by(np.argsort(model.params_[:, 0]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
    plot_components(axes[0, 0], points, model, colours)
    plot_model_selection(axes[0, 1], scores)
    plot_convergence(axes[1, 0], model)
    plot_density(axes[1, 1], fig, points, model)
    fig.suptitle(
        f"EM on a {len(TRUE_COMPONENTS)}-component 2-D normal mixture ({N_SAMPLES:,} points)",
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
