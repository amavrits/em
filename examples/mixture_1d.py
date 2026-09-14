"""Fit a normal mixture to the galaxy velocities and plot what EM did.

Run with ``python examples/mixture_1d.py`` (add ``--show`` to open a window).
The figure is written next to this script.

The data are the radial velocities of 82 galaxies in the Corona Borealis
region (Postman, Huchra and Geller 1986; made a mixture-model benchmark by
Roeder 1990 — see ``examples/datasets.py``). Superclusters are separated by
voids, so velocity should arrive in clumps, and *how many* clumps there are is
the scientific question rather than a nuisance parameter. Published answers
range from three to seven, which makes this a more honest test of a mixture
fitter than synthetic data: there is no true answer to recover.

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
from _style import HAIRLINE, INK, INK_SOFT, SERIES, SURFACE, colour_by, style_axes
from datasets import galaxy_velocities
from em import EM

CANDIDATE_GROUPS = range(1, 9)
# 82 observations is few enough that a restart can land in a poor local
# optimum, so every candidate gets a generous number of them.
N_INIT = 24


def colour_by_position(model: EM) -> dict[int, str]:
    """Map each component to a hue by its position on the x-axis.

    The labels keep the model's own component indices, but the palette runs
    left to right, and every panel shares this mapping so a colour always means
    the same group.
    """
    return colour_by(np.argsort(model.params_[:, 0]))


def plot_fit(ax, x: np.ndarray, model: EM, colours: dict[int, str]) -> None:
    """Data, the fitted mixture, and each weighted component.

    With 82 points a histogram is mostly binning artefact, so the rug along the
    bottom carries the raw data and the histogram stays faint behind it.
    """
    grid = np.linspace(x.min() - 1.5, x.max() + 1.5, 800)
    ax.hist(x, bins=24, density=True, color="#e6e5e1", edgecolor=SURFACE, linewidth=0.4)

    components = np.exp(model.loglike(grid, with_priors=False)) * model.weights_
    mixture = components.sum(axis=1)
    # The three components barely overlap, so the mixture sits exactly on top of
    # them and a solid black line would simply hide them. Drawn as a wide soft
    # halo instead, it reads as the envelope the components fill in.
    ax.plot(grid, mixture, color=INK, linewidth=4.0, alpha=0.22, solid_capstyle="round",
            label="fitted mixture")
    for k in np.argsort(model.params_[:, 0]):
        curve = components[:, k]
        ax.plot(grid, curve, color=colours[int(k)], linewidth=1.8, label=f"group {k}")
        peak = grid[np.argmax(curve)]
        # Direct labels: identity never rests on colour alone.
        ax.annotate(
            f"group {k}\nw={model.weights_[k]:.2f}",
            xy=(peak, curve.max()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_SOFT,
        )

    top = mixture.max() * 1.34
    ax.plot(
        x, np.full(x.size, -0.02 * top), marker="|", linestyle="none",
        markersize=7, markeredgewidth=0.9, color=INK_SOFT, alpha=0.8,
    )
    ax.set_ylim(-0.05 * top, top)
    style_axes(ax, "Galaxy velocities and fitted mixture", "velocity (1000 km/s)", "density")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT, loc="upper right")


def plot_model_selection(ax, scores: dict[int, tuple[float, float]], chosen: int) -> None:
    """AIC and BIC share units, so they share one axis."""
    groups = sorted(scores)
    # AIC and BIC separate here, unlike on large samples: n=82 makes the log n
    # penalty roughly twice the AIC one, so the dash pattern is for contrast
    # rather than for prising apart two overlapping lines.
    styles = (("AIC", 0, SERIES[0], (0, (5, 2))), ("BIC", 1, SERIES[1], "solid"))
    for name, index, color, dashes in styles:
        values = [scores[k][index] for k in groups]
        ax.plot(
            groups, values, color=color, linewidth=2.0, linestyle=dashes,
            marker="o", markersize=6, label=name,
        )

    ax.annotate(
        f"BIC picks {chosen} groups",
        xy=(chosen, scores[chosen][1]),
        xytext=(14, 20),
        textcoords="offset points",
        fontsize=8,
        color=INK_SOFT,
        arrowprops={"arrowstyle": "-", "color": HAIRLINE, "linewidth": 1.0},
    )
    # The AIC line keeps descending, and it is not finding structure: past five
    # groups EM starts parking a narrow component on two or three points, which
    # buys likelihood cheaply. Worth saying on the chart, not just in a README.
    ax.annotate(
        "AIC keeps falling: the extra\ngroups are spikes on a few points",
        xy=(groups[-1], scores[groups[-1]][0]),
        xytext=(-6, 26),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_SOFT,
    )
    style_axes(ax, "Choosing the number of groups", "number of groups", "criterion (lower is better)")
    ax.set_xticks(groups)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT, loc="upper right")


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
    grid = np.linspace(x.min() - 1.5, x.max() + 1.5, 800)
    resp = model.predict_proba(grid)
    for k in np.argsort(model.params_[:, 0]):
        ax.plot(grid, resp[:, k], color=colours[int(k)], linewidth=2.0, label=f"group {k}")
        peak = grid[np.argmax(resp[:, k])]
        ax.annotate(
            f"group {k}", xy=(peak, 1.0), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=8, color=INK_SOFT,
        )
    ax.plot(
        x, np.full(x.size, -0.025), marker="|", linestyle="none",
        markersize=7, markeredgewidth=0.9, color=INK_SOFT, alpha=0.8,
    )
    ax.set_ylim(-0.06, 1.18)
    style_axes(ax, "Responsibilities P(group | velocity)", "velocity (1000 km/s)", "posterior probability")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure in a window")
    args = parser.parse_args()

    x = galaxy_velocities()

    scores = {}
    for k in CANDIDATE_GROUPS:
        candidate = EM("normal", seed=0).train(x, n_groups=k, n_init=N_INIT)
        scores[k] = (candidate.aic(), candidate.bic())

    n_groups = min(scores, key=lambda k: scores[k][1])
    model = EM("normal", seed=0).train(x, n_groups=n_groups, n_init=N_INIT)
    print(model.summary())
    print(f"\n{x.size} galaxies, velocities {x.min():.2f}-{x.max():.2f} (1000 km/s)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
    colours = colour_by_position(model)
    plot_fit(axes[0, 0], x, model, colours)
    plot_model_selection(axes[0, 1], scores, n_groups)
    plot_convergence(axes[1, 0], model)
    plot_responsibilities(axes[1, 1], x, model, colours)
    fig.suptitle(
        f"EM on the galaxy velocities ({x.size} galaxies, BIC picks {n_groups} groups)",
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
