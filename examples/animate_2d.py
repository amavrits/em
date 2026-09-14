"""Animate EM converging on the Old Faithful eruptions.

Run with ``python examples/animate_2d.py``; writes ``em_old_faithful.gif``
next to this script. ``--mp4`` writes an MP4 instead (needs ffmpeg), and
``--show`` opens a window.

This is the top-left panel of ``mixture_2d.py`` — the points with their 2-sigma
covariance ellipses — drawn once per EM iteration instead of once at the end.
Two things are worth watching:

* **the colours**, which are the responsibilities rather than hard labels. Each
  point is blended between the two group hues by its posterior, so while the
  fit is still moving the points it cannot yet place stay muddy, and they only
  saturate as the components pull apart. That soft assignment *is* the E-step;
  the hard labels people usually plot are an argmax taken afterwards.
* **the ellipses**, which start as two nearly identical blobs covering the whole
  dataset and end on the two regimes of the geyser.

The start is deliberately uninformative — both components at the sample mean
with the pooled covariance, nudged a hair apart to break the symmetry — because
that is the case where there is something to watch. The library's own
initialization (an ordered split along the first principal direction) already
lands most of the way there and converges in a dozen iterations. Either way EM
reaches the same optimum, to six decimals: mean log-likelihood -4.155382.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse

from _style import HAIRLINE, INK, INK_SOFT, SERIES, SURFACE, style_axes
from datasets import old_faithful
from em import EM

N_GROUPS = 2
TOL = 1e-8
REG = 1e-6
MAX_ITERS = 200
# Frames per second, and how long to hold the converged fit before looping.
FPS = 5
HOLD_SECONDS = 2.0
# Symmetry-breaking nudge, in (minutes, minutes). Two identical components sit
# on a saddle — every responsibility is exactly 0.5 and EM crawls — so they are
# offset before the first E-step. Small relative to the spread of the data, so
# they still start visibly on top of each other; large enough that the run does
# not open with fifteen frames of nothing moving.
NUDGE = np.array([0.4, 5.0])


def pack(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """(mean, covariance) -> the flat vector the family stores."""
    return np.array([mean[0], mean[1], cov[0, 0], cov[1, 0], cov[1, 1]])


def unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat parameter vector -> (mean, covariance)."""
    c11, c21, c22 = p[2:]
    return np.asarray(p[:2]), np.array([[c11, c21], [c21, c22]])


def run_em(points: np.ndarray) -> list[dict]:
    """Replay EM one iteration at a time, recording the state of each.

    ``e_step`` and ``m_step`` are public for exactly this reason, so nothing
    here reaches into the model's internals. Component 0 is nudged toward
    shorter eruptions so it stays the left-hand group for the whole run — the
    labels must not swap halfway through an animation.
    """
    model = EM("multivariate-normal", seed=0, tol=TOL, reg=REG)
    model.X_, model.n_features_ = points, points.shape[1]

    centre, pooled = points.mean(axis=0), np.cov(points.T)
    params = np.array([pack(centre - NUDGE, pooled), pack(centre + NUDGE, pooled)])
    weights = np.full(N_GROUPS, 1.0 / N_GROUPS)

    frames: list[dict] = []
    previous = -np.inf
    for n_iter in range(1, MAX_ITERS + 1):
        log_resp, loglike = model.e_step(points, weights, params)
        frames.append({
            "iter": n_iter,
            "resp": np.exp(log_resp),
            "weights": weights.copy(),
            "params": params.copy(),
            "mean_loglike": loglike / points.shape[0],
        })
        if abs(loglike - previous) / points.shape[0] <= TOL:
            break
        previous = loglike
        weights, params = model.m_step(points, log_resp, params)
    return frames


def blend(resp: np.ndarray) -> np.ndarray:
    """Per-point colour: the two group hues mixed by the posterior.

    A hard label would throw away the only thing EM knows that k-means does
    not. Points the model is unsure about come out muddy, which is the honest
    picture of an overlapping region.
    """
    hues = np.array([to_rgb(SERIES[0]), to_rgb(SERIES[1])])
    return np.clip(resp @ hues, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mp4", action="store_true", help="write an MP4 (needs ffmpeg)")
    parser.add_argument("--show", action="store_true", help="open a window")
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    points = old_faithful()
    frames = run_em(points)
    print(f"{len(frames)} iterations, mean loglike {frames[-1]['mean_loglike']:.6f}")

    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    scatter = ax.scatter(points[:, 0], points[:, 1], s=18, linewidths=0, alpha=0.75)
    ellipses = [
        Ellipse((0, 0), 1, 1, facecolor="none", edgecolor=SERIES[k], linewidth=2.2)
        for k in range(N_GROUPS)
    ]
    for patch in ellipses:
        ax.add_patch(patch)
    # Direct labels ride on the component means. The two offsets point in
    # opposite directions so the boxes stay legible in the first few frames,
    # where both components sit on top of each other at the sample mean.
    labels = [
        ax.annotate(
            "", xy=(0, 0), xytext=(0, -16 if k == 0 else 16), textcoords="offset points",
            ha="center", va="center", fontsize=8, color=INK,
            bbox={"facecolor": SURFACE, "edgecolor": HAIRLINE, "linewidth": 0.6, "pad": 3},
        )
        for k in range(N_GROUPS)
    ]
    readout = ax.text(
        0.985, 0.04, "", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color=INK_SOFT, family="monospace",
    )

    # Limits fixed across every frame: an axis that rescales as the fit moves
    # makes the fit look like it is standing still.
    ax.set_xlim(points[:, 0].min() - 0.45, points[:, 0].max() + 0.45)
    ax.set_ylim(points[:, 1].min() - 6.0, points[:, 1].max() + 6.0)
    style_axes(
        ax, "EM fitting the Old Faithful eruptions",
        "eruption duration (min)", "waiting time to next eruption (min)",
    )
    fig.tight_layout()

    hold = max(1, int(round(args.fps * HOLD_SECONDS)))
    order = list(range(len(frames))) + [len(frames) - 1] * hold

    def draw(index: int):
        frame = frames[index]
        scatter.set_facecolor(blend(frame["resp"]))
        for k in range(N_GROUPS):
            mean, cov = unpack(frame["params"][k])
            spread, axes_ = np.linalg.eigh(cov)
            ellipses[k].set_center(mean)
            ellipses[k].set_width(2 * 2.0 * np.sqrt(spread[1]))
            ellipses[k].set_height(2 * 2.0 * np.sqrt(spread[0]))
            ellipses[k].set_angle(np.degrees(np.arctan2(axes_[1, -1], axes_[0, -1])))

            labels[k].xy = tuple(mean)
            labels[k].set_text(f"group {k + 1}\nw={frame['weights'][k]:.2f}")
        settled = index == len(frames) - 1
        readout.set_text(
            f"iteration {frame['iter']:>2}"
            f"{'  (converged)' if settled else ''}\n"
            f"mean loglike {frame['mean_loglike']:.4f}"
        )
        return [scatter, *ellipses, *labels, readout]

    anim = FuncAnimation(fig, draw, frames=order, interval=1000 / args.fps, blit=False)

    out = Path(__file__).parent / ("em_old_faithful.mp4" if args.mp4 else "em_old_faithful.gif")
    if args.mp4:
        anim.save(out, writer="ffmpeg", fps=args.fps, dpi=120, savefig_kwargs={"facecolor": SURFACE})
    else:
        anim.save(out, writer=PillowWriter(fps=args.fps), dpi=110,
                  savefig_kwargs={"facecolor": SURFACE})
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
