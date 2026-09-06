"""Benchmark `em` against scikit-learn's GaussianMixture.

Run with ``uv run --group bench python benchmarks/vs_sklearn.py``.

Both implementations maximize the same objective, so the achieved
log-likelihood is a fair head-to-head number: higher is a better fit of the
same model to the same data, whatever the code inside. Recovery error and
adjusted Rand index say whether that likelihood corresponds to the right
answer, and the timing says what it cost.

Settings are matched as closely as the two APIs allow:

===================  ==========================  ==========================
                     em                          GaussianMixture
===================  ==========================  ==========================
convergence          ``tol`` on mean loglike     ``tol`` on mean lower bound
variance floor       ``reg``                     ``reg_covar``
restarts             ``n_init``                  ``n_init``
seeding              k-means++ (seeding only)    ``init_params="k-means++"``
covariance           full, per component         ``covariance_type="full"``
===================  ==========================  ==========================

sklearn's default ``init_params="kmeans"`` runs k-means to convergence before
EM even starts, which is a different (and more expensive) algorithm; the
k-means++ setting is the like-for-like comparison.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from em import EM

TOL = 1e-8
REG = 1e-6
MAX_ITER = 1_000
N_INIT = 5
REPEATS = 2


def make_blobs(n_samples, n_features, n_groups, separation, seed):
    """Well-formed mixture data with known labels and known means."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, separation, size=(n_groups, n_features))
    covs = []
    for _ in range(n_groups):
        a = rng.normal(size=(n_features, n_features)) / np.sqrt(n_features)
        covs.append(a @ a.T + np.eye(n_features) * 0.5)
    weights = rng.dirichlet(np.full(n_groups, 5.0))
    labels = rng.choice(n_groups, size=n_samples, p=weights)
    x = np.empty((n_samples, n_features))
    for k in range(n_groups):
        mask = labels == k
        x[mask] = rng.multivariate_normal(means[k], covs[k], size=int(mask.sum()))
    return x, labels, means


def mean_error(fitted_means, true_means):
    """Mean absolute error under the optimal matching of components to truth.

    Component identities are arbitrary, so fitted components have to be paired
    with true ones before their positions can be compared. Pairing greedily —
    walk the fitted components, give each its nearest unclaimed true mean — is
    not good enough: one component's locally better choice can force a terrible
    pairing on a later one, and the error that produces is a property of the
    matcher, not of the fit. On overlapping mixtures that inflated the reported
    error by 68% in one case here and reversed the winner in another.

    ``linear_sum_assignment`` (Hungarian) minimizes the total instead, which is
    what the metric always meant.
    """
    cost = np.abs(fitted_means[:, None, :] - true_means[None, :, :]).mean(axis=2)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def time_it(fn):
    """Best of REPEATS, to keep noise out of the comparison."""
    best, out = np.inf, None
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return best, out


def run_case(name, n_samples, n_features, n_groups, separation, seed=0):
    x, labels, true_means = make_blobs(n_samples, n_features, n_groups, separation, seed)
    family = "multivariate-normal" if n_features > 1 else "normal"
    flat = x[:, 0] if n_features == 1 else x

    def fit_em():
        return EM(family, seed=0, tol=TOL, reg=REG).train(
            flat, n_groups=n_groups, n_iters=MAX_ITER, n_init=N_INIT
        )

    def fit_sklearn():
        return GaussianMixture(
            n_components=n_groups, covariance_type="full", tol=TOL, reg_covar=REG,
            max_iter=MAX_ITER, n_init=N_INIT, init_params="k-means++", random_state=0,
        ).fit(x)

    t_em, model = time_it(fit_em)
    t_sk, gmm = time_it(fit_sklearn)

    em_means = model.params_[:, :n_features]
    rows = [
        ("mean loglike", model.score(), float(gmm.score(x)), "higher"),
        ("mean abs error", mean_error(em_means, true_means), mean_error(gmm.means_, true_means), "lower"),
        ("adjusted Rand", adjusted_rand_score(labels, model.classify()),
         adjusted_rand_score(labels, gmm.predict(x)), "higher"),
        ("fit time (s)", t_em, t_sk, "lower"),
    ]

    print(f"\n{name}  (n={n_samples:,}, d={n_features}, k={n_groups})")
    print(f"  {'metric':<16}{'em':>14}{'sklearn':>14}{'':>4}{'winner':>10}")
    for metric, ours, theirs, direction in rows:
        better = ours > theirs if direction == "higher" else ours < theirs
        close = np.isclose(ours, theirs, rtol=1e-6, atol=1e-9)
        winner = "tie" if close else ("em" if better else "sklearn")
        print(f"  {metric:<16}{ours:>14.6g}{theirs:>14.6g}{'':>4}{winner:>10}")
    return t_em, t_sk


def main() -> None:
    print(__doc__.split("\n\n")[1].strip())
    cases = [
        ("1-D, well separated", 20_000, 1, 3, 4.0),
        ("1-D, overlapping", 20_000, 1, 4, 1.5),
        ("2-D, well separated", 20_000, 2, 3, 4.0),
        ("2-D, overlapping", 20_000, 2, 5, 2.0),
        ("5-D", 20_000, 5, 4, 3.0),
        ("large 2-D", 100_000, 2, 4, 3.0),
    ]
    ratios = []
    for case in cases:
        t_em, t_sk = run_case(*case)
        ratios.append(t_em / t_sk)
    print(f"\nfit time, em / sklearn: median {np.median(ratios):.2f}x, "
          f"range {min(ratios):.2f}-{max(ratios):.2f}x")


if __name__ == "__main__":
    main()
