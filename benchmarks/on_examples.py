"""Benchmark `em` against scikit-learn on the two example datasets.

Run with ``uv run --group bench python benchmarks/on_examples.py``.

``vs_sklearn.py`` uses synthetic mixtures, where the true means are known and
recovery error is measurable. This one runs the same head-to-head on the real
data the examples fit — the galaxy velocities and the Old Faithful eruptions —
where there is no ground truth, so the questions are different:

* **same answer?** Both maximize the same objective, so the achieved mean
  log-likelihood is directly comparable, and the adjusted Rand index between
  the two labellings says whether the two fits are the same solution rather
  than two different local optima that happen to score alike.
* **same model?** Each implementation picks its own component count by BIC over
  the same candidate range. Agreeing there matters more than agreeing on the
  sixth decimal of a log-likelihood.
* **what does it cost?** Two timings: one fit at the chosen ``k``, and the whole
  BIC sweep, which is what the examples actually run.

Settings are matched as in ``vs_sklearn.py``: ``tol``, the variance floor,
``n_init``, k-means++ seeding, and full covariances.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from em import EM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from datasets import galaxy_velocities, old_faithful  # noqa: E402

TOL = 1e-8
REG = 1e-6
MAX_ITER = 1_000
# These are small datasets with a rough likelihood surface, so both
# implementations get the generous restart budget the examples use.
N_INIT = 24
REPEATS = 3


def time_it(fn):
    """Best of REPEATS, to keep noise out of the comparison."""
    best, out = np.inf, None
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return best, out


def fit_em(x, k):
    family = "multivariate-normal" if x.ndim == 2 and x.shape[1] > 1 else "normal"
    flat = x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x
    return EM(family, seed=0, tol=TOL, reg=REG).train(
        flat, n_groups=k, n_iters=MAX_ITER, n_init=N_INIT
    )


def fit_sklearn(x, k):
    return GaussianMixture(
        n_components=k, covariance_type="full", tol=TOL, reg_covar=REG,
        max_iter=MAX_ITER, n_init=N_INIT, init_params="k-means++", random_state=0,
    ).fit(x)


def run_case(name, x, candidates):
    """Fit both implementations, let each choose its own k by BIC, compare."""
    rows = np.atleast_2d(x.T).T if x.ndim == 1 else x  # sklearn always wants 2-D

    def sweep_em():
        return {k: fit_em(x, k).bic() for k in candidates}

    def sweep_sk():
        return {k: fit_sklearn(rows, k).bic(rows) for k in candidates}

    t_sweep_em, bic_em = time_it(sweep_em)
    t_sweep_sk, bic_sk = time_it(sweep_sk)
    k_em = min(bic_em, key=bic_em.get)
    k_sk = min(bic_sk, key=bic_sk.get)

    # Compare the fits at a shared k, so the log-likelihoods stay comparable
    # even in the case where the two disagree on the component count.
    k = k_em
    t_em, model = time_it(lambda: fit_em(x, k))
    t_sk, gmm = time_it(lambda: fit_sklearn(rows, k))
    agreement = adjusted_rand_score(model.classify(), gmm.predict(rows))

    print(f"\n{name}  (n={rows.shape[0]:,}, d={rows.shape[1]}, k from BIC over {list(candidates)})")
    print(f"  {'metric':<22}{'em':>14}{'sklearn':>14}{'':>4}{'winner':>10}")
    metrics = [
        ("groups chosen (BIC)", float(k_em), float(k_sk), "same"),
        ("BIC at chosen k", bic_em[k_em], bic_sk[k_sk], "lower"),
        (f"mean loglike (k={k})", model.score(), float(gmm.score(rows)), "higher"),
        (f"one fit, s (k={k})", t_em, t_sk, "lower"),
        ("whole BIC sweep, s", t_sweep_em, t_sweep_sk, "lower"),
    ]
    for metric, ours, theirs, direction in metrics:
        close = np.isclose(ours, theirs, rtol=1e-6, atol=1e-9)
        if direction == "same":
            winner = "agree" if close else "differ"
        else:
            better = ours > theirs if direction == "higher" else ours < theirs
            winner = "tie" if close else ("em" if better else "sklearn")
        print(f"  {metric:<22}{ours:>14.6g}{theirs:>14.6g}{'':>4}{winner:>10}")
    print(f"  {'label agreement':<22}{agreement:>14.6g}    (adjusted Rand, em vs sklearn)")
    return t_sweep_em, t_sweep_sk


def main() -> None:
    print(__doc__.split("\n\n")[1].strip())
    cases = [
        ("galaxy velocities", galaxy_velocities(), range(1, 9)),
        ("Old Faithful", old_faithful(), range(1, 7)),
    ]
    ratios = []
    for name, data, candidates in cases:
        t_em, t_sk = run_case(name, data, candidates)
        ratios.append(t_em / t_sk)
    print(f"\nBIC sweep, em / sklearn: {' and '.join(f'{r:.2f}x' for r in ratios)}")


if __name__ == "__main__":
    main()
