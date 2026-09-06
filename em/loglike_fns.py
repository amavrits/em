"""Component families for finite-mixture EM.

Every entry of :data:`LOGLIKE_FNS` is a :class:`Family`: the log-density of one
mixture component, the *weighted* MLE that the M-step needs, and a sampler used
by the examples and tests.

Parameters travel as plain 1-D float arrays, so a user-supplied
``loglike_fn(x, p)`` drops into the same slot as a built-in family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import digamma, gammaln, polygamma

__all__ = ["Family", "LOGLIKE_FNS", "get_family"]

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class Family:
    """One parametric component of a mixture.

    Attributes:
        name: Key under which the family is registered.
        param_names: Names of the parameters, in the order they appear in ``p``.
        logpdf: ``(x, p) -> log density``, evaluated elementwise over ``x``.
        mle: ``(x, w, reg, p0) -> p``, the weighted maximum-likelihood fit.
            ``w`` are the responsibilities of one component, ``reg`` is a
            variance/scale floor guarding against collapse onto a single point,
            and ``p0`` is the current estimate (a warm start for families that
            need to iterate; ignored by the closed-form ones).
        rvs: ``(p, size, rng) -> samples`` from a single component.
        support: ``"real"``, ``"positive"`` or ``"count"``; used to reject data
            the family cannot possibly have generated.
        start: Initial parameter vector, for families whose M-step iterates.
            ``None`` for the closed-form built-ins, which need no guess.
        multivariate: True for a family whose observations are rows of an
            ``(n_samples, n_features)`` array rather than scalars.
        conditional: True for a family that models ``p(y | x)`` rather than a
            density over ``x``. Its data arrives packed as ``(n_samples,
            n_features + 1)`` with the target in the last column, and
            :class:`~em.em.EM` requires an explicit ``y`` to build it.
        param_names_fn: For families whose parameter count depends on the width
            of the data (a regression has one coefficient per feature), a
            ``n_features -> names`` callable. Takes precedence over
            ``param_names``.
    """

    name: str
    param_names: tuple[str, ...]
    logpdf: Callable[[np.ndarray, np.ndarray], np.ndarray]
    mle: Callable[[np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]
    rvs: Callable[[np.ndarray, int, np.random.Generator], np.ndarray]
    support: str = "real"
    start: np.ndarray | None = None
    multivariate: bool = False
    conditional: bool = False
    param_names_fn: Callable[[int], tuple[str, ...]] | None = None

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def names(self, n_features: int = 1) -> tuple[str, ...]:
        """Parameter names for data of the given width."""
        if self.param_names_fn is not None:
            return self.param_names_fn(n_features)
        return self.param_names

    def n_params_for(self, n_features: int = 1) -> int:
        """Number of parameters for data of the given width."""
        return len(self.names(n_features))

    def validate(self, data: np.ndarray) -> None:
        """Raise if the data falls outside the family's support.

        For a conditional family the support constrains the target, so only the
        last column is checked.
        """
        x = data[:, -1] if self.conditional else data
        if self.support == "positive" and np.any(x <= 0):
            raise ValueError(f"The '{self.name}' family requires strictly positive data.")
        if self.support == "count":
            if np.any(x < 0):
                raise ValueError(f"The '{self.name}' family requires non-negative data.")
            if np.any(x != np.round(x)):
                raise ValueError(f"The '{self.name}' family requires integer counts.")

    def describe(self, p: np.ndarray, n_features: int = 1) -> str:
        """Render one parameter vector as ``name=value`` pairs."""
        return ", ".join(f"{k}={v:.4g}" for k, v in zip(self.names(n_features), np.ravel(p)))


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Return the weight total and the weighted mean of ``x`` (0.0 if no mass)."""
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0, float(np.mean(x)) if x.size else 0.0
    return total, float(np.dot(w, x) / total)


# --- normal ------------------------------------------------------------------

def _normal_logpdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    mu, sigma = float(p[0]), float(p[1])
    z = (x - mu) / sigma
    return -0.5 * _LOG_2PI - np.log(sigma) - 0.5 * z * z


def _normal_mle(x: np.ndarray, w: np.ndarray, reg: float, p0: np.ndarray) -> np.ndarray:
    total, mu = _weighted_mean(x, w)
    if total <= 0.0:
        return np.asarray(p0, dtype=float)
    var = float(np.dot(w, (x - mu) ** 2) / total)
    return np.array([mu, np.sqrt(var + reg)])


def _normal_rvs(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(float(p[0]), float(p[1]), size=size)


# --- lognormal ---------------------------------------------------------------

def _lognormal_logpdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    log_x = np.log(x)
    return _normal_logpdf(log_x, p) - log_x


def _lognormal_mle(x: np.ndarray, w: np.ndarray, reg: float, p0: np.ndarray) -> np.ndarray:
    return _normal_mle(np.log(x), w, reg, p0)


def _lognormal_rvs(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.lognormal(float(p[0]), float(p[1]), size=size)


# --- exponential -------------------------------------------------------------

def _exponential_logpdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    rate = float(p[0])
    return np.log(rate) - rate * x


def _exponential_mle(x: np.ndarray, w: np.ndarray, reg: float, p0: np.ndarray) -> np.ndarray:
    total, mean = _weighted_mean(x, w)
    if total <= 0.0:
        return np.asarray(p0, dtype=float)
    return np.array([1.0 / max(mean, np.sqrt(reg))])


def _exponential_rvs(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.exponential(1.0 / float(p[0]), size=size)


# --- poisson -----------------------------------------------------------------

def _poisson_logpmf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    lam = float(p[0])
    return x * np.log(lam) - lam - gammaln(x + 1.0)


def _poisson_mle(x: np.ndarray, w: np.ndarray, reg: float, p0: np.ndarray) -> np.ndarray:
    total, mean = _weighted_mean(x, w)
    if total <= 0.0:
        return np.asarray(p0, dtype=float)
    return np.array([max(mean, np.sqrt(reg))])


def _poisson_rvs(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.poisson(float(p[0]), size=size).astype(float)


# --- gamma -------------------------------------------------------------------

def _gamma_logpdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    shape, scale = float(p[0]), float(p[1])
    return (shape - 1.0) * np.log(x) - x / scale - gammaln(shape) - shape * np.log(scale)


def _gamma_mle(x: np.ndarray, w: np.ndarray, reg: float, p0: np.ndarray) -> np.ndarray:
    """Weighted gamma MLE: Newton on the shape, then the scale in closed form.

    There is no closed form for the shape, but ``log(k) - digamma(k) = s`` is
    smooth and one-dimensional, so a few Newton steps from the standard
    Minka approximation converge to machine precision.
    """
    total, mean = _weighted_mean(x, w)
    if total <= 0.0 or mean <= 0.0:
        return np.asarray(p0, dtype=float)

    s = float(np.log(mean) - np.dot(w, np.log(x)) / total)
    if not np.isfinite(s) or s <= 0.0:
        # Degenerate spread: fall back to a very peaked component.
        shape = 1.0 / max(reg, 1e-12)
    else:
        shape = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
        for _ in range(50):
            num = np.log(shape) - digamma(shape) - s
            den = 1.0 / shape - polygamma(1, shape)
            if den == 0.0:
                break
            step = num / den
            new_shape = shape - step
            if new_shape <= 0.0:
                new_shape = shape / 2.0
            converged = abs(new_shape - shape) <= 1e-12 * shape
            shape = new_shape
            if converged:
                break
    shape = float(np.clip(shape, 1e-8, 1e8))
    return np.array([shape, max(mean / shape, np.sqrt(reg))])


def _gamma_rvs(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.gamma(float(p[0]), float(p[1]), size=size)

# --- linear regression -------------------------------------------------------

def _lr_split(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack ``(n, d+1)`` data into a design matrix with intercept, and y."""
    features, y = data[:, :-1], data[:, -1]
    return np.column_stack([np.ones(features.shape[0]), features]), y


def _lr_names(n_features: int) -> tuple[str, ...]:
    slopes = tuple(f"beta{i}" for i in range(1, n_features + 1))
    return ("intercept",) + slopes + ("sigma",)


def _lr_logpdf(data: np.ndarray, p: np.ndarray) -> np.ndarray:
    design, y = _lr_split(data)
    beta, sigma = np.asarray(p[:-1], dtype=float), float(p[-1])
    z = (y - design @ beta) / sigma
    return -0.5 * _LOG_2PI - np.log(sigma) - 0.5 * z * z


def _lr_mle(data: np.ndarray, w: np.ndarray, reg: float, p0: np.ndarray) -> np.ndarray:
    """Weighted least squares: the exact M-step for a mixture of regressions.

    Maximizing the expected complete-data log-likelihood for one component is
    the normal-equations problem with the responsibilities as weights. Scaling
    the rows by ``sqrt(w)`` turns it into ordinary least squares, which
    ``lstsq`` solves without forming or inverting ``X'WX``.
    """
    total = float(np.sum(w))
    if total <= 0.0:
        return np.asarray(p0, dtype=float)

    design, y = _lr_split(data)
    root = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    residual = y - design @ beta
    var = float(np.dot(w, residual ** 2) / total)
    return np.concatenate([beta, [np.sqrt(var + reg)]])


def _lr_rvs(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    raise NotImplementedError(
        "sample() needs predictors for a conditional family; generate X yourself "
        "and draw y ~ Normal(X @ beta, sigma)."
    )


# --- multivariate normal -----------------------------------------------------

def _mvn_names(n_features: int) -> tuple[str, ...]:
    means = tuple(f"mu{i + 1}" for i in range(n_features))
    cov = tuple(f"cov{i + 1}{j + 1}" for i in range(n_features) for j in range(i + 1))
    return means + cov


def _mvn_dim(n_params: int) -> int:
    """Recover the number of features from the length of a parameter vector."""
    d = int(round((-3.0 + np.sqrt(9.0 + 8.0 * n_params)) / 2.0))
    if d + d * (d + 1) // 2 != n_params:
        raise ValueError(f"{n_params} is not a valid multivariate-normal parameter count.")
    return d


def _mvn_unpack(p: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a flat vector into the mean and the full covariance matrix.

    Only the lower triangle is stored, so the parameter count matches the
    number of *free* parameters — which is what AIC and BIC need.
    """
    mean = np.asarray(p[:n_features], dtype=float)
    lower = np.zeros((n_features, n_features))
    lower[np.tril_indices(n_features)] = p[n_features:]
    cov = lower + lower.T - np.diag(np.diag(lower))
    return mean, cov


def _mvn_logpdf(data: np.ndarray, p: np.ndarray) -> np.ndarray:
    n_features = data.shape[1]
    mean, cov = _mvn_unpack(p, n_features)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("A component covariance is not positive definite.") from exc
    # Solve rather than invert: mahalanobis = ||L^-1 (x - mu)||^2.
    whitened = solve_triangular(chol, (data - mean).T, lower=True)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return -0.5 * (n_features * _LOG_2PI + log_det + np.sum(whitened ** 2, axis=0))


def _mvn_mle(data: np.ndarray, w: np.ndarray, reg: float, p0: np.ndarray) -> np.ndarray:
    total = float(np.sum(w))
    if total <= 0.0:
        return np.asarray(p0, dtype=float)
    n_features = data.shape[1]
    mean = (w @ data) / total
    delta = data - mean
    cov = (delta * w[:, None]).T @ delta / total
    cov.flat[:: n_features + 1] += reg          # ridge the diagonal, as in the 1-D case
    return np.concatenate([mean, cov[np.tril_indices(n_features)]])


def _mvn_rvs(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    n_features = _mvn_dim(len(p))
    mean, cov = _mvn_unpack(np.asarray(p, dtype=float), n_features)
    return rng.multivariate_normal(mean, cov, size=size)


LOGLIKE_FNS: dict[str, Family] = {
    "normal": Family(
        name="normal",
        param_names=("mu", "sigma"),
        logpdf=_normal_logpdf,
        mle=_normal_mle,
        rvs=_normal_rvs,
        support="real",
    ),
    "lognormal": Family(
        name="lognormal",
        param_names=("mu", "sigma"),
        logpdf=_lognormal_logpdf,
        mle=_lognormal_mle,
        rvs=_lognormal_rvs,
        support="positive",
    ),
    "exponential": Family(
        name="exponential",
        param_names=("rate",),
        logpdf=_exponential_logpdf,
        mle=_exponential_mle,
        rvs=_exponential_rvs,
        support="positive",
    ),
    "poisson": Family(
        name="poisson",
        param_names=("lam",),
        logpdf=_poisson_logpmf,
        mle=_poisson_mle,
        rvs=_poisson_rvs,
        support="count",
    ),
    "gamma": Family(
        name="gamma",
        param_names=("shape", "scale"),
        logpdf=_gamma_logpdf,
        mle=_gamma_mle,
        rvs=_gamma_rvs,
        support="positive",
    ),
    "multivariate-normal": Family(
        name="multivariate-normal",
        param_names=("mu1", "cov11"),  # placeholder; see param_names_fn
        logpdf=_mvn_logpdf,
        mle=_mvn_mle,
        rvs=_mvn_rvs,
        support="real",
        multivariate=True,
        param_names_fn=_mvn_names,
    ),
    "linear-regression": Family(
        name="linear-regression",
        param_names=("intercept", "beta1", "sigma"),  # placeholder; see param_names_fn
        logpdf=_lr_logpdf,
        mle=_lr_mle,
        rvs=_lr_rvs,
        support="real",
        conditional=True,
        param_names_fn=_lr_names,
    ),
}


def get_family(name: str) -> Family:
    """Look up a built-in family, with the available names in the error."""
    try:
        return LOGLIKE_FNS[name]
    except KeyError:
        known = ", ".join(sorted(LOGLIKE_FNS))
        raise ValueError(f"Unknown loglikelihood function {name!r}. Available: {known}.") from None
