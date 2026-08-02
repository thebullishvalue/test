"""
Latent-factor extraction from the cross-section.

The engine refits its factor model every ``refit_every`` sessions. Two things
matter and are easy to get wrong:

1. **Scaling must be frozen at fit time.** The window's mean/std are part of
   the model; applying the fitted model to future rows requires reusing those
   statistics, not recomputing them on the new data.

2. **Components must be aligned across refits.** PCA/ICA components carry an
   arbitrary sign, and ICA an arbitrary order. Without alignment, "factor 3"
   means something different after every refit, coefficient paths look like
   noise, and beta-stability diagnostics measure nothing but sign flips. We
   solve an optimal assignment between successive loading matrices and flip
   signs to match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, FastICA, FactorAnalysis


# ---------------------------------------------------------------------------
# Compact linear/non-linear autoencoder (numpy, Adam)
# ---------------------------------------------------------------------------
class Autoencoder:
    """Single-hidden-layer autoencoder used as a non-linear factor model.

    Deliberately small and dependency-free: the panels here are at most a few
    hundred columns by a few hundred rows, so a full-batch numpy trainer is
    faster than spinning up a deep-learning framework and keeps the engine
    reproducible.
    """

    def __init__(self, n_components: int, epochs: int = 400, lr: float = 0.02,
                 seed: int = 0, activation: str = "tanh"):
        self.k = n_components
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.activation = activation
        self.W1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.W2: np.ndarray | None = None
        self.b2: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def _act(self, z):
        return np.tanh(z) if self.activation == "tanh" else z

    def _dact(self, a):
        return 1.0 - a ** 2 if self.activation == "tanh" else np.ones_like(a)

    def fit(self, X: np.ndarray) -> "Autoencoder":
        rng = np.random.default_rng(self.seed)
        T, n = X.shape
        k = min(self.k, n)
        scale = 1.0 / np.sqrt(n)
        self.W1 = rng.normal(0, scale, (n, k))
        self.b1 = np.zeros(k)
        self.W2 = rng.normal(0, scale, (k, n))
        self.b2 = np.zeros(n)

        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1_, b2_, eps = 0.9, 0.999, 1e-8

        for step in range(1, self.epochs + 1):
            H = self._act(X @ self.W1 + self.b1)
            Xh = H @ self.W2 + self.b2
            E = (Xh - X) / T

            gW2 = H.T @ E
            gb2 = E.sum(axis=0)
            dH = (E @ self.W2.T) * self._dact(H)
            gW1 = X.T @ dH
            gb1 = dH.sum(axis=0)

            for i, g in enumerate([gW1, gb1, gW2, gb2]):
                m[i] = b1_ * m[i] + (1 - b1_) * g
                v[i] = b2_ * v[i] + (1 - b2_) * g * g
                mh = m[i] / (1 - b1_ ** step)
                vh = v[i] / (1 - b2_ ** step)
                params[i] -= self.lr * mh / (np.sqrt(vh) + eps)

        # Per-latent share of reconstructed variance, reported like an
        # explained-variance ratio so downstream code can treat all methods
        # uniformly.
        H = self._act(X @ self.W1 + self.b1)
        total = np.var(X, axis=0).sum()
        shares = np.array([np.var(np.outer(H[:, j], self.W2[j]), axis=0).sum()
                           for j in range(H.shape[1])])
        self.explained_variance_ratio_ = shares / total if total > 0 else shares
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._act(X @ self.W1 + self.b1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def components_(self) -> np.ndarray:
        """Decoder weights, in the (k, n) convention sklearn uses."""
        return self.W2


# ---------------------------------------------------------------------------
# Fitted factor model wrapper
# ---------------------------------------------------------------------------
@dataclass
class FittedFactors:
    """A factor model frozen at fit time, plus its scaler and diagnostics."""

    model: object
    mean: np.ndarray
    scale: np.ndarray
    loadings: np.ndarray                 # (k, n_instruments)
    explained: np.ndarray                # (k,) variance share, may be NaN
    method: str
    signs: np.ndarray = field(default=None)   # applied after alignment
    order: np.ndarray = field(default=None)

    @property
    def k(self) -> int:
        return self.loadings.shape[0]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project raw (unscaled) rows into factor space using frozen stats."""
        Xs = (np.asarray(X, dtype="float64") - self.mean) / self.scale
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        S = self.model.transform(Xs)
        if self.order is not None:
            S = S[:, self.order]
        if self.signs is not None:
            S = S * self.signs
        return S

    def effective_rank(self) -> float:
        """Entropy-based effective rank of the retained factor spectrum."""
        ev = np.asarray(self.explained, dtype="float64")
        ev = ev[np.isfinite(ev) & (ev > 0)]
        if ev.size == 0:
            return float("nan")
        p = ev / ev.sum()
        return float(np.exp(-(p * np.log(p)).sum()))


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype="float64").copy()
    X[~np.isfinite(X)] = np.nan
    mean = np.nanmean(X, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    idx = np.where(np.isnan(X))
    X[idx] = np.take(mean, idx[1])
    scale = X.std(axis=0)
    scale[(scale == 0) | ~np.isfinite(scale)] = 1.0
    return (X - mean) / scale, mean, scale


def fit_factors(X: np.ndarray, method: str, k: int,
                previous: FittedFactors | None = None,
                seed: int = 0) -> FittedFactors:
    """Fit a factor model on one calibration window and align it to ``previous``."""
    Xs, mean, scale = _standardise(X)
    k = int(max(1, min(k, Xs.shape[1], max(1, Xs.shape[0] - 1))))

    if method == "ICA":
        model = FastICA(n_components=k, random_state=seed, max_iter=800,
                        tol=1e-3, whiten="unit-variance")
    elif method == "FactorAnalysis":
        model = FactorAnalysis(n_components=k, random_state=seed)
    elif method == "Autoencoder":
        model = Autoencoder(n_components=k, seed=seed)
    else:
        model = PCA(n_components=k, random_state=seed)

    try:
        model.fit(Xs)
    except Exception:
        model = PCA(n_components=k, random_state=seed).fit(Xs)
        method = "PCA"

    loadings = np.asarray(getattr(model, "components_", np.zeros((k, Xs.shape[1]))))

    ev = getattr(model, "explained_variance_ratio_", None)
    if ev is None or len(ev) != loadings.shape[0]:
        # ICA and FactorAnalysis do not define one; derive the variance share
        # each component's scores carry relative to the panel total.
        S = model.transform(Xs)
        var_s = S.var(axis=0)
        ev = var_s / max(Xs.var(axis=0).sum(), 1e-12)
    ev = np.asarray(ev, dtype="float64")

    fitted = FittedFactors(model=model, mean=mean, scale=scale, loadings=loadings,
                           explained=ev, method=method,
                           signs=np.ones(loadings.shape[0]),
                           order=np.arange(loadings.shape[0]))

    if previous is not None and previous.loadings.shape == loadings.shape:
        fitted = align_factors(fitted, previous)
    else:
        fitted = _canonicalise_signs(fitted)
    return fitted


def _canonicalise_signs(f: FittedFactors) -> FittedFactors:
    """Fix an arbitrary but deterministic sign: majority of loadings positive."""
    signs = np.where(f.loadings.sum(axis=1) < 0, -1.0, 1.0)
    f.loadings = f.loadings * signs[:, None]
    f.signs = signs
    return f


def align_factors(new: FittedFactors, prev: FittedFactors) -> FittedFactors:
    """Reorder and sign-flip ``new``'s components to match ``prev``'s.

    Matching maximises total absolute cosine similarity between loading
    vectors via optimal assignment, then each matched pair's sign is set so
    the correlation is positive.
    """
    A = new.loadings
    B = prev.loadings
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sim = An @ Bn.T                                   # (k_new, k_prev)

    row, col = linear_sum_assignment(-np.abs(sim))
    order = np.empty(A.shape[0], dtype=int)
    order[col] = row                                  # position i <- component order[i]
    signs = np.sign(sim[order, np.arange(len(order))])
    signs[signs == 0] = 1.0

    new.loadings = A[order] * signs[:, None]
    new.explained = np.asarray(new.explained)[order]
    new.order = order
    new.signs = signs
    return new


def loading_drift(new: FittedFactors, prev: FittedFactors) -> float:
    """Mean absolute change in aligned loadings — a structural-stability gauge."""
    if prev is None or new.loadings.shape != prev.loadings.shape:
        return float("nan")
    return float(np.mean(np.abs(new.loadings - prev.loadings)))


def orthogonalise(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Residualise columns of ``X`` on factor scores ``S`` (least squares).

    What remains is the part of each instrument the common factors do *not*
    explain — the orthogonal information the engine mines for peers.
    """
    S1 = np.column_stack([np.ones(len(S)), S])
    coef, *_ = np.linalg.lstsq(S1, X, rcond=None)
    return X - S1 @ coef
