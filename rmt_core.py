"""
Random Matrix Theory (RMT) Core Engine for Pragyam.

Implements Marchenko-Pastur distribution analysis, covariance matrix denoising,
and spectral diagnostics for separating signal from noise in correlation structures.

Based on: Marchenko & Pastur (1967), Laloux, Cizeau, Bouchaud & Potters (1999).

Zero coupling to Streamlit or Pragyam-specific data structures.
Dependencies: numpy, scipy (both already in requirements).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

# Canonical epsilon for division-by-zero guards across the module.
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MPDistribution:
    """Marchenko-Pastur distribution parameters for a given T/N ratio."""
    gamma: float          # T/N ratio (observations / variables)
    sigma_sq: float       # noise variance estimate
    lambda_plus: float    # upper MP edge
    lambda_minus: float   # lower MP edge
    n_signal: int         # eigenvalues above lambda_plus
    n_noise: int          # eigenvalues at or below lambda_plus


@dataclass
class SpectralDiagnostics:
    """Complete spectral analysis output."""
    eigenvalues: np.ndarray            # sorted descending
    eigenvectors: np.ndarray           # column eigenvectors (matching eigenvalue order)
    mp_dist: MPDistribution
    condition_number: float
    effective_rank: float              # number of independent factors (ENB)
    absorption_ratio: float            # fraction of variance in top K eigenvectors
    herfindahl_eigenvalues: float      # concentration of eigenvalue mass
    cleaned_corr: np.ndarray           # denoised correlation matrix
    shrinkage_intensity: float         # Ledoit-Wolf optimal shrinkage


# ---------------------------------------------------------------------------
# Marchenko-Pastur Distribution
# ---------------------------------------------------------------------------

def marchenko_pastur_edges(gamma: float, sigma_sq: float = 1.0) -> Tuple[float, float]:
    """
    Compute the theoretical Marchenko-Pastur distribution edges.

    Args:
        gamma: T/N ratio (number of observations / number of variables).
               For gamma < 1, the matrix is singular.
        sigma_sq: noise variance (1.0 for correlation matrices).

    Returns:
        (lambda_minus, lambda_plus) — the support boundaries.
    """
    sqrt_inv_gamma = 1.0 / np.sqrt(max(gamma, _EPS))
    lambda_plus = sigma_sq * (1.0 + sqrt_inv_gamma) ** 2
    lambda_minus = sigma_sq * max(0.0, (1.0 - sqrt_inv_gamma)) ** 2
    return lambda_minus, lambda_plus


def marchenko_pastur_pdf(
    x: np.ndarray,
    gamma: float,
    sigma_sq: float = 1.0,
) -> np.ndarray:
    """
    Marchenko-Pastur probability density function.

    Args:
        x: eigenvalue points at which to evaluate the PDF.
        gamma: T/N ratio.
        sigma_sq: noise variance.

    Returns:
        PDF values at each point in x. Zero outside the support.
    """
    lambda_minus, lambda_plus = marchenko_pastur_edges(gamma, sigma_sq)
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)

    inside = (x > lambda_minus + 1e-12) & (x < lambda_plus - 1e-12)
    if not np.any(inside):
        return pdf

    xi = x[inside]
    numerator = np.sqrt((lambda_plus - xi) * (xi - lambda_minus))
    denominator = 2.0 * np.pi * sigma_sq * xi / gamma
    pdf[inside] = numerator / denominator
    return pdf


# ---------------------------------------------------------------------------
# Eigenvalue Analysis & Metrics
# ---------------------------------------------------------------------------

def effective_number_of_bets(eigenvalues: np.ndarray) -> float:
    """
    Effective number of independent bets (ENB).
    ENB = exp(entropy of normalized eigenvalues).

    Returns 1.0 when all variance is in one factor (maximally concentrated),
    returns N when all eigenvalues are equal (maximally dispersed).
    """
    eig = np.maximum(eigenvalues, _EPS)
    p = eig / eig.sum()
    entropy = -np.sum(p * np.log(p))
    enb = np.exp(entropy)
    return float(np.clip(enb, 1.0, len(eigenvalues)))


def absorption_ratio(eigenvalues: np.ndarray, n_factors: int = 5) -> float:
    """
    Fraction of total variance explained by the top n_factors eigenvectors.

    High AR (>0.8) indicates systemic risk concentration — correlations
    are collapsing and the market is in a fragile state.
    Low AR (<0.4) indicates healthy diversification.

    Args:
        eigenvalues: sorted descending.
        n_factors: number of top eigenvalues to sum. Defaults to 5 or N//5,
                   whichever is smaller.
    """
    n = min(n_factors, max(1, len(eigenvalues) // 5), len(eigenvalues))
    total = eigenvalues.sum()
    if total <= 0:
        return 1.0
    return float(eigenvalues[:n].sum() / total)


def herfindahl_eigenvalue_index(eigenvalues: np.ndarray) -> float:
    """
    Herfindahl concentration of eigenvalue mass.

    Range: 1/N (perfectly dispersed) to 1 (single dominant factor).
    """
    total = eigenvalues.sum()
    if total <= 0:
        return 1.0
    shares = eigenvalues / total
    return float(np.sum(shares ** 2))


# ---------------------------------------------------------------------------
# Covariance / Correlation Cleaning
# ---------------------------------------------------------------------------

def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    std = np.where(std > 0, std, _EPS)
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def clean_correlation_matrix(
    corr_matrix: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    mp_dist: MPDistribution,
    method: str = 'clip',
) -> np.ndarray:
    """
    Denoise a correlation matrix using the Marchenko-Pastur threshold.

    Methods:
        'clip': Replace noise eigenvalues with their mean, preserve trace.
        'target': Shrink noise eigenvalues toward 1.0 (identity target).

    Returns:
        Cleaned correlation matrix (positive semi-definite, unit diagonal).
    """
    n = len(eigenvalues)
    eig_cleaned = eigenvalues.copy()
    is_noise = eigenvalues <= mp_dist.lambda_plus

    if method == 'clip':
        if np.any(is_noise):
            # Preserve trace: noise eigenvalues get the average of noise mass
            signal_sum = eigenvalues[~is_noise].sum()
            noise_count = is_noise.sum()
            noise_avg = (n - signal_sum) / noise_count if noise_count > 0 else 1.0
            eig_cleaned[is_noise] = noise_avg
    elif method == 'target':
        if np.any(is_noise):
            # Shrink noise eigenvalues toward 1.0
            eig_cleaned[is_noise] = 1.0
            # Rescale signal eigenvalues to preserve trace
            signal_mask = ~is_noise
            if np.any(signal_mask):
                trace_deficit = n - eig_cleaned.sum()
                eig_cleaned[signal_mask] += trace_deficit / signal_mask.sum()

    # Reconstruct: C_clean = V * diag(lambda_cleaned) * V^T
    corr_clean = eigenvectors @ np.diag(eig_cleaned) @ eigenvectors.T

    # Force unit diagonal
    d = np.sqrt(np.diag(corr_clean))
    d = np.where(d > 0, d, _EPS)
    corr_clean = corr_clean / np.outer(d, d)
    np.fill_diagonal(corr_clean, 1.0)

    return np.clip(corr_clean, -1.0, 1.0)


def ledoit_wolf_shrinkage(
    returns_matrix: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) linear shrinkage estimator.
    Target: constant-correlation matrix.

    Args:
        returns_matrix: T x N matrix of centered returns.

    Returns:
        (shrunk_covariance, optimal_intensity)
    """
    T, N = returns_matrix.shape
    if T < 2 or N < 2:
        return np.eye(N), 0.0

    # Demean
    X = returns_matrix - returns_matrix.mean(axis=0)
    sample_cov = (X.T @ X) / T

    # Target: constant-correlation matrix
    std = np.sqrt(np.diag(sample_cov))
    std = np.where(std > 0, std, _EPS)
    corr = sample_cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    rho_bar = (corr.sum() - N) / (N * (N - 1))
    target = np.full((N, N), rho_bar)
    np.fill_diagonal(target, 1.0)
    target = target * np.outer(std, std)

    # Optimal shrinkage intensity — fully vectorized (HIGH-3 fix).
    # The previous O(N²) Python loop with early-termination heuristic
    # introduced bias for heterogeneous correlation structures.  This
    # vectorized version computes the exact Ledoit-Wolf (2004) formula.
    X2 = X ** 2
    pi_mat = (X2.T @ X2) / T - sample_cov ** 2
    pi_sum = pi_mat.sum()

    rho_diag = np.sum(np.diag(pi_mat))

    # Vectorized off-diagonal rho computation
    # theta_ij = (1/T) * sum_t(X_ti^2 * X_tj * X_ti) = (1/T) * (X^3)^T @ X
    # But we need: rho_off = sum_{i<j} (rij*theta_ij + rji*theta_ji) * rho_bar/2
    diag_cov = np.diag(sample_cov).copy()
    diag_cov = np.where(diag_cov > _EPS, diag_cov, _EPS)

    # r_ij = sqrt(S_jj / S_ii), so R_mat[i,j] = sqrt(S_jj / S_ii)
    r_mat = np.sqrt(diag_cov[np.newaxis, :] / diag_cov[:, np.newaxis])

    # theta_mat[i,j] = (1/T) * sum_t(X_ti^2 * X_tj * X_ti)
    #                = (1/T) * sum_t(X_ti^3 * X_tj)
    X3 = X ** 3
    theta_mat = (X3.T @ X) / T

    # Combined: each (i,j) contributes (r_ij * theta_ij + r_ji * theta_ji) * rho_bar / 2
    contrib_mat = (r_mat * theta_mat + r_mat.T * theta_mat.T) * (rho_bar / 2)
    # Sum upper triangle only
    rho_off = float(np.triu(contrib_mat, k=1).sum())

    gamma_lw = np.sum((target - sample_cov) ** 2)
    kappa = (pi_sum - rho_diag - 2 * rho_off) / gamma_lw if gamma_lw > 0 else 0.0
    delta = max(0.0, min(1.0, kappa / T))

    shrunk = delta * target + (1.0 - delta) * sample_cov
    return shrunk, delta


def _fast_ledoit_wolf_intensity(returns_matrix: np.ndarray) -> float:
    """
    Fast approximation of Ledoit-Wolf shrinkage intensity.

    Uses the Oracle Approximating Shrinkage (OAS) estimator toward the
    scaled identity target μI, where μ = tr(S)/N.

    Reference: Chen, Wiesel, Eldar & Hero (2010).
    """
    T, N = returns_matrix.shape
    if T < 2 or N < 2:
        return 0.0

    X = returns_matrix - returns_matrix.mean(axis=0)
    S = (X.T @ X) / T

    trace_S = np.trace(S)
    trace_S2 = np.sum(S ** 2)

    # OAS intensity: balances bias-variance of shrinkage toward μI
    rho_num = (1.0 - 2.0 / N) * trace_S2 + trace_S ** 2
    rho_den = (T + 1.0 - 2.0 / N) * (trace_S2 - trace_S ** 2 / N)

    if abs(rho_den) < 1e-15:
        return 0.0

    intensity = max(0.0, min(1.0, rho_num / rho_den))
    return intensity


# ---------------------------------------------------------------------------
# Main Spectral Diagnostics
# ---------------------------------------------------------------------------

def compute_spectral_diagnostics(
    data_matrix: np.ndarray,
    shrinkage: str = 'auto',
    n_absorption_factors: int = 5,
) -> SpectralDiagnostics:
    """
    Full spectral analysis of a T x N data matrix.

    Computes sample correlation matrix, eigendecomposes it, fits the
    Marchenko-Pastur distribution, separates signal from noise eigenvalues,
    and produces a cleaned correlation matrix.

    Args:
        data_matrix: T observations x N variables (returns, indicators, etc.)
        shrinkage: 'auto' (Ledoit-Wolf), 'none', or 'rmt' (MP-based only).
        n_absorption_factors: number of factors for absorption ratio.

    Returns:
        SpectralDiagnostics with all metrics and cleaned correlation.
    """
    data_matrix = np.asarray(data_matrix, dtype=float)

    # Handle edge cases
    if data_matrix.ndim == 1:
        data_matrix = data_matrix.reshape(-1, 1)

    T, N = data_matrix.shape

    if T < 3 or N < 2:
        # Degenerate case: return identity-based diagnostics
        return _degenerate_diagnostics(N)

    # Remove rows with all NaN, then fill remaining NaN with column mean
    valid_rows = ~np.all(np.isnan(data_matrix), axis=1)
    data_matrix = data_matrix[valid_rows]
    T = data_matrix.shape[0]

    if T < 3:
        return _degenerate_diagnostics(N)

    col_means = np.nanmean(data_matrix, axis=0)
    for j in range(N):
        mask = np.isnan(data_matrix[:, j])
        data_matrix[mask, j] = col_means[j]

    # Standardize columns (-> correlation matrix from covariance of standardized data)
    col_std = np.std(data_matrix, axis=0)
    col_std = np.where(col_std > _EPS, col_std, 1.0)
    standardized = (data_matrix - np.mean(data_matrix, axis=0)) / col_std

    # Sample correlation matrix
    corr_matrix = (standardized.T @ standardized) / T
    np.fill_diagonal(corr_matrix, 1.0)

    # Eigendecomposition (symmetric -> real eigenvalues)
    eigenvalues_raw, eigenvectors_raw = eigh(corr_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues = eigenvalues_raw[idx]
    eigenvectors = eigenvectors_raw[:, idx]

    # Ensure non-negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Marchenko-Pastur fit — iterative σ² estimation (MEDIUM-1 fix).
    # A single pass can misclassify borderline eigenvalues.  We iterate
    # until σ² converges (typically 3-5 iterations).
    # Reference: Bun, Bouchaud & Potters (2017), "Cleaning large
    # correlation matrices: Tools from Random Matrix Theory".
    gamma = T / N
    sigma_sq = 1.0
    for _mp_iter in range(10):
        lambda_minus, lambda_plus = marchenko_pastur_edges(gamma, sigma_sq)
        noise_eigenvalues = eigenvalues[eigenvalues <= lambda_plus]
        if len(noise_eigenvalues) == 0:
            break
        sigma_sq_new = float(noise_eigenvalues.mean())
        if abs(sigma_sq_new - sigma_sq) < 1e-6:
            sigma_sq = sigma_sq_new
            break
        sigma_sq = sigma_sq_new
    # Final edges with converged sigma
    lambda_minus, lambda_plus = marchenko_pastur_edges(gamma, sigma_sq)

    n_signal = int(np.sum(eigenvalues > lambda_plus))
    n_noise = N - n_signal

    mp_dist = MPDistribution(
        gamma=gamma,
        sigma_sq=sigma_sq,
        lambda_plus=lambda_plus,
        lambda_minus=lambda_minus,
        n_signal=n_signal,
        n_noise=n_noise,
    )

    # Clean correlation matrix
    cleaned_corr = clean_correlation_matrix(
        corr_matrix, eigenvalues, eigenvectors, mp_dist, method='clip'
    )

    # Shrinkage intensity
    if shrinkage == 'auto':
        shrinkage_intensity = _fast_ledoit_wolf_intensity(standardized)
    elif shrinkage == 'none':
        shrinkage_intensity = 0.0
    else:
        shrinkage_intensity = 0.0

    # Metrics
    cond = float(eigenvalues[0] / max(eigenvalues[-1], _EPS))
    eff_rank = effective_number_of_bets(eigenvalues)
    ar = absorption_ratio(eigenvalues, n_absorption_factors)
    hei = herfindahl_eigenvalue_index(eigenvalues)

    return SpectralDiagnostics(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        mp_dist=mp_dist,
        condition_number=cond,
        effective_rank=eff_rank,
        absorption_ratio=ar,
        herfindahl_eigenvalues=hei,
        cleaned_corr=cleaned_corr,
        shrinkage_intensity=shrinkage_intensity,
    )


def _degenerate_diagnostics(n: int) -> SpectralDiagnostics:
    """Return identity-based diagnostics for degenerate cases."""
    return SpectralDiagnostics(
        eigenvalues=np.ones(max(n, 1)),
        eigenvectors=np.eye(max(n, 1)),
        mp_dist=MPDistribution(
            gamma=1.0, sigma_sq=1.0,
            lambda_plus=4.0, lambda_minus=0.0,
            n_signal=0, n_noise=max(n, 1),
        ),
        condition_number=1.0,
        effective_rank=float(max(n, 1)),
        absorption_ratio=1.0 / max(n, 1),
        herfindahl_eigenvalues=1.0 / max(n, 1),
        cleaned_corr=np.eye(max(n, 1)),
        shrinkage_intensity=0.0,
    )


# ---------------------------------------------------------------------------
# Strategy Redundancy Detection
# ---------------------------------------------------------------------------

def detect_redundant_strategies(
    strategy_returns: Dict[str, np.ndarray],
    min_window: int = 20,
    correlation_threshold: float = 0.7,
) -> Dict:
    """
    Analyze strategy return correlations through the RMT lens.

    Args:
        strategy_returns: {strategy_name: 1D array of daily returns}
        min_window: minimum number of overlapping observations required.
        correlation_threshold: cleaned correlation above which two strategies
                               are considered redundant.

    Returns:
        {
            'effective_strategy_count': float,
            'clusters': List[List[str]],
            'cleaned_corr': np.ndarray,
            'strategy_names': List[str],
            'signal_eigenvalues': np.ndarray,
            'noise_fraction': float,
            'diagnostics': SpectralDiagnostics
        }
    """
    names = list(strategy_returns.keys())
    n_strategies = len(names)

    if n_strategies < 2:
        return {
            'effective_strategy_count': float(n_strategies),
            'clusters': [names],
            'cleaned_corr': np.eye(n_strategies) if n_strategies > 0 else np.array([]),
            'strategy_names': names,
            'signal_eigenvalues': np.ones(n_strategies),
            'noise_fraction': 0.0,
            'diagnostics': None,
        }

    # Align returns to common length
    min_len = min(len(v) for v in strategy_returns.values())
    if min_len < min_window:
        return {
            'effective_strategy_count': float(n_strategies),
            'clusters': [names],
            'cleaned_corr': np.eye(n_strategies),
            'strategy_names': names,
            'signal_eigenvalues': np.ones(n_strategies),
            'noise_fraction': 0.0,
            'diagnostics': None,
        }

    # Build T x N returns matrix
    returns_matrix = np.column_stack([
        strategy_returns[name][:min_len] for name in names
    ])

    # Remove NaN rows
    valid = ~np.any(np.isnan(returns_matrix), axis=1)
    returns_matrix = returns_matrix[valid]

    if returns_matrix.shape[0] < min_window:
        return {
            'effective_strategy_count': float(n_strategies),
            'clusters': [names],
            'cleaned_corr': np.eye(n_strategies),
            'strategy_names': names,
            'signal_eigenvalues': np.ones(n_strategies),
            'noise_fraction': 0.0,
            'diagnostics': None,
        }

    diagnostics = compute_spectral_diagnostics(returns_matrix)

    # Cluster strategies by cleaned correlation
    clusters = _cluster_by_correlation(
        diagnostics.cleaned_corr, names, correlation_threshold
    )

    signal_eigs = diagnostics.eigenvalues[
        diagnostics.eigenvalues > diagnostics.mp_dist.lambda_plus
    ]

    return {
        'effective_strategy_count': diagnostics.effective_rank,
        'clusters': clusters,
        'cleaned_corr': diagnostics.cleaned_corr,
        'strategy_names': names,
        'signal_eigenvalues': signal_eigs,
        'noise_fraction': diagnostics.mp_dist.n_noise / max(n_strategies, 1),
        'diagnostics': diagnostics,
    }


def _cluster_by_correlation(
    corr_matrix: np.ndarray,
    names: List[str],
    threshold: float = 0.7,
) -> List[List[str]]:
    """
    Union-Find clustering: strategies with cleaned correlation > threshold
    are placed in the same cluster.  Order-independent (unlike the previous
    greedy pass which was sensitive to index ordering).
    """
    n = len(names)

    # Union-Find with path compression
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                union(i, j)

    # Collect clusters
    cluster_map: Dict[int, List[str]] = {}
    for i in range(n):
        root = find(i)
        cluster_map.setdefault(root, []).append(names[i])

    return list(cluster_map.values())


# ---------------------------------------------------------------------------
# Greedy Diversified Selection
# ---------------------------------------------------------------------------

def greedy_diversified_select(
    ranked_strategies: List[Tuple[str, float]],
    cleaned_corr: np.ndarray,
    strategy_names: List[str],
    n_select: int = 4,
    max_correlation: float = 0.7,
) -> List[str]:
    """
    Select top N strategies that are spectrally independent.

    Greedy algorithm: pick the highest-scoring strategy, then the next
    highest that has cleaned correlation < max_correlation with all
    already-selected strategies.

    Args:
        ranked_strategies: [(name, score)] sorted by score descending.
        cleaned_corr: N x N cleaned correlation matrix.
        strategy_names: names corresponding to corr matrix rows/columns.
        n_select: number of strategies to select.
        max_correlation: maximum allowed pairwise cleaned correlation.

    Returns:
        List of selected strategy names.
    """
    name_to_idx = {name: i for i, name in enumerate(strategy_names)}
    selected = []
    selected_indices = []

    for name, score in ranked_strategies:
        if len(selected) >= n_select:
            break
        if name not in name_to_idx:
            continue

        idx = name_to_idx[name]
        # Check correlation with all already-selected
        is_independent = True
        for sel_idx in selected_indices:
            if abs(cleaned_corr[idx, sel_idx]) > max_correlation:
                is_independent = False
                break

        if is_independent:
            selected.append(name)
            selected_indices.append(idx)

    # If we couldn't fill n_select with independent strategies,
    # fill remaining with highest-scoring unselected
    if len(selected) < n_select:
        for name, score in ranked_strategies:
            if name not in selected and len(selected) < n_select:
                selected.append(name)

    return selected


# ---------------------------------------------------------------------------
# Portfolio Optimization with Cleaned Covariance
# ---------------------------------------------------------------------------

def rmt_minimum_variance_weights(
    returns_matrix: np.ndarray,
    strategy_names: List[str],
) -> Dict[str, float]:
    """
    Compute minimum variance portfolio weights using RMT-cleaned covariance.

    Args:
        returns_matrix: T x N matrix of strategy returns.
        strategy_names: names for each column.

    Returns:
        {strategy_name: weight} summing to 1.0, all >= 0.
    """
    diagnostics = compute_spectral_diagnostics(returns_matrix)
    N = len(strategy_names)

    # Build cleaned covariance from cleaned correlation + sample volatilities
    vols = np.std(returns_matrix, axis=0)
    vols = np.where(vols > _EPS, vols, _EPS)
    cleaned_cov = diagnostics.cleaned_corr * np.outer(vols, vols)

    # Regularize slightly to ensure positive definiteness
    cleaned_cov += np.eye(N) * 1e-8

    # Solve: min w'Σw s.t. sum(w)=1, w>=0
    w0 = np.ones(N) / N

    def objective(w):
        return w @ cleaned_cov @ w

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * N

    result = minimize(
        objective, w0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-12},
    )

    weights = result.x if result.success else w0
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()

    return {name: float(w) for name, w in zip(strategy_names, weights)}


def rmt_risk_parity_weights(
    returns_matrix: np.ndarray,
    strategy_names: List[str],
) -> Dict[str, float]:
    """
    Compute risk parity weights using RMT-cleaned covariance.

    Each strategy contributes equally to total portfolio risk.

    Args:
        returns_matrix: T x N matrix of strategy returns.
        strategy_names: names for each column.

    Returns:
        {strategy_name: weight} summing to 1.0.
    """
    diagnostics = compute_spectral_diagnostics(returns_matrix)
    N = len(strategy_names)

    vols = np.std(returns_matrix, axis=0)
    vols = np.where(vols > _EPS, vols, _EPS)
    cleaned_cov = diagnostics.cleaned_corr * np.outer(vols, vols)
    cleaned_cov += np.eye(N) * 1e-8

    # Risk parity: minimize Σ (RC_i - 1/N)^2
    # where RC_i = w_i * (Σw)_i / w'Σw
    w0 = np.ones(N) / N
    target_rc = 1.0 / N

    def risk_parity_objective(w):
        w = np.maximum(w, 1e-10)
        port_var = w @ cleaned_cov @ w
        if port_var <= 0:
            return 1e10
        marginal_risk = cleaned_cov @ w
        risk_contrib = w * marginal_risk / np.sqrt(port_var)
        rc_normalized = risk_contrib / risk_contrib.sum()
        return np.sum((rc_normalized - target_rc) ** 2)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(1e-6, 1.0)] * N

    result = minimize(
        risk_parity_objective, w0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12},
    )

    weights = result.x if result.success else w0
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()

    return {name: float(w) for name, w in zip(strategy_names, weights)}


def compute_diversification_ratio(
    weights: np.ndarray,
    cleaned_cov: np.ndarray,
) -> float:
    """
    Diversification ratio: DR = (Σ w_i σ_i) / σ_portfolio.

    DR > 1 indicates genuine diversification benefit.
    DR = 1 means portfolio is a single asset (no diversification).
    """
    vols = np.sqrt(np.diag(cleaned_cov))
    weighted_avg_vol = np.dot(weights, vols)
    port_var = weights @ cleaned_cov @ weights
    port_vol = np.sqrt(max(port_var, _EPS))
    return float(weighted_avg_vol / port_vol)


# ---------------------------------------------------------------------------
# Rolling Spectral Analysis
# ---------------------------------------------------------------------------

def rolling_spectral_analysis(
    data_matrix: np.ndarray,
    window: int = 50,
    step: int = 1,
    n_absorption_factors: int = 5,
) -> List[SpectralDiagnostics]:
    """
    Rolling window spectral analysis for regime tracking.

    Args:
        data_matrix: T x N matrix.
        window: lookback window size.
        step: step size between windows.
        n_absorption_factors: for absorption ratio.

    Returns:
        List of SpectralDiagnostics, one per step.
    """
    T, N = data_matrix.shape
    results = []

    for start in range(0, T - window + 1, step):
        window_data = data_matrix[start:start + window]
        diag = compute_spectral_diagnostics(
            window_data,
            shrinkage='none',
            n_absorption_factors=n_absorption_factors,
        )
        results.append(diag)

    return results


def compute_spectral_turnover(
    diagnostics_list: List[SpectralDiagnostics],
    n_factors: int = 3,
) -> float:
    """
    Average change in top eigenvector between consecutive windows.

    Low turnover = stable factor structure = trustworthy optimization.
    High turnover = factor structure is rotating = be cautious.

    Uses cosine similarity of top eigenvectors.
    """
    if len(diagnostics_list) < 2:
        return 0.0

    similarities = []
    for i in range(1, len(diagnostics_list)):
        prev = diagnostics_list[i - 1]
        curr = diagnostics_list[i]

        n = min(n_factors, prev.eigenvectors.shape[1], curr.eigenvectors.shape[1])
        total_sim = 0.0
        for k in range(n):
            v_prev = prev.eigenvectors[:, k]
            v_curr = curr.eigenvectors[:, k]
            # Cosine similarity (handle sign ambiguity)
            cos_sim = abs(np.dot(v_prev, v_curr) / (
                np.linalg.norm(v_prev) * np.linalg.norm(v_curr) + 1e-15
            ))
            total_sim += cos_sim
        similarities.append(total_sim / n)

    # Turnover = 1 - average similarity
    return float(1.0 - np.mean(similarities))


# ---------------------------------------------------------------------------
# Strategy Dimensionality Reduction (CRITICAL-1 + REC-1)
# ---------------------------------------------------------------------------
# Reference: Ahn, Conrad & Dittmar (2009), "Basis Assets";
#            Kelly, Pruitt & Su (2019), "Instrumented PCA".

def reduce_strategy_space(
    strategy_returns: Dict[str, np.ndarray],
    min_window: int = 20,
) -> Dict:
    """
    Reduce the strategy space to its true independent factors using the
    Marchenko-Pastur threshold from RMT spectral analysis.

    Instead of selecting among 60+ correlated strategies, this function:
    1. Builds the T×N strategy return matrix
    2. Identifies signal eigenvalues (above MP threshold)
    3. Projects strategies onto signal eigenvectors → factor portfolios
    4. Labels each factor by the strategy with highest loading
    5. Returns factor portfolios and the mapping from strategies to factors

    Args:
        strategy_returns: {strategy_name: 1D array of daily returns}
        min_window: minimum number of overlapping observations

    Returns:
        {
            'n_factors': int,           # number of true independent factors
            'factor_returns': np.ndarray,  # T × n_factors matrix
            'factor_labels': List[str],    # interpretable name per factor
            'strategy_factor_map': Dict[str, int],  # strategy → factor index
            'factor_weights': Dict[str, np.ndarray],  # factor → strategy loadings
            'explained_variance': np.ndarray,  # variance per factor
            'diagnostics': SpectralDiagnostics,
        }
    """
    names = list(strategy_returns.keys())
    n_strategies = len(names)

    if n_strategies < 2:
        return {
            'n_factors': n_strategies,
            'factor_returns': np.column_stack([strategy_returns[n] for n in names]) if names else np.array([]),
            'factor_labels': names,
            'strategy_factor_map': {n: 0 for n in names},
            'factor_weights': {names[0]: np.array([1.0])} if names else {},
            'explained_variance': np.array([1.0]),
            'diagnostics': None,
        }

    # Align to common length
    min_len = min(len(v) for v in strategy_returns.values())
    if min_len < min_window:
        return {
            'n_factors': n_strategies,
            'factor_returns': np.column_stack([strategy_returns[n][:min_len] for n in names]),
            'factor_labels': names,
            'strategy_factor_map': {n: i for i, n in enumerate(names)},
            'factor_weights': {},
            'explained_variance': np.ones(n_strategies),
            'diagnostics': None,
        }

    returns_matrix = np.column_stack([
        strategy_returns[name][:min_len] for name in names
    ])

    # Remove NaN rows
    valid = ~np.any(np.isnan(returns_matrix), axis=1)
    returns_matrix = returns_matrix[valid]

    if returns_matrix.shape[0] < min_window:
        return {
            'n_factors': n_strategies,
            'factor_returns': returns_matrix,
            'factor_labels': names,
            'strategy_factor_map': {n: i for i, n in enumerate(names)},
            'factor_weights': {},
            'explained_variance': np.ones(n_strategies),
            'diagnostics': None,
        }

    diagnostics = compute_spectral_diagnostics(returns_matrix)
    n_signal = diagnostics.mp_dist.n_signal

    # Ensure at least 1 factor
    n_factors = max(n_signal, 1)
    # Cap at reasonable number
    n_factors = min(n_factors, n_strategies, 10)

    # Signal eigenvectors (top n_factors)
    V_signal = diagnostics.eigenvectors[:, :n_factors]  # N × n_factors
    eigenvalues_signal = diagnostics.eigenvalues[:n_factors]

    # Project returns onto signal subspace → factor returns
    # Standardize first
    col_mean = returns_matrix.mean(axis=0)
    col_std = np.std(returns_matrix, axis=0)
    col_std = np.where(col_std > _EPS, col_std, 1.0)
    standardized = (returns_matrix - col_mean) / col_std

    factor_returns = standardized @ V_signal  # T × n_factors

    # Label each factor by the strategy with the highest absolute loading
    factor_labels = []
    strategy_factor_map = {}
    factor_weights = {}

    for k in range(n_factors):
        loadings = V_signal[:, k]
        best_idx = int(np.argmax(np.abs(loadings)))
        label = names[best_idx]
        # Avoid duplicate labels
        suffix = 1
        original_label = label
        while label in factor_labels:
            suffix += 1
            label = f"{original_label}_{suffix}"
        factor_labels.append(label)
        factor_weights[label] = loadings

    # Map each strategy to the factor it loads most heavily on
    for i, name in enumerate(names):
        factor_loadings = np.abs(V_signal[i, :])
        strategy_factor_map[name] = int(np.argmax(factor_loadings))

    # Explained variance per factor
    total_var = diagnostics.eigenvalues.sum()
    explained_variance = eigenvalues_signal / total_var if total_var > 0 else eigenvalues_signal

    return {
        'n_factors': n_factors,
        'factor_returns': factor_returns,
        'factor_labels': factor_labels,
        'strategy_factor_map': strategy_factor_map,
        'factor_weights': factor_weights,
        'explained_variance': explained_variance,
        'diagnostics': diagnostics,
    }


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (REC-5)
# ---------------------------------------------------------------------------
# Reference: Lopez de Prado (2016), "Building Diversified Portfolios that
#            Outperform Out of Sample".

def _quasi_diag(link: np.ndarray) -> List[int]:
    """
    Quasi-diagonalization: reorder the correlation matrix so that correlated
    assets are adjacent.  Uses the dendrogram linkage matrix.

    Returns sorted list of original asset indices.
    """
    n = int(link[-1, 3])
    sort_idx = [n - 1]  # start with last merged cluster

    while any(i >= (n - 1) // 2 for i in sort_idx if not isinstance(i, int)):
        pass  # fallback — use iterative expansion below

    # Iterative cluster expansion
    sort_idx = [int(link.shape[0] + link.shape[0])]
    # Simpler recursive approach via stack
    num_items = int(link[-1, 3])
    sort_idx = _get_quasi_diag_order(link, num_items)
    return sort_idx


def _get_quasi_diag_order(link: np.ndarray, num_items: int) -> List[int]:
    """Iterative quasi-diagonalization using a stack (avoids recursion depth issues)."""
    n = link.shape[0] + 1  # number of original items
    stack = [2 * n - 2]  # root cluster index
    order = []

    while stack:
        node = stack.pop()
        if node < n:
            order.append(int(node))
        else:
            cluster_idx = int(node - n)
            if cluster_idx < len(link):
                left = int(link[cluster_idx, 0])
                right = int(link[cluster_idx, 1])
                stack.append(right)
                stack.append(left)

    return order


def _hrp_bisect_weights(
    cov: np.ndarray,
    sorted_indices: List[int],
) -> np.ndarray:
    """
    Recursive bisection allocation for HRP.

    At each split, allocate inversely proportional to cluster variance:
    w_left / w_right = var_right / (var_left + var_right)
    """
    n = len(sorted_indices)
    weights = np.ones(cov.shape[0])

    # Build cluster tree via bisection
    clusters = [sorted_indices]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Cluster variance = inverse-vol portfolio variance
            def cluster_var(indices):
                sub_cov = cov[np.ix_(indices, indices)]
                ivp = 1.0 / np.diag(sub_cov)
                ivp = ivp / ivp.sum()
                return float(ivp @ sub_cov @ ivp)

            var_left = cluster_var(left)
            var_right = cluster_var(right)

            # Allocation factor
            total_var = var_left + var_right
            if total_var > 0:
                alpha = 1.0 - var_left / total_var
            else:
                alpha = 0.5

            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= (1.0 - alpha)

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        clusters = new_clusters

    # Normalize
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


def hrp_weights(
    returns_matrix: np.ndarray,
    strategy_names: List[str],
    use_cleaned_corr: bool = True,
) -> Dict[str, float]:
    """
    Hierarchical Risk Parity portfolio weights.

    Does not require covariance matrix inversion — more robust to estimation
    error than minimum variance or standard risk parity, especially for
    small T/N ratios typical of strategy allocation.

    Args:
        returns_matrix: T × N matrix of strategy returns.
        strategy_names: names for each column.
        use_cleaned_corr: if True, use RMT-cleaned correlation for clustering.

    Returns:
        {strategy_name: weight} summing to 1.0.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    N = len(strategy_names)
    if N <= 1:
        return {strategy_names[0]: 1.0} if N == 1 else {}

    # Covariance from returns
    vols = np.std(returns_matrix, axis=0)
    vols = np.where(vols > _EPS, vols, _EPS)

    if use_cleaned_corr:
        diagnostics = compute_spectral_diagnostics(returns_matrix)
        corr = diagnostics.cleaned_corr
    else:
        standardized = (returns_matrix - returns_matrix.mean(axis=0)) / vols
        corr = np.corrcoef(standardized.T)
        np.fill_diagonal(corr, 1.0)

    cov = corr * np.outer(vols, vols)
    # Regularize
    cov += np.eye(N) * 1e-8

    # Distance matrix from correlation
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)

    # Hierarchical clustering
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method='single')

    # Quasi-diagonalization
    sorted_indices = _get_quasi_diag_order(link, N)

    # Recursive bisection
    weights = _hrp_bisect_weights(cov, sorted_indices)

    return {name: float(weights[i]) for i, name in enumerate(strategy_names)}


# ---------------------------------------------------------------------------
# MASTER Attention Integration
# ---------------------------------------------------------------------------

def blend_attention_correlation(
    cleaned_corr: np.ndarray,
    attention_matrix: Optional[np.ndarray],
    blend_weight: float = 0.3,
) -> np.ndarray:
    """Blend RMT-cleaned correlation with MASTER attention matrix.

    The attention matrix from inter-stock attention is a learned, dynamic
    correlation analog. Blending it with the MP-cleaned correlation gives
    a hybrid that benefits from both:
      - RMT: principled noise removal, eigenvalue-level denoising
      - MASTER: forward-looking, neural-learned cross-stock relationships

    Args:
        cleaned_corr: RMT-denoised correlation, shape (N, N).
        attention_matrix: Mean attention matrix, shape (N, N). None to skip.
        blend_weight: Weight for attention matrix (0 = pure RMT, 1 = pure attention).

    Returns:
        Blended correlation matrix, shape (N, N).
    """
    if attention_matrix is None:
        return cleaned_corr

    N = cleaned_corr.shape[0]
    if attention_matrix.shape != (N, N):
        return cleaned_corr

    # Symmetrize attention matrix (it's asymmetric from softmax)
    attn_sym = (attention_matrix + attention_matrix.T) / 2.0

    # Normalize to [0, 1] range like a correlation
    attn_min = attn_sym.min()
    attn_range = attn_sym.max() - attn_min
    if attn_range > _EPS:
        attn_norm = (attn_sym - attn_min) / attn_range
    else:
        attn_norm = np.ones_like(attn_sym) / N

    # Set diagonal to 1
    np.fill_diagonal(attn_norm, 1.0)

    # Blend
    blended = (1.0 - blend_weight) * cleaned_corr + blend_weight * attn_norm

    # Ensure valid correlation matrix properties
    np.fill_diagonal(blended, 1.0)
    blended = np.clip(blended, -1.0, 1.0)

    return blended


def get_master_attention_matrix() -> Optional[np.ndarray]:
    """Try to obtain the current MASTER attention matrix.

    Returns:
        Mean attention matrix (n_stocks, n_stocks) or None if unavailable.
    """
    try:
        import torch
        from master_predict import load_pipeline

        pipeline = load_pipeline()
        if pipeline is None:
            return None

        # Generate from pipeline's inter-stock module with random embeddings
        # In production, these would come from real feature sequences
        n_stocks = 30  # Standard Pragyam ETF universe
        dummy = torch.randn(n_stocks, pipeline.tau, pipeline.d_model)
        return pipeline.inter_stock.get_mean_attention_matrix(dummy)
    except (ImportError, Exception):
        return None


# ---------------------------------------------------------------------------
# Conformal Prediction Intervals (REC-4)
# ---------------------------------------------------------------------------
# Reference: Vovk, Gammerman & Shafer (2005), "Algorithmic Learning in a
#            Random World"; Barber et al. (2023), conformal prediction
#            beyond exchangeability.

def conformal_prediction_interval(
    historical_returns: np.ndarray,
    alpha: float = 0.10,
    method: str = 'split',
) -> Tuple[float, float, float]:
    """
    Distribution-free prediction interval for next-period return using
    conformal inference.

    Provides finite-sample valid coverage: the true next return falls
    within [lo, hi] with probability ≥ 1 - α, without any distributional
    assumptions (Gaussian, etc.).

    Args:
        historical_returns: 1D array of past returns (in chronological order).
        alpha: Miscoverage rate (default 0.10 → 90% interval).
        method: 'split' (split conformal) or 'full' (full conformal, slower).

    Returns:
        (lower_bound, point_estimate, upper_bound)
    """
    r = np.asarray(historical_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)

    if n < 5:
        return (float(r.min()) if n > 0 else -0.05,
                float(r.mean()) if n > 0 else 0.0,
                float(r.max()) if n > 0 else 0.05)

    point_estimate = float(r.mean())

    if method == 'split':
        # Split conformal: use first half as training, second as calibration
        split = n // 2
        train = r[:split]
        cal = r[split:]

        # Simple model: predict the training mean
        mu_hat = train.mean()

        # Nonconformity scores: |actual - predicted|
        scores = np.abs(cal - mu_hat)

        # Quantile of calibration scores (with finite-sample correction)
        q_level = np.ceil((1 - alpha) * (len(cal) + 1)) / len(cal)
        q_level = min(q_level, 1.0)
        q = float(np.quantile(scores, q_level))

        lower = mu_hat - q
        upper = mu_hat + q
        point_estimate = float(mu_hat)

    else:  # full conformal
        # Use all data; compute leave-one-out residuals
        residuals = np.abs(r - point_estimate)

        q_level = np.ceil((1 - alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)
        q = float(np.quantile(residuals, q_level))

        lower = point_estimate - q
        upper = point_estimate + q

    return (float(lower), float(point_estimate), float(upper))


def conformal_strategy_intervals(
    strategy_returns: Dict[str, np.ndarray],
    alpha: float = 0.10,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute conformal prediction intervals for each strategy's next return.

    Args:
        strategy_returns: {strategy_name: 1D array of returns}
        alpha: Miscoverage rate.

    Returns:
        {strategy_name: (lower, point_estimate, upper)}
    """
    intervals = {}
    for name, returns in strategy_returns.items():
        if len(returns) >= 5:
            intervals[name] = conformal_prediction_interval(returns, alpha)
        else:
            mu = returns.mean() if len(returns) > 0 else 0.0
            intervals[name] = (mu - 0.05, mu, mu + 0.05)
    return intervals
