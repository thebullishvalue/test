"""Configuration objects for the Fair Value Engine."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

FactorMethod = Literal["PCA", "ICA", "FactorAnalysis", "Autoencoder"]
SmoothMethod = Literal["None", "EMA", "Kalman"]

# Regime labels and their presentation metadata.
REGIME_META: dict[str, dict[str, str]] = {
    "RISK-ON":  {"color": "#2fd08c", "icon": "▲", "desc": "Broad advance, contained volatility"},
    "TREND":    {"color": "#38c7dc", "icon": "↗", "desc": "Persistent directional drift"},
    "MEAN-REV": {"color": "#b7a4f3", "icon": "⇄", "desc": "High dispersion, low persistence"},
    "HIGH-VOL": {"color": "#f2b544", "icon": "≈", "desc": "Elevated volatility, unstable betas"},
    "RISK-OFF": {"color": "#ff5d6c", "icon": "▼", "desc": "Correlated drawdown, flight to quality"},
}
REGIME_NAMES: list[str] = list(REGIME_META.keys())


@dataclass
class EngineConfig:
    """Everything that determines a fair-value calibration.

    Grouped by concern: universe -> factor model -> regression -> fair value
    -> oscillator. Defaults are the ones we consider statistically defensible
    for a daily-frequency, liquid target asset.
    """

    # --- target & universe -------------------------------------------------
    target: str = "QQQ"
    tier: str = "core"

    # --- latent factor model ----------------------------------------------
    method: FactorMethod = "PCA"
    n_factors: int = 8

    # --- walk-forward regression ------------------------------------------
    lookback: int = 252          # calibration window length (sessions)
    refit_every: int = 21        # sessions between recalibrations
    n_peers: int = 12            # orthogonal peer instruments retained
    ridge_alpha: float = 10.0    # L2 penalty; ignored when auto_alpha is set
    auto_alpha: bool = True      # pick alpha by generalised cross-validation
    halflife_days: float = 63.0  # sample-weight decay half-life; 0 => uniform
    regime_weighting: bool = True

    # Instruments whose window correlation with the target exceeds this are
    # dropped from the explanatory set. A near-perfect proxy (QQQ vs ^NDX)
    # would drive fair value to the price itself and the oscillator to noise.
    exclude_corr_above: float = 0.99
    exclude_same_class: bool = False

    # --- fair value --------------------------------------------------------
    fv_horizon: int = 63         # anchor horizon H for the fair-value path
    mtf_horizons: tuple[int, ...] = (21, 63, 126)

    # --- oscillator --------------------------------------------------------
    norm_window: int = 126       # adaptive normalisation window
    robust_norm: bool = True     # median/MAD instead of mean/std
    saturation: float = 2.5      # z at which the oscillator reaches ~±76
    smoothing: SmoothMethod = "Kalman"
    smooth_span: int = 6
    threshold_q: float = 0.12    # dynamic OB/OS quantile
    threshold_window: int = 504  # rolling window for threshold quantiles
    mr_horizon: int = 10         # mean-reversion probability horizon (sessions)

    # --- regimes -----------------------------------------------------------
    n_regimes: int = 5

    def key(self) -> tuple:
        """Hashable identity for caching."""
        d = asdict(self)
        return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                            for k, v in d.items()))


@dataclass
class BacktestConfig:
    """Parameters for replaying oscillator signals."""

    allow_short: bool = True
    exit_level: float = 10.0       # |FVO| at/below which a position is closed
    max_hold: int = 21             # maximum holding period in sessions
    cost_bps: float = 2.0          # round-trip cost in basis points, per side
    confidence_floor: float = 0.0  # skip entries below this confidence score
    size_by_confidence: bool = False


@dataclass
class DataConfig:
    """Data acquisition settings.

    Cache lifetimes, retention and namespacing live in :mod:`fve.cache`; this
    covers only what a caller would reasonably vary per run.
    """

    years: int = 8
    min_coverage: float = 0.60   # fraction of calendar an instrument must cover
    max_ffill: int = 5           # max consecutive sessions to forward fill
    chunk_size: int = 45         # tickers per yfinance request
    synthetic: bool = False      # bypass the network, use the simulator
    seed: int = 7


@dataclass
class RunArtifacts:
    """Container for everything a calibration produces (kept loosely typed)."""

    data: dict = field(default_factory=dict)

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)
