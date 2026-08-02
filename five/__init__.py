"""
FVE — Fair Value Engine.

A cross-sectional, market-relative valuation model. The target asset's fair
value is the price path implied by a 200+ instrument cross-asset universe,
compressed into latent factors plus orthogonal peers and mapped through a
walk-forward, regime-aware regression. The deviation between market price and
that implied path is published as a bounded, adaptively-normalised oscillator.
"""
from .config import BacktestConfig, DataConfig, EngineConfig, REGIME_META, REGIME_NAMES

__version__ = "3.0.0"

__all__ = [
    "EngineConfig", "DataConfig", "BacktestConfig",
    "REGIME_META", "REGIME_NAMES", "__version__",
]
