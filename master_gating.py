"""
MASTER × Pragyam — Phase 1: Market-Guided Feature Gating
==========================================================

Implements the market-guided gating mechanism from the MASTER paper
(Li et al., AAAI-24, Section: Market-Guided Gating).

    α(mτ) = F · softmax_β(Wα · mτ + bα)
    x̃ = α ⊙ x

The gating network takes a market status vector mτ and produces
per-feature scaling coefficients that dynamically rescale indicator
values based on the current market regime. This achieves automatic
feature selection — features that are effective in the current regime
get amplified, while irrelevant features get suppressed.

Author: Hemrek Capital
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, List
import logging
import os

logger = logging.getLogger("master_gating")

# Default gating temperature: higher = weaker gating.
# β=8 is appropriate for the 30-ETF universe (predictable, like CSI300).
DEFAULT_BETA = 8.0

# Default path for saving/loading trained gating model
DEFAULT_MODEL_PATH = "master_gating_model.pt"


class MarketGatingNetwork(nn.Module):
    """Market-guided feature gating network.

    Transforms a market status vector mτ into per-feature scaling
    coefficients α via a linear layer + temperature-scaled softmax.

    Architecture:
        Linear(market_dim, n_features) → softmax(·/β) → ×F

    The coefficients α sum to F (number of features), so the average
    scaling is 1.0. Features deemed effective by the gating mechanism
    receive coefficients > 1 (amplified), while ineffective features
    receive coefficients < 1 (suppressed).

    Args:
        market_dim: Dimensionality of the market status vector mτ.
        n_features: Number of numeric indicator columns (F).
        beta: Temperature for softmax. Smaller = sharper gating.
        dropout: Dropout probability applied to the linear layer.
    """

    def __init__(
        self,
        market_dim: int = 21,
        n_features: int = 25,
        beta: float = DEFAULT_BETA,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.market_dim = market_dim
        self.n_features = n_features
        self.beta = beta

        self.linear = nn.Linear(market_dim, n_features)
        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights near zero for gentle initial gating
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, m_tau: torch.Tensor) -> torch.Tensor:
        """Compute gating coefficients α from market status vector.

        Args:
            m_tau: Market status vector of shape (batch, market_dim) or (market_dim,).

        Returns:
            Scaling coefficients α of shape (batch, n_features) or (n_features,).
            Each coefficient ∈ (0, F) and the vector sums to F.
        """
        squeeze = m_tau.dim() == 1
        if squeeze:
            m_tau = m_tau.unsqueeze(0)

        h = self.dropout(self.linear(m_tau))  # (batch, n_features)

        # Temperature-scaled softmax: softer distribution when β is large
        alpha = torch.softmax(h / self.beta, dim=-1)  # sums to 1

        # Scale by F so average coefficient = 1.0
        alpha = alpha * self.n_features

        if squeeze:
            alpha = alpha.squeeze(0)

        return alpha

    def gate_features(
        self,
        m_tau_np: np.ndarray,
        feature_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Apply gating to a snapshot DataFrame's numeric columns.

        This is the primary integration point with backdata.py. It takes
        a market status vector and a DataFrame of indicators, applies the
        learned gating, and returns a new DataFrame with gated values.
        Column names and count are preserved — only values change.

        Args:
            m_tau_np: Market status vector as numpy array, shape (market_dim,).
            feature_df: DataFrame with indicator columns for all ETFs.
            feature_cols: List of numeric column names to gate.

        Returns:
            New DataFrame with gated feature values. Non-numeric and
            non-gated columns are unchanged.
        """
        self.eval()

        # Convert market vector to tensor
        m_tau = torch.tensor(m_tau_np, dtype=torch.float32)

        with torch.no_grad():
            alpha = self.forward(m_tau).numpy()  # shape (n_features,)

        # Apply Hadamard product: x̃ = α ⊙ x
        gated_df = feature_df.copy()
        for i, col in enumerate(feature_cols):
            if col in gated_df.columns and i < len(alpha):
                gated_df[col] = gated_df[col] * alpha[i]

        return gated_df

    def get_gating_coefficients(self, m_tau_np: np.ndarray) -> np.ndarray:
        """Get raw gating coefficients for inspection/visualization.

        Args:
            m_tau_np: Market status vector as numpy array.

        Returns:
            numpy array of gating coefficients, shape (n_features,).
        """
        self.eval()
        m_tau = torch.tensor(m_tau_np, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(m_tau).numpy()


def save_gating_model(
    model: MarketGatingNetwork,
    path: str = DEFAULT_MODEL_PATH,
) -> None:
    """Save the gating model checkpoint.

    Args:
        model: Trained MarketGatingNetwork instance.
        path: File path for the checkpoint.
    """
    checkpoint = {
        'state_dict': model.state_dict(),
        'market_dim': model.market_dim,
        'n_features': model.n_features,
        'beta': model.beta,
    }
    torch.save(checkpoint, path)
    logger.info("Gating model saved to %s", path)


def load_gating_model(
    path: str = DEFAULT_MODEL_PATH,
) -> Optional[MarketGatingNetwork]:
    """Load a trained gating model from checkpoint.

    Args:
        path: File path of the checkpoint.

    Returns:
        MarketGatingNetwork instance, or None if file doesn't exist.
    """
    if not os.path.exists(path):
        logger.debug("No gating model found at %s", path)
        return None

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = MarketGatingNetwork(
            market_dim=checkpoint['market_dim'],
            n_features=checkpoint['n_features'],
            beta=checkpoint['beta'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        logger.info(
            "Gating model loaded from %s (market_dim=%d, n_features=%d, β=%.1f)",
            path, model.market_dim, model.n_features, model.beta,
        )
        return model
    except Exception as e:
        logger.error("Failed to load gating model from %s: %s", path, e)
        return None
