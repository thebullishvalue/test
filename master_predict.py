"""
MASTER × Pragyam — Phase 4: Full MASTER Pipeline + MASTERStrategy
====================================================================

Assembles the complete MASTER pipeline:
  1. Market-Guided Gating (master_gating.py)
  2. Intra-Stock Temporal Encoder (master_temporal.py)
  3. Inter-Stock Attention (master_cross_stock.py)
  4. Temporal Aggregation (this module)
  5. Prediction Head (this module)

Also implements MASTERStrategy(BaseStrategy) as strategy #27 (or higher),
registered in STRATEGY_REGISTRY via the existing auto-discovery mechanism.

Author: Hemrek Capital
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

from master_market import MarketStatusVector, MARKET_VECTOR_DIM
from master_gating import MarketGatingNetwork, DEFAULT_BETA
from master_temporal import IntraStockEncoder, TemporalBuffer, DEFAULT_TAU, DEFAULT_D, DEFAULT_N1
from master_cross_stock import InterStockAttention, DEFAULT_N2, compute_attention_entropy

logger = logging.getLogger("master_predict")

# Full pipeline checkpoint path
DEFAULT_PIPELINE_PATH = "master_pipeline.pt"

# Hybrid blending ratio (from Grok's analysis in vision plan)
DEFAULT_MASTER_WEIGHT = 0.7
DEFAULT_HEURISTIC_WEIGHT = 0.3
DEFAULT_DROPOUT = 0.3


class TemporalAggregation(nn.Module):
    """Temporal aggregation layer (MASTER paper Section: Temporal Aggregation).

    For each stock, the latest temporal embedding z(u,τ) queries against
    all historical z(u,t) via attention, producing a comprehensive stock
    embedding e_u.

        λ_{u,t} = softmax(z_{u,t}^T W_λ z_{u,τ})
        e_u = Σ_t λ_{u,t} z_{u,t}

    Args:
        d_model: Hidden dimension (D).
    """

    def __init__(self, d_model: int = DEFAULT_D):
        super().__init__()
        self.W_lambda = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Aggregate temporal embeddings into stock embeddings.

        Args:
            z: Temporal embeddings, shape (n_stocks, tau, D).

        Returns:
            Comprehensive stock embeddings e_u, shape (n_stocks, D).
        """
        # Query: latest temporal embedding z(u,τ)
        z_tau = z[:, -1, :]  # (n_stocks, D)

        # Attention scores: z_{u,t}^T W_λ z_{u,τ}
        # Transform the query
        q = self.W_lambda(z_tau)  # (n_stocks, D)

        # Score each time step against the query
        scores = torch.bmm(
            z,                          # (n_stocks, tau, D)
            q.unsqueeze(-1)             # (n_stocks, D, 1)
        ).squeeze(-1)                   # (n_stocks, tau)

        # Softmax attention weights
        lambda_weights = torch.softmax(scores, dim=-1)  # (n_stocks, tau)

        # Weighted sum: e_u = Σ_t λ_{u,t} z_{u,t}
        e = torch.bmm(
            lambda_weights.unsqueeze(1),   # (n_stocks, 1, tau)
            z                              # (n_stocks, tau, D)
        ).squeeze(1)                       # (n_stocks, D)

        return e


class MASTERPipeline(nn.Module):
    """Complete MASTER pipeline: Gating → Intra → Inter → Temporal → Predict.

    Assembles all five steps from the MASTER paper into a single module
    that takes raw features + market vector and produces cross-sectional
    return predictions for all ETFs jointly.

    Args:
        n_features: Number of input feature dimensions (F).
        market_dim: Dimension of market status vector.
        d_model: Hidden dimension (D). Default 64.
        n_heads_intra: Intra-stock attention heads (N₁). Default 2.
        n_heads_inter: Inter-stock attention heads (N₂). Default 2.
        tau: Lookback window. Default 10.
        beta: Gating temperature. Default 8.0.
        dropout: Dropout rate. Default 0.3.
    """

    def __init__(
        self,
        n_features: int,
        market_dim: int = MARKET_VECTOR_DIM,
        d_model: int = DEFAULT_D,
        n_heads_intra: int = DEFAULT_N1,
        n_heads_inter: int = DEFAULT_N2,
        tau: int = DEFAULT_TAU,
        beta: float = DEFAULT_BETA,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.tau = tau

        # Step 1: Market-Guided Gating
        self.gating = MarketGatingNetwork(
            market_dim=market_dim,
            n_features=n_features,
            beta=beta,
            dropout=dropout,
        )

        # Step 2: Intra-Stock Aggregation
        self.intra_stock = IntraStockEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads_intra,
            dropout=dropout,
            tau=tau,
        )

        # Step 3: Inter-Stock Aggregation
        self.inter_stock = InterStockAttention(
            d_model=d_model,
            n_heads=n_heads_inter,
            dropout=dropout,
        )

        # Step 4: Temporal Aggregation
        self.temporal_agg = TemporalAggregation(d_model=d_model)

        # Step 5: Prediction Head
        self.predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        m_tau: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the full MASTER pipeline.

        Args:
            features: Raw feature sequences, shape (n_stocks, tau, F).
            m_tau: Market status vector, shape (market_dim,).
            return_attention: If True, return inter-stock attention weights.

        Returns:
            predictions: Predicted Z-score normalized returns, shape (n_stocks,).
            attention_weights: If requested, shape (tau, n_stocks, n_stocks).
        """
        n_stocks, tau, F = features.shape

        # Step 1: Market-Guided Gating
        alpha = self.gating(m_tau)  # (F,)
        gated = features * alpha.unsqueeze(0).unsqueeze(0)  # broadcast

        # Step 2: Intra-Stock Aggregation (per stock)
        local_embeddings = self.intra_stock(gated)  # (n_stocks, tau, D)

        # Step 3: Inter-Stock Aggregation
        temporal_embeddings, attention = self.inter_stock(
            local_embeddings, return_attention=return_attention
        )

        # Step 4: Temporal Aggregation
        stock_embeddings = self.temporal_agg(temporal_embeddings)  # (n_stocks, D)

        # Step 5: Prediction
        predictions = self.predictor(stock_embeddings).squeeze(-1)  # (n_stocks,)

        return predictions, attention

    def predict(
        self,
        features_np: np.ndarray,
        m_tau_np: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Numpy convenience method for inference.

        Args:
            features_np: shape (n_stocks, tau, F).
            m_tau_np: shape (market_dim,).

        Returns:
            predictions: shape (n_stocks,).
            attention_matrix: shape (tau, n_stocks, n_stocks).
        """
        self.eval()
        with torch.no_grad():
            features = torch.tensor(features_np, dtype=torch.float32)
            m_tau = torch.tensor(m_tau_np, dtype=torch.float32)
            preds, attn = self.forward(features, m_tau, return_attention=True)

        attn_np = attn.cpu().numpy() if attn is not None else None
        return preds.cpu().numpy(), attn_np



def save_pipeline(model: MASTERPipeline, path: str = DEFAULT_PIPELINE_PATH) -> None:
    """Save the full MASTER pipeline checkpoint."""
    checkpoint = {
        'state_dict': model.state_dict(),
        'n_features': model.n_features,
        'd_model': model.d_model,
        'tau': model.tau,
        'gating_beta': model.gating.beta,
        'gating_n_features': model.gating.n_features,
        'gating_market_dim': model.gating.market_dim,
    }
    torch.save(checkpoint, path)
    logger.info("MASTER pipeline saved to %s", path)


def load_pipeline(path: str = DEFAULT_PIPELINE_PATH) -> Optional[MASTERPipeline]:
    """Load a trained MASTER pipeline from checkpoint."""
    if not os.path.exists(path):
        logger.debug("No MASTER pipeline found at %s", path)
        return None

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = MASTERPipeline(
            n_features=checkpoint['n_features'],
            d_model=checkpoint['d_model'],
            tau=checkpoint['tau'],
            beta=checkpoint['gating_beta'],
            market_dim=checkpoint['gating_market_dim'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        logger.info("MASTER pipeline loaded from %s", path)
        return model
    except Exception as e:
        logger.error("Failed to load MASTER pipeline from %s: %s", path, e)
        return None


# ============================================================================
# MASTERStrategy — Registered as a BaseStrategy for Pragyam
# ============================================================================

# Import here to avoid circular imports at module level
try:
    from strategies import BaseStrategy, PORTFOLIO_COLUMNS
    _STRATEGIES_AVAILABLE = True
except ImportError:
    _STRATEGIES_AVAILABLE = False
    logger.debug("strategies.py not available; MASTERStrategy disabled.")


if _STRATEGIES_AVAILABLE:

    class MASTERStrategy(BaseStrategy):
        """MASTER-based cross-sectional ETF selection strategy.

        Uses the full MASTER pipeline to predict cross-sectional forward
        returns, then allocates portfolio weights proportional to predicted
        returns (top-ranked ETFs get more weight).

        Hybrid scoring: 0.7 × MASTER prediction + 0.3 × composite heuristic
        (mean of RSI-based, oscillator-based, and MA-based signals from the
        input DataFrame). This prevents black-box risk while leveraging
        MASTER's cross-sectional ranking superiority.
        """

        def __init__(
            self,
            pipeline_path: str = DEFAULT_PIPELINE_PATH,
            master_weight: float = DEFAULT_MASTER_WEIGHT,
        ):
            self.pipeline_path = pipeline_path
            self.master_weight = master_weight
            self.heuristic_weight = 1.0 - master_weight
            self._pipeline: Optional[MASTERPipeline] = None
            self._temporal_buffer = TemporalBuffer()
            self._msv: Optional[MarketStatusVector] = None

        def _ensure_pipeline(self) -> bool:
            """Lazy-load the MASTER pipeline."""
            if self._pipeline is not None:
                return True
            self._pipeline = load_pipeline(self.pipeline_path)
            return self._pipeline is not None

        def _compute_heuristic_scores(self, df: pd.DataFrame) -> pd.Series:
            """Compute simple heuristic scores from existing indicators.

            Combines RSI reversal signal, oscillator signal, and MA proximity
            into a composite score. Used for the 30% heuristic blend.
            """
            scores = pd.Series(0.0, index=df.index)

            # RSI reversal: lower RSI = more attractive (mean reversion)
            if 'rsi latest' in df.columns:
                rsi = pd.to_numeric(df['rsi latest'], errors='coerce').fillna(50)
                scores += (50 - rsi) / 100  # [-0.5, 0.5]

            # Oscillator: lower = more attractive
            if 'osc latest' in df.columns:
                osc = pd.to_numeric(df['osc latest'], errors='coerce').fillna(0)
                scores += -osc / 200  # Normalize roughly

            # MA proximity: price below MA20 = attractive
            if 'ma20 latest' in df.columns and 'price' in df.columns:
                price = pd.to_numeric(df['price'], errors='coerce').fillna(0)
                ma20 = pd.to_numeric(df['ma20 latest'], errors='coerce').fillna(price)
                safe_ma20 = ma20.replace(0, 1)
                scores += (safe_ma20 - price) / safe_ma20  # Positive when below MA

            return scores

        def generate_portfolio(
            self, df: pd.DataFrame, sip_amount: float = 100000.0
        ) -> pd.DataFrame:
            """Generate portfolio using MASTER predictions + heuristic blend.

            If the MASTER pipeline is not available (not trained yet),
            falls back entirely to the heuristic score.
            """
            from backdata import NUMERIC_INDICATOR_COLS

            if df.empty or len(df) < 5:
                return pd.DataFrame(columns=list(PORTFOLIO_COLUMNS))

            # Compute heuristic scores
            heuristic = self._compute_heuristic_scores(df)

            # Try MASTER predictions
            master_preds = None
            if self._ensure_pipeline():
                try:
                    # Extract features for current snapshot
                    feature_cols = [c for c in NUMERIC_INDICATOR_COLS if c in df.columns]
                    features = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

                    # Update temporal buffer
                    for i, (_, row) in enumerate(df.iterrows()):
                        sym = str(row.get('symbol', f'ETF{i}'))
                        self._temporal_buffer.update(sym, features[i])

                    # Get sequences for all stocks
                    symbols = [str(row.get('symbol', f'ETF{i}')) for i, (_, row) in enumerate(df.iterrows())]
                    sequences = self._temporal_buffer.get_all_sequences(symbols)

                    if sequences is not None:
                        # Get market vector (use zeros if MSV not available)
                        m_tau = np.zeros(MARKET_VECTOR_DIM, dtype=np.float64)

                        preds, _ = self._pipeline.predict(sequences, m_tau)
                        master_preds = pd.Series(preds, index=df.index)
                except Exception as e:
                    logger.debug("MASTER prediction failed: %s", e)

            # Blend scores
            if master_preds is not None:
                final_scores = (
                    self.master_weight * master_preds
                    + self.heuristic_weight * heuristic
                )
            else:
                final_scores = heuristic

            # Cross-sectional Z-score normalization
            std = final_scores.std()
            if std > 1e-8:
                final_scores = (final_scores - final_scores.mean()) / std

            # Convert scores to weights: higher score = higher weight
            # Use softmax-like transformation
            exp_scores = np.exp(np.clip(final_scores.values, -3, 3))
            weights = exp_scores / exp_scores.sum()

            # Build portfolio
            portfolio = df[['symbol', 'price']].copy()
            portfolio['price'] = pd.to_numeric(portfolio['price'], errors='coerce').fillna(0)
            portfolio['weightage_pct'] = weights * 100

            # Allocate units
            portfolio['value'] = sip_amount * weights
            safe_price = portfolio['price'].replace(0, np.nan)
            portfolio['units'] = np.floor(portfolio['value'] / safe_price).fillna(0).astype(int)
            portfolio['value'] = portfolio['units'] * portfolio['price']

            # Filter out zero-unit positions
            portfolio = portfolio[portfolio['units'] > 0].copy()

            if portfolio.empty:
                return pd.DataFrame(columns=list(PORTFOLIO_COLUMNS))

            # Recalculate weights after unit rounding
            total_value = portfolio['value'].sum()
            if total_value > 0:
                portfolio['weightage_pct'] = (portfolio['value'] / total_value) * 100

            return portfolio[list(PORTFOLIO_COLUMNS)]
