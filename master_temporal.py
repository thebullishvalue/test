"""
MASTER × Pragyam — Phase 2: Intra-Stock Temporal Encoder
==========================================================

Implements the intra-stock aggregation from the MASTER paper (Section:
Intra-Stock Aggregation). A single-layer transformer encoder processes
each ETF's τ-day feature sequence independently, producing local
embeddings h(u,t) that preserve time-step-specific detail while
aggregating cross-time signals within the same stock.

Key design: maintains a *sequence* of local embeddings (not one summary
vector), enabling the inter-stock module to access any time step.

Author: Hemrek Capital
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List
import logging

logger = logging.getLogger("master_temporal")

# Default hyperparameters (from vision plan, adjusted for 30-ETF universe)
DEFAULT_TAU = 10        # Lookback window: 2 trading weeks
DEFAULT_D = 64          # Hidden dimension (conservative for 30 ETFs)
DEFAULT_N1 = 2          # Intra-stock attention heads
DEFAULT_DROPOUT = 0.3


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Marks the chronological order within the lookback window.
    """

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x + positional encoding, same shape.
        """
        return x + self.pe[:, :x.size(1), :]


class IntraStockEncoder(nn.Module):
    """Single-layer transformer encoder for intra-stock temporal aggregation.

    For each stock, processes the τ-day feature sequence and produces
    local embeddings h(u,t) at every time step.

    Architecture:
        feature_layer: Linear(F, D)
        positional_encoding: sinusoidal
        layer_norm
        single-layer transformer encoder (N₁ heads)
        FFN with residual connection

    Args:
        n_features: Number of input feature dimensions (F).
        d_model: Hidden dimension (D). Default 64.
        n_heads: Number of attention heads (N₁). Default 2.
        dropout: Dropout probability. Default 0.3.
        tau: Maximum lookback window length. Default 10.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = DEFAULT_D,
        n_heads: int = DEFAULT_N1,
        dropout: float = DEFAULT_DROPOUT,
        tau: int = DEFAULT_TAU,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.tau = tau

        # Feature projection: Linear(F, D)
        self.feature_layer = nn.Linear(n_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=tau + 10)

        # Single-layer transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single stock's temporal feature sequence.

        Args:
            x: Gated feature sequence, shape (batch, tau, F) or (tau, F).

        Returns:
            Local embeddings h(u,t), shape (batch, tau, D) or (tau, D).
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)

        # Feature projection + Layer Norm + Positional Encoding
        # Y_u = LN(f(x̃_{u,t}) + p_t) for each t
        y = self.feature_layer(x)               # (batch, tau, D)
        y = self.layer_norm(y)
        y = self.pos_encoding(y)

        # Single-layer transformer encoder: self-attention across time steps
        # Each time step queries from all time steps within the same stock
        h = self.transformer_encoder(y)          # (batch, tau, D)
        h = self.dropout(h)

        if squeeze:
            h = h.squeeze(0)

        return h


class TemporalBuffer:
    """Maintains a τ-day rolling buffer of features per symbol.

    Used by backdata.py to accumulate daily snapshots into the lookback
    windows needed by the temporal encoder.

    Args:
        tau: Lookback window length. Default 10.
        feature_cols: List of numeric feature column names.
    """

    def __init__(self, tau: int = DEFAULT_TAU, feature_cols: Optional[List[str]] = None):
        self.tau = tau
        self.feature_cols = feature_cols
        self._buffers: Dict[str, List[np.ndarray]] = {}

    def update(self, symbol: str, feature_vector: np.ndarray) -> None:
        """Push a new daily feature vector for a symbol.

        Args:
            symbol: ETF ticker (without .NS suffix).
            feature_vector: 1-D array of feature values for this day.
        """
        if symbol not in self._buffers:
            self._buffers[symbol] = []

        self._buffers[symbol].append(feature_vector.copy())

        # Keep only the last τ days
        if len(self._buffers[symbol]) > self.tau:
            self._buffers[symbol] = self._buffers[symbol][-self.tau:]

    def get_sequence(self, symbol: str) -> Optional[np.ndarray]:
        """Get the τ-day feature sequence for a symbol.

        Returns:
            numpy array of shape (tau, F) if enough history, else None.
            If fewer than τ days available, pads with zeros at the front.
        """
        if symbol not in self._buffers or len(self._buffers[symbol]) == 0:
            return None

        buf = self._buffers[symbol]
        seq = np.array(buf, dtype=np.float64)

        # Pad with zeros if we don't have τ days yet
        if seq.shape[0] < self.tau:
            pad = np.zeros((self.tau - seq.shape[0], seq.shape[1]), dtype=np.float64)
            seq = np.vstack([pad, seq])

        return seq

    def get_all_sequences(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Get sequences for all specified symbols as a 3D tensor.

        Args:
            symbols: List of symbol names.

        Returns:
            numpy array of shape (n_stocks, tau, F) or None if any missing.
        """
        sequences = []
        for sym in symbols:
            seq = self.get_sequence(sym)
            if seq is None:
                return None
            sequences.append(seq)

        return np.stack(sequences, axis=0)

    @property
    def n_symbols(self) -> int:
        return len(self._buffers)

    def clear(self) -> None:
        self._buffers.clear()
