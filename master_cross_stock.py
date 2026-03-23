"""
MASTER × Pragyam — Phase 3: Inter-Stock Attention
====================================================

Implements the inter-stock aggregation from the MASTER paper (Section:
Inter-Stock Aggregation). At each time step t, gathers local embeddings
of all 30 ETFs and computes multi-head attention (N₂ heads).

The alternating structure (intra then inter) enables cross-time
correlation: signal from stock v at time j propagates to h(v,i) via
intra-stock, then to z(u,i) via inter-stock — effectively modeling
correlation from (v,j) to (u,i).

The attention weight matrices (30×30 per timestep) are the key output
for RMT integration and visualization.

Author: Hemrek Capital
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger("master_cross_stock")

DEFAULT_N2 = 2          # Inter-stock attention heads
DEFAULT_D = 64           # Must match temporal encoder
DEFAULT_DROPOUT = 0.3


class InterStockAttention(nn.Module):
    """Multi-head attention across stocks at each time step.

    At each time step t, gathers all stocks' local embeddings from the
    intra-stock encoder and computes cross-stock attention. Produces
    temporal embeddings z(u,t) that encode momentary correlations.

    The attention weight matrix at each timestep is a learned, dynamic
    correlation matrix — a neural analog to the RMT-cleaned correlation.

    Architecture per timestep:
        Q = W_Q · H_t,  K = W_K · H_t,  V = W_V · H_t
        Z_t = FFN(MHA(Q, K, V) + H_t)

    Args:
        d_model: Hidden dimension (D). Must match IntraStockEncoder.
        n_heads: Number of attention heads (N₂). Default 2.
        dropout: Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D,
        n_heads: int = DEFAULT_N2,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head attention for inter-stock
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN with residual connection (same as paper)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        local_embeddings: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute inter-stock attention at each time step.

        Args:
            local_embeddings: From IntraStockEncoder, shape (n_stocks, tau, D).
            return_attention: If True, return attention weights for visualization.

        Returns:
            temporal_embeddings: z(u,t), shape (n_stocks, tau, D).
            attention_weights: If requested, shape (tau, n_stocks, n_stocks),
                averaged across heads. None otherwise.
        """
        n_stocks, tau, d = local_embeddings.shape
        all_attention_weights = [] if return_attention else None

        temporal_embeddings = torch.zeros_like(local_embeddings)

        for t in range(tau):
            # Gather local embeddings of all stocks at time step t
            # H_t = h_{u,t} for all u ∈ S, shape (1, n_stocks, D)
            H_t = local_embeddings[:, t, :].unsqueeze(0)  # (1, n_stocks, D)

            # Multi-head attention across stocks
            attn_out, attn_weights = self.mha(
                H_t, H_t, H_t,
                need_weights=return_attention,
                average_attn_weights=True,  # Average across heads
            )

            # Residual + LayerNorm
            H_t_res = self.layer_norm1(attn_out + H_t)

            # FFN + Residual + LayerNorm
            Z_t = self.layer_norm2(self.ffn(H_t_res) + H_t_res)

            temporal_embeddings[:, t, :] = Z_t.squeeze(0)

            if return_attention and attn_weights is not None:
                # attn_weights: (1, n_stocks, n_stocks) → (n_stocks, n_stocks)
                all_attention_weights.append(attn_weights.squeeze(0).detach())

        attn_tensor = None
        if return_attention and all_attention_weights:
            # Stack: (tau, n_stocks, n_stocks)
            attn_tensor = torch.stack(all_attention_weights, dim=0)

        return temporal_embeddings, attn_tensor

    def get_attention_matrix(
        self,
        local_embeddings: torch.Tensor,
    ) -> np.ndarray:
        """Extract the inter-stock attention weight matrices.

        Convenience method for RMT integration and visualization.

        Args:
            local_embeddings: Shape (n_stocks, tau, D).

        Returns:
            numpy array of shape (tau, n_stocks, n_stocks).
        """
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(local_embeddings, return_attention=True)
        if attn is None:
            n_stocks, tau, _ = local_embeddings.shape
            return np.ones((tau, n_stocks, n_stocks)) / n_stocks
        return attn.cpu().numpy()

    def get_mean_attention_matrix(
        self,
        local_embeddings: torch.Tensor,
    ) -> np.ndarray:
        """Get the mean attention matrix across all time steps.

        This serves as a neural analog to the RMT denoised correlation
        matrix — a dynamic, learned correlation for the current window.

        Args:
            local_embeddings: Shape (n_stocks, tau, D).

        Returns:
            numpy array of shape (n_stocks, n_stocks), mean over τ timesteps.
        """
        attn = self.get_attention_matrix(local_embeddings)
        return attn.mean(axis=0)


def compute_attention_entropy(attention_matrix: np.ndarray) -> float:
    """Compute entropy of the inter-stock attention distribution.

    Concentrated attention = herding/crisis regime (low entropy).
    Dispersed attention = diversified/healthy regime (high entropy).

    This is a learned, real-time analog of the absorption ratio.

    Args:
        attention_matrix: shape (n_stocks, n_stocks), averaged over time.

    Returns:
        Normalized entropy in [0, 1]. 0 = fully concentrated, 1 = uniform.
    """
    # Flatten and normalize
    flat = attention_matrix.flatten()
    flat = flat[flat > 0]
    flat = flat / flat.sum()

    # Shannon entropy
    entropy = -np.sum(flat * np.log(flat + 1e-12))

    # Maximum entropy for this matrix size
    max_entropy = np.log(len(flat) + 1e-12)

    if max_entropy < 1e-12:
        return 1.0

    return float(entropy / max_entropy)


def compute_cross_time_correlation(
    intra_attention: np.ndarray,
    inter_attention: np.ndarray,
    source_stock: int,
    target_stock: int,
) -> np.ndarray:
    """Compute the τ×τ cross-time correlation map between two stocks.

    From the paper: I_{u←v}[i,j] = S¹_v[i,j] · S²_i[u,v]
    where S¹ is intra-stock attention and S² is inter-stock attention.

    Args:
        intra_attention: Not directly available from transformer, approximate
            using identity (intra-stock attention distributes info across time).
        inter_attention: shape (tau, n_stocks, n_stocks).
        source_stock: Index of source stock v.
        target_stock: Index of target stock u.

    Returns:
        numpy array of shape (tau, tau) representing cross-time correlation.
    """
    tau = inter_attention.shape[0]
    cross_time = np.zeros((tau, tau))

    for i in range(tau):
        for j in range(tau):
            # Inter-stock attention at time i from v to u
            inter_weight = inter_attention[i, target_stock, source_stock]
            # Approximate intra-stock propagation from j to i
            # (decays with time distance)
            time_decay = np.exp(-abs(i - j) / (tau / 2))
            cross_time[i, j] = inter_weight * time_decay

    return cross_time
