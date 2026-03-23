"""
MASTER × Pragyam — Phase 5: Enhanced Regime Detection
=======================================================

Provides two sub-analyzers that integrate into MarketRegimeDetectorV2's
composite scoring:

  1. GatingRegimeAnalyzer: Extracts market regime signal from gating
     coefficient patterns (α vector entropy + dominant feature groups).

  2. AttentionRegimeAnalyzer: Uses inter-stock attention entropy as a
     real-time, learned analog of the absorption ratio.

Both are optional: if MASTER modules are unavailable or untrained,
each analyzer returns a neutral score (0.0) so the existing detector
continues to work unchanged.

Author: Hemrek Capital
"""

import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger("master_regime")


# ---------------------------------------------------------------------------
# 1. Gating Regime Analyzer
# ---------------------------------------------------------------------------

class GatingRegimeAnalyzer:
    """Extract regime signals from market-guided gating coefficients.

    The gating vector α(mτ) reveals which feature groups the model considers
    important under current market conditions. The *entropy* of α tells us
    whether the model is routing through many features (calm/diversified)
    or concentrating on few (stressed/crisis).

    Integration point: called by MarketRegimeDetectorV2._calculate_composite_score()
    """

    def __init__(self):
        self._gating_model = None
        self._loaded = False

    def _ensure_model(self) -> bool:
        if self._loaded:
            return self._gating_model is not None
        self._loaded = True
        try:
            from master_gating import load_gating_model
            self._gating_model = load_gating_model()
            return self._gating_model is not None
        except ImportError:
            return False

    def analyze(self, m_tau: Optional[np.ndarray]) -> Dict:
        """Analyze gating coefficients for regime signal.

        Args:
            m_tau: Normalized market status vector, shape (market_dim,).
                   None if MarketStatusVector unavailable.

        Returns:
            Dict with 'score' in [-1, 1] and diagnostic fields.
        """
        neutral = {'score': 0.0, 'regime': 'NEUTRAL', 'entropy': None, 'dominant_features': []}

        if m_tau is None or not self._ensure_model():
            return neutral

        try:
            import torch
            self._gating_model.eval()
            with torch.no_grad():
                m_tensor = torch.tensor(m_tau, dtype=torch.float32)
                alpha = self._gating_model(m_tensor).cpu().numpy()

            # Normalize α to a probability distribution
            alpha_pos = np.maximum(alpha, 1e-12)
            p = alpha_pos / alpha_pos.sum()

            # Shannon entropy (normalized to [0, 1])
            entropy = -np.sum(p * np.log(p + 1e-12))
            max_entropy = np.log(len(p) + 1e-12)
            norm_entropy = entropy / max_entropy if max_entropy > 1e-12 else 1.0

            # Top-3 dominant features (indices with highest α)
            top_k = min(3, len(alpha))
            dominant_idx = np.argsort(alpha)[-top_k:][::-1].tolist()

            # Score mapping:
            # High entropy (>0.85) → calm, diversified → positive score
            # Low entropy (<0.60) → concentrated, stressed → negative score
            if norm_entropy > 0.85:
                score = 0.4
                regime = 'DIVERSIFIED'
            elif norm_entropy > 0.70:
                score = 0.1
                regime = 'BALANCED'
            elif norm_entropy > 0.55:
                score = -0.2
                regime = 'CONCENTRATING'
            else:
                score = -0.5
                regime = 'STRESSED'

            return {
                'score': score,
                'regime': regime,
                'entropy': float(norm_entropy),
                'dominant_features': dominant_idx,
            }

        except Exception as e:
            logger.debug("GatingRegimeAnalyzer failed: %s", e)
            return neutral


# ---------------------------------------------------------------------------
# 2. Attention Regime Analyzer
# ---------------------------------------------------------------------------

class AttentionRegimeAnalyzer:
    """Extract regime signal from inter-stock attention entropy.

    Concentrated attention (low entropy) → herding/crisis regime.
    Dispersed attention (high entropy) → diversified/healthy regime.

    This is a neural, real-time analog of the absorption ratio from RMT.

    Integration point: called by MarketRegimeDetectorV2._calculate_composite_score()
    """

    def analyze(self, attention_matrix: Optional[np.ndarray]) -> Dict:
        """Analyze inter-stock attention for regime signal.

        Args:
            attention_matrix: Mean attention matrix, shape (n_stocks, n_stocks).
                              None if pipeline unavailable.

        Returns:
            Dict with 'score' in [-1, 1] and diagnostic fields.
        """
        neutral = {'score': 0.0, 'regime': 'NEUTRAL', 'attention_entropy': None}

        if attention_matrix is None:
            return neutral

        try:
            from master_cross_stock import compute_attention_entropy
            attn_entropy = compute_attention_entropy(attention_matrix)

            # Score mapping:
            # High entropy (>0.80) → healthy diversification → positive
            # Low entropy (<0.50) → herding/crisis → negative
            if attn_entropy > 0.80:
                score = 0.5
                regime = 'DISPERSED'
            elif attn_entropy > 0.65:
                score = 0.2
                regime = 'MODERATE'
            elif attn_entropy > 0.50:
                score = -0.2
                regime = 'CONCENTRATING'
            else:
                score = -0.6
                regime = 'HERDING'

            return {
                'score': score,
                'regime': regime,
                'attention_entropy': float(attn_entropy),
            }

        except Exception as e:
            logger.debug("AttentionRegimeAnalyzer failed: %s", e)
            return neutral


# ---------------------------------------------------------------------------
# Convenience: composite MASTER regime signal
# ---------------------------------------------------------------------------

def get_master_regime_signal(
    m_tau: Optional[np.ndarray] = None,
    attention_matrix: Optional[np.ndarray] = None,
) -> Dict:
    """Get combined MASTER regime signal for integration into composite score.

    Returns a dict with:
        'score': float in [-1, 1] (0.0 if both unavailable)
        'gating': gating analyzer result
        'attention': attention analyzer result
    """
    gating_result = GatingRegimeAnalyzer().analyze(m_tau)
    attention_result = AttentionRegimeAnalyzer().analyze(attention_matrix)

    # Weighted blend: attention is the primary signal (more informative),
    # gating is supplementary
    gating_score = gating_result['score']
    attn_score = attention_result['score']

    has_gating = gating_result['entropy'] is not None
    has_attention = attention_result['attention_entropy'] is not None

    if has_gating and has_attention:
        combined = 0.4 * gating_score + 0.6 * attn_score
    elif has_attention:
        combined = attn_score
    elif has_gating:
        combined = gating_score
    else:
        combined = 0.0

    return {
        'score': float(combined),
        'gating': gating_result,
        'attention': attention_result,
    }
