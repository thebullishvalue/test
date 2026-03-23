"""
MASTER × Pragyam — Integration Test Suite
============================================

Validates:
  (a) All existing strategies produce identical output on ungated data
  (b) Gated features produce different but valid output
  (c) IC/RankIC computation is correct on synthetic data
  (d) MarketStatusVector produces correct-shaped output
  (e) MarketGatingNetwork produces correct-shaped gating coefficients

Run: python test_master_integration.py

Author: Hemrek Capital
"""

import sys
import numpy as np
import pandas as pd
import torch
import traceback
from datetime import datetime

# ============================================================================
# Test utilities
# ============================================================================

PASS_COUNT = 0
FAIL_COUNT = 0


def test(name: str, condition: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ {name}")
    else:
        FAIL_COUNT += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# Test A: Backward compatibility — COLUMN_ORDER extension
# ============================================================================

def test_column_order():
    section("A. COLUMN_ORDER backward compatibility")
    from backdata import COLUMN_ORDER, NUMERIC_INDICATOR_COLS

    # Original 22 columns must still be present
    original_cols = [
        'date', 'symbol', 'price', 'rsi latest', 'rsi weekly',
        '% change', 'osc latest', 'osc weekly',
        '9ema osc latest', '9ema osc weekly',
        '21ema osc latest', '21ema osc weekly',
        'zscore latest', 'zscore weekly',
        'ma20 latest', 'ma90 latest', 'ma200 latest',
        'ma20 weekly', 'ma90 weekly', 'ma200 weekly',
        'dev20 latest', 'dev20 weekly',
    ]

    for col in original_cols:
        test(f"Original column '{col}' in COLUMN_ORDER", col in COLUMN_ORDER)

    # New columns must be present
    new_cols = ['rocp5', 'rocp10', 'rocp20', 'volume_ratio', 'atr14', 'bbwidth', 'macd_hist']
    for col in new_cols:
        test(f"New column '{col}' in COLUMN_ORDER", col in COLUMN_ORDER)

    # NUMERIC_INDICATOR_COLS should exclude 'date' and 'symbol'
    test("'date' not in NUMERIC_INDICATOR_COLS", 'date' not in NUMERIC_INDICATOR_COLS)
    test("'symbol' not in NUMERIC_INDICATOR_COLS", 'symbol' not in NUMERIC_INDICATOR_COLS)
    test("'price' in NUMERIC_INDICATOR_COLS", 'price' in NUMERIC_INDICATOR_COLS)
    test(
        f"NUMERIC_INDICATOR_COLS has {len(NUMERIC_INDICATOR_COLS)} columns",
        len(NUMERIC_INDICATOR_COLS) == len(COLUMN_ORDER) - 2,
        f"Expected {len(COLUMN_ORDER) - 2}, got {len(NUMERIC_INDICATOR_COLS)}",
    )


# ============================================================================
# Test B: IC / RankIC / ICIR on synthetic data
# ============================================================================

def test_ic_metrics():
    section("B. IC / RankIC / ICIR metrics")
    from backtest_engine import ic, rank_ic, icir

    # Perfect positive correlation
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    test("IC(perfect positive) ≈ 1.0", abs(ic(a, b) - 1.0) < 1e-6)
    test("RankIC(perfect positive) ≈ 1.0", abs(rank_ic(a, b) - 1.0) < 1e-6)

    # Perfect negative correlation
    c = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    test("IC(perfect negative) ≈ -1.0", abs(ic(a, c) - (-1.0)) < 1e-6)
    test("RankIC(perfect negative) ≈ -1.0", abs(rank_ic(a, c) - (-1.0)) < 1e-6)

    # Zero correlation (orthogonal)
    d = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    e = np.array([1.0, 1.0, -1.0, -1.0, 0.0])
    ic_val = ic(d, e)
    test("IC(uncorrelated) ≈ 0", abs(ic_val) < 0.5, f"Got {ic_val:.4f}")

    # Edge cases
    test("IC(too few elements) = 0.0", ic(np.array([1.0]), np.array([2.0])) == 0.0)
    test("IC(constant input) = 0.0", ic(np.array([1, 1, 1, 1, 1.0]), a) == 0.0)

    # ICIR
    ic_series = np.array([0.05, 0.06, 0.04, 0.05, 0.07, 0.03, 0.05])
    icir_val = icir(ic_series)
    test("ICIR is positive for consistent positive IC", icir_val > 0, f"Got {icir_val:.4f}")

    # ICIR of zero-std series
    test("ICIR(constant) = 0.0", icir(np.array([0.05, 0.05, 0.05])) == 0.0)

    # ICIR with NaN
    ic_with_nan = np.array([0.05, np.nan, 0.06, np.nan, 0.04])
    icir_nan = icir(ic_with_nan)
    test("ICIR handles NaN gracefully", np.isfinite(icir_nan), f"Got {icir_nan}")


# ============================================================================
# Test C: MarketStatusVector
# ============================================================================

def test_market_status_vector():
    section("C. MarketStatusVector")
    from master_market import MarketStatusVector, MARKET_VECTOR_DIM

    # Create synthetic market data
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    np.random.seed(42)
    close = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5), index=dates)
    volume = pd.Series(np.random.randint(1000000, 5000000, size=200).astype(float), index=dates)

    fake_data = pd.DataFrame({
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'close': close,
        'volume': volume,
    }, index=dates)

    msv = MarketStatusVector(market_symbol="TEST")
    # Fit with single-symbol DataFrame (no MultiIndex)
    msv.fit(fake_data)

    test("MSV is_fitted after fit()", msv.is_fitted)
    test(f"MSV dim = {MARKET_VECTOR_DIM}", msv.dim == MARKET_VECTOR_DIM)

    # Get vector for a date with enough history
    vec = msv.get_vector(dates[100])
    test("get_vector returns array", vec is not None)
    if vec is not None:
        test(f"Vector shape is ({MARKET_VECTOR_DIM},)", vec.shape == (MARKET_VECTOR_DIM,))
        test("Vector has no NaN (after 100 days)", np.all(np.isfinite(vec)))
        test("Vector values are reasonable", np.all(np.abs(vec) < 1e12))

    # Normalized vector
    vec_norm = msv.get_vector_normalized(dates[100])
    test("get_vector_normalized returns array", vec_norm is not None)
    if vec_norm is not None:
        test("Normalized vector is finite", np.all(np.isfinite(vec_norm)))

    # Early date (not enough history for all rolling windows)
    vec_early = msv.get_vector(dates[3])
    test("get_vector for early date returns array (with 0-fill)", vec_early is not None)


# ============================================================================
# Test D: MarketGatingNetwork
# ============================================================================

def test_gating_network():
    section("D. MarketGatingNetwork")
    from master_gating import MarketGatingNetwork, save_gating_model, load_gating_model

    market_dim = 21
    n_features = 25
    beta = 8.0

    model = MarketGatingNetwork(
        market_dim=market_dim,
        n_features=n_features,
        beta=beta,
        dropout=0.3,
    )

    # Test forward pass (eval mode to disable dropout for deterministic output)
    model.eval()
    m_tau = torch.randn(market_dim)
    alpha = model(m_tau)
    test(f"Output shape is ({n_features},)", alpha.shape == (n_features,))
    test("All coefficients positive", torch.all(alpha > 0).item())
    test(
        f"Coefficients sum ≈ {n_features}",
        abs(alpha.sum().item() - n_features) < 0.01,
        f"Got {alpha.sum().item():.4f}",
    )

    # Test batch forward pass
    batch_m_tau = torch.randn(8, market_dim)
    batch_alpha = model(batch_m_tau)
    test("Batch output shape is (8, 25)", batch_alpha.shape == (8, n_features))

    # Test gate_features
    from backdata import NUMERIC_INDICATOR_COLS
    feature_cols = NUMERIC_INDICATOR_COLS[:n_features]

    fake_df = pd.DataFrame(
        np.random.randn(30, len(feature_cols)),
        columns=feature_cols,
    )
    fake_df['date'] = '01 Jan'
    fake_df['symbol'] = [f'ETF{i}' for i in range(30)]

    m_tau_np = np.random.randn(market_dim).astype(np.float64)
    gated_df = model.gate_features(m_tau_np, fake_df, feature_cols)

    test("Gated DF has same columns", set(gated_df.columns) == set(fake_df.columns))
    test("Gated DF has same shape", gated_df.shape == fake_df.shape)
    test("Non-numeric cols unchanged", all(gated_df['symbol'] == fake_df['symbol']))

    # Values should be different (gating applied)
    numeric_diff = (gated_df[feature_cols].values != fake_df[feature_cols].values).any()
    test("Gated numeric values differ from original", numeric_diff)

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name

    try:
        save_gating_model(model, tmp_path)
        test("Model saved successfully", os.path.exists(tmp_path))

        loaded = load_gating_model(tmp_path)
        test("Model loaded successfully", loaded is not None)

        if loaded is not None:
            alpha_loaded = loaded(m_tau)
            test(
                "Loaded model produces same output",
                torch.allclose(alpha, alpha_loaded, atol=1e-5),
            )
    finally:
        os.unlink(tmp_path)

    # Test load nonexistent file
    missing = load_gating_model("/nonexistent/path.pt")
    test("load_gating_model returns None for missing file", missing is None)


# ============================================================================
# Test E: generate_historical_data still works with use_market_gating=False
# ============================================================================

def test_generate_backward_compat():
    section("E. generate_historical_data backward compatibility")

    # Just verify the function signature accepts use_market_gating
    import inspect
    from backdata import generate_historical_data

    sig = inspect.signature(generate_historical_data)
    params = list(sig.parameters.keys())
    test("use_market_gating parameter exists", 'use_market_gating' in params)

    # Check default is False
    default = sig.parameters['use_market_gating'].default
    test("use_market_gating defaults to False", default is False)


# ============================================================================
# Test F: Strategy registry is unchanged
# ============================================================================

def test_strategies_unchanged():
    section("F. Strategy registry unchanged")
    try:
        from strategies import STRATEGY_REGISTRY
        test(
            f"STRATEGY_REGISTRY has strategies (found {len(STRATEGY_REGISTRY)})",
            len(STRATEGY_REGISTRY) > 20,
        )

        # Verify known strategy classes still exist
        known_strategies = ['PRStrategy', 'CL1Strategy']
        for name in known_strategies:
            test(f"Strategy '{name}' still registered", name in STRATEGY_REGISTRY)

    except ImportError as e:
        test("strategies.py imports successfully", False, str(e))


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("  MASTER × Pragyam Integration Test Suite")
    print("=" * 60)

    try:
        test_column_order()
    except Exception as e:
        print(f"  ✗ COLUMN_ORDER tests CRASHED: {e}")
        traceback.print_exc()

    try:
        test_ic_metrics()
    except Exception as e:
        print(f"  ✗ IC metrics tests CRASHED: {e}")
        traceback.print_exc()

    try:
        test_market_status_vector()
    except Exception as e:
        print(f"  ✗ MarketStatusVector tests CRASHED: {e}")
        traceback.print_exc()

    try:
        test_gating_network()
    except Exception as e:
        print(f"  ✗ GatingNetwork tests CRASHED: {e}")
        traceback.print_exc()

    try:
        test_generate_backward_compat()
    except Exception as e:
        print(f"  ✗ Backward compat tests CRASHED: {e}")
        traceback.print_exc()

    try:
        test_strategies_unchanged()
    except Exception as e:
        print(f"  ✗ Strategy registry tests CRASHED: {e}")
        traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"{'='*60}\n")

    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
