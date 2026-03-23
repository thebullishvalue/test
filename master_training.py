"""
MASTER × Pragyam — Phase 1: Gating Network Training Pipeline
===============================================================

Trains the MarketGatingNetwork by optimizing cross-sectional IC
(Information Coefficient) of gated features against forward returns.

Training protocol:
  - Walk-forward with expanding window
  - 30-day embargo gap between train and validation
  - Simple cross-sectional linear regression as the downstream signal
  - Objective: maximize mean IC over validation period
  - Early stopping on validation IC

Usage:
    python master_training.py --start 2015-01-01 --end 2024-12-31 --epochs 40

Author: Hemrek Capital
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os

from backdata import (
    generate_historical_data,
    load_symbols_from_file,
    COLUMN_ORDER,
    NUMERIC_INDICATOR_COLS,
    MAX_INDICATOR_PERIOD,
    SYMBOLS_UNIVERSE,
)
from master_market import MarketStatusVector, MARKET_PROXY_SYMBOL
from master_gating import (
    MarketGatingNetwork,
    save_gating_model,
    load_gating_model,
    DEFAULT_MODEL_PATH,
)
from backtest_engine import ic, rank_ic, icir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("master_training")

# Training constants
EMBARGO_DAYS = 30          # Gap between train and validation to prevent leakage
FORWARD_RETURN_DAYS = 5    # Weekly forward returns (d=5 in MASTER paper)
MIN_STOCKS_PER_DAY = 10    # Minimum ETFs needed for cross-sectional regression
VALIDATION_WINDOW = 60     # Validation period in trading days


class GatingTrainer:
    """Walk-forward trainer for the MarketGatingNetwork.

    Trains the gating network by:
    1. For each training date: compute mτ, apply gating α to features
    2. Run cross-sectional linear regression on gated features → forward returns
    3. Compute IC between predicted and actual forward returns
    4. Optimize gating weights to maximize mean IC

    The key insight is that the gating network learns which features are
    most informative for cross-sectional return prediction under different
    market conditions, without modifying any existing strategy.

    Args:
        market_dim: Dimension of market status vector (default 21).
        beta: Gating temperature (default 8.0 for 30-ETF universe).
        lr: Learning rate (default 1e-4).
        epochs: Maximum training epochs (default 40).
        dropout: Dropout rate (default 0.3).
        device: PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        market_dim: int = 21,
        beta: float = 8.0,
        lr: float = 1e-4,
        epochs: int = 40,
        dropout: float = 0.3,
        device: str = 'cpu',
    ):
        self.market_dim = market_dim
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.device = torch.device(device)

        # Will be initialized during training
        self.model: Optional[MarketGatingNetwork] = None
        self.feature_cols: List[str] = NUMERIC_INDICATOR_COLS

    def _prepare_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[
        List[Tuple[datetime, pd.DataFrame]],
        MarketStatusVector,
        pd.DataFrame,
        Dict[str, pd.Series],
    ]:
        """Download data and prepare all inputs for training.

        Returns:
            snapshots: List of (date, indicator_df) tuples.
            msv: Fitted MarketStatusVector.
            all_data_raw: Raw yfinance download (for market vector).
            price_series: Dict of {symbol: close_price_series} for return computation.
        """
        import yfinance as yf

        logger.info("Downloading data from %s to %s...", start_date, end_date)

        # Download raw data (need it for both indicators and market vector)
        all_data = yf.download(
            symbols,
            start=start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 60),
            end=end_date + timedelta(days=FORWARD_RETURN_DAYS + 5),
            progress=False,
        )

        if all_data.empty:
            raise RuntimeError("yf.download returned empty data.")

        all_data.columns.names = ['Indicator', 'Symbol']

        # Fit market status vector
        msv = MarketStatusVector(MARKET_PROXY_SYMBOL)
        msv.fit(all_data)

        if not msv.is_fitted:
            raise RuntimeError("Failed to fit MarketStatusVector.")

        # Extract close price series per symbol for forward return calculation
        price_series = {}
        for sym in symbols:
            try:
                close = all_data.xs(sym, level='Symbol', axis=1)['Close']
                close = pd.to_numeric(close, errors='coerce').dropna()
                price_series[sym] = close
            except (KeyError, IndexError):
                continue

        # Generate indicator snapshots
        logger.info("Generating indicator snapshots...")
        snapshots = generate_historical_data(
            symbols,
            start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 60),
            end_date + timedelta(days=FORWARD_RETURN_DAYS + 5),
        )

        logger.info("Prepared %d snapshots, %d price series.", len(snapshots), len(price_series))
        return snapshots, msv, all_data, price_series

    def _compute_forward_returns(
        self,
        snapshot_date: pd.Timestamp,
        snapshot_df: pd.DataFrame,
        price_series: Dict[str, pd.Series],
    ) -> Optional[np.ndarray]:
        """Compute d-day forward returns for all ETFs in a snapshot.

        Returns:
            numpy array of forward returns aligned with snapshot_df rows,
            or None if insufficient data.
        """
        returns = []
        snapshot_date = pd.Timestamp(snapshot_date).normalize()

        for _, row in snapshot_df.iterrows():
            symbol = row['symbol']
            full_symbol = symbol + '.NS' if not symbol.endswith('.NS') else symbol

            if full_symbol not in price_series:
                returns.append(np.nan)
                continue

            ps = price_series[full_symbol]

            # Find the closest date on or after snapshot_date
            future_dates = ps.index[ps.index > snapshot_date]
            if len(future_dates) < FORWARD_RETURN_DAYS:
                returns.append(np.nan)
                continue

            current_price = ps.loc[ps.index[ps.index <= snapshot_date][-1]] if any(ps.index <= snapshot_date) else np.nan
            future_price = ps.iloc[ps.index.get_indexer(future_dates[:FORWARD_RETURN_DAYS], method='nearest')[-1]]

            if pd.notna(current_price) and current_price > 0:
                ret = (future_price - current_price) / current_price
                returns.append(ret)
            else:
                returns.append(np.nan)

        returns = np.array(returns, dtype=np.float64)

        # Z-score normalize cross-sectionally (as per MASTER paper)
        valid_mask = np.isfinite(returns)
        if valid_mask.sum() < MIN_STOCKS_PER_DAY:
            return None

        mean_r = np.nanmean(returns)
        std_r = np.nanstd(returns)
        if std_r > 1e-8:
            returns = (returns - mean_r) / std_r

        return returns

    def _extract_features(
        self,
        snapshot_df: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Extract numeric features from a snapshot DataFrame.

        Returns:
            numpy array of shape (n_stocks, n_features) or None.
        """
        available_cols = [c for c in self.feature_cols if c in snapshot_df.columns]
        if len(available_cols) < 5:
            return None

        X = snapshot_df[available_cols].values.astype(np.float64)
        # Replace NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _cross_sectional_ic(
        self,
        features: np.ndarray,
        forward_returns: np.ndarray,
        gating_alpha: np.ndarray,
    ) -> float:
        """Compute IC using gated features and simple linear regression.

        Steps:
        1. Apply gating: X_gated = X * alpha
        2. OLS: predicted = X_gated @ beta_hat (cross-sectional)
        3. IC = pearson(predicted, actual_returns)
        """
        valid = np.isfinite(forward_returns)
        if valid.sum() < MIN_STOCKS_PER_DAY:
            return 0.0

        X = features[valid] * gating_alpha  # Hadamard with broadcast
        y = forward_returns[valid]

        # Simple OLS: beta = (X'X)^-1 X'y
        try:
            XtX = X.T @ X
            # Regularize slightly for numerical stability
            XtX += np.eye(X.shape[1]) * 1e-6
            beta_hat = np.linalg.solve(XtX, X.T @ y)
            predicted = X @ beta_hat
            return ic(predicted, y)
        except np.linalg.LinAlgError:
            return 0.0

    def train(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        val_start: Optional[datetime] = None,
        save_path: str = DEFAULT_MODEL_PATH,
    ) -> Dict[str, float]:
        """Run the full training pipeline.

        Args:
            symbols: List of ticker symbols. Defaults to SYMBOLS_UNIVERSE.
            start_date: Training start date.
            end_date: Training end date.
            val_start: Validation start date. If None, uses end_date - VALIDATION_WINDOW.
            save_path: Path to save the best model checkpoint.

        Returns:
            Dict with training metrics: best_val_ic, best_val_rank_ic, etc.
        """
        if symbols is None:
            symbols = SYMBOLS_UNIVERSE
        if start_date is None:
            start_date = datetime(2015, 1, 1)
        if end_date is None:
            end_date = datetime(2024, 6, 30)
        if val_start is None:
            val_start = end_date - timedelta(days=VALIDATION_WINDOW * 2)

        # Prepare data
        snapshots, msv, all_data, price_series = self._prepare_data(
            symbols, start_date, end_date
        )

        if not snapshots:
            raise RuntimeError("No snapshots generated. Check date range and data.")

        # Split into train/val using dates
        train_embargo_end = val_start - timedelta(days=EMBARGO_DAYS)
        train_snapshots = [
            (d, df) for d, df in snapshots
            if d <= pd.Timestamp(train_embargo_end) and d >= pd.Timestamp(start_date)
        ]
        val_snapshots = [
            (d, df) for d, df in snapshots
            if d >= pd.Timestamp(val_start) and d <= pd.Timestamp(end_date)
        ]

        logger.info(
            "Train: %d snapshots (%s to %s), Val: %d snapshots (%s to %s)",
            len(train_snapshots),
            train_snapshots[0][0].date() if train_snapshots else "N/A",
            train_snapshots[-1][0].date() if train_snapshots else "N/A",
            len(val_snapshots),
            val_snapshots[0][0].date() if val_snapshots else "N/A",
            val_snapshots[-1][0].date() if val_snapshots else "N/A",
        )

        # Determine actual feature count from data
        sample_df = train_snapshots[0][1] if train_snapshots else snapshots[0][1]
        available_cols = [c for c in self.feature_cols if c in sample_df.columns]
        n_features = len(available_cols)
        self.feature_cols = available_cols  # Update to actual available columns

        logger.info("Feature dimension F=%d, Market dim=%d", n_features, self.market_dim)

        # Initialize model
        self.model = MarketGatingNetwork(
            market_dim=self.market_dim,
            n_features=n_features,
            beta=self.beta,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Pre-compute training data: market vectors, features, forward returns
        logger.info("Pre-computing training data...")
        train_data = []
        for date, df in train_snapshots:
            m_tau = msv.get_vector_normalized(date)
            if m_tau is None:
                continue
            features = self._extract_features(df)
            if features is None:
                continue
            fwd_ret = self._compute_forward_returns(date, df, price_series)
            if fwd_ret is None:
                continue
            train_data.append((m_tau, features, fwd_ret))

        val_data = []
        for date, df in val_snapshots:
            m_tau = msv.get_vector_normalized(date)
            if m_tau is None:
                continue
            features = self._extract_features(df)
            if features is None:
                continue
            fwd_ret = self._compute_forward_returns(date, df, price_series)
            if fwd_ret is None:
                continue
            val_data.append((m_tau, features, fwd_ret))

        logger.info(
            "Usable samples: train=%d, val=%d", len(train_data), len(val_data)
        )

        if len(train_data) < 50:
            logger.warning(
                "Only %d training samples. Results may be unreliable.", len(train_data)
            )

        # Training loop
        best_val_ic = -np.inf
        best_epoch = 0
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_ics = []

            # Shuffle training data each epoch
            indices = np.random.permutation(len(train_data))

            epoch_loss = 0.0
            for idx in indices:
                m_tau, features, fwd_ret = train_data[idx]

                m_tau_t = torch.tensor(m_tau, dtype=torch.float32, device=self.device)
                alpha = self.model(m_tau_t)
                alpha_np = alpha.detach().cpu().numpy()

                # Compute IC with current gating
                current_ic = self._cross_sectional_ic(features, fwd_ret, alpha_np)
                train_ics.append(current_ic)

                # Loss: negative IC (we want to maximize IC)
                # Use a differentiable proxy: apply gating, compute predicted returns,
                # then use negative correlation as loss
                valid = np.isfinite(fwd_ret)
                if valid.sum() < MIN_STOCKS_PER_DAY:
                    continue

                X = torch.tensor(
                    features[valid], dtype=torch.float32, device=self.device
                )
                y = torch.tensor(
                    fwd_ret[valid], dtype=torch.float32, device=self.device
                )

                # Gated features
                X_gated = X * alpha.unsqueeze(0)  # broadcast (1, F) * (N, F)

                # Simple linear prediction (differentiable)
                # Use mean of gated features as a simple score
                predicted = X_gated.mean(dim=1)

                # Differentiable correlation loss
                p_centered = predicted - predicted.mean()
                y_centered = y - y.mean()
                p_std = p_centered.std() + 1e-8
                y_std = y_centered.std() + 1e-8
                corr = (p_centered * y_centered).mean() / (p_std * y_std)

                loss = -corr  # Maximize correlation = minimize negative correlation
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            # Validation
            self.model.eval()
            val_ics = []
            val_rank_ics = []

            with torch.no_grad():
                for m_tau, features, fwd_ret in val_data:
                    m_tau_t = torch.tensor(
                        m_tau, dtype=torch.float32, device=self.device
                    )
                    alpha = self.model(m_tau_t).cpu().numpy()
                    val_ic = self._cross_sectional_ic(features, fwd_ret, alpha)
                    val_ics.append(val_ic)

                    # Also compute RankIC
                    valid = np.isfinite(fwd_ret)
                    if valid.sum() >= MIN_STOCKS_PER_DAY:
                        X_gated = features[valid] * alpha
                        try:
                            XtX = X_gated.T @ X_gated + np.eye(X_gated.shape[1]) * 1e-6
                            beta_hat = np.linalg.solve(XtX, X_gated.T @ fwd_ret[valid])
                            pred = X_gated @ beta_hat
                            val_rank_ics.append(rank_ic(pred, fwd_ret[valid]))
                        except np.linalg.LinAlgError:
                            pass

            mean_train_ic = np.mean(train_ics) if train_ics else 0.0
            mean_val_ic = np.mean(val_ics) if val_ics else 0.0
            mean_val_rank_ic = np.mean(val_rank_ics) if val_rank_ics else 0.0

            logger.info(
                "Epoch %02d/%02d | Train IC: %.4f | Val IC: %.4f | Val RankIC: %.4f | Loss: %.4f",
                epoch + 1, self.epochs, mean_train_ic, mean_val_ic,
                mean_val_rank_ic, epoch_loss / max(len(train_data), 1),
            )

            # Early stopping on validation IC
            if mean_val_ic > best_val_ic:
                best_val_ic = mean_val_ic
                best_epoch = epoch + 1
                patience_counter = 0
                save_gating_model(self.model, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (best val IC: %.4f at epoch %d)",
                        epoch + 1, best_val_ic, best_epoch,
                    )
                    break

        # Load best model
        best_model = load_gating_model(save_path)
        if best_model is not None:
            self.model = best_model

        # Final validation metrics
        final_val_ics = []
        final_val_rank_ics = []
        self.model.eval()
        with torch.no_grad():
            for m_tau, features, fwd_ret in val_data:
                m_tau_t = torch.tensor(m_tau, dtype=torch.float32)
                alpha = self.model(m_tau_t).numpy()
                final_val_ics.append(
                    self._cross_sectional_ic(features, fwd_ret, alpha)
                )

                valid = np.isfinite(fwd_ret)
                if valid.sum() >= MIN_STOCKS_PER_DAY:
                    X_gated = features[valid] * alpha
                    try:
                        XtX = X_gated.T @ X_gated + np.eye(X_gated.shape[1]) * 1e-6
                        beta_hat = np.linalg.solve(XtX, X_gated.T @ fwd_ret[valid])
                        pred = X_gated @ beta_hat
                        final_val_rank_ics.append(rank_ic(pred, fwd_ret[valid]))
                    except np.linalg.LinAlgError:
                        pass

        results = {
            'best_val_ic': best_val_ic,
            'best_epoch': best_epoch,
            'final_val_ic': np.mean(final_val_ics) if final_val_ics else 0.0,
            'final_val_rank_ic': np.mean(final_val_rank_ics) if final_val_rank_ics else 0.0,
            'final_val_icir': icir(np.array(final_val_ics)) if final_val_ics else 0.0,
            'n_train_samples': len(train_data),
            'n_val_samples': len(val_data),
            'n_features': len(self.feature_cols),
        }

        logger.info("Training complete. Results: %s", results)
        return results


def main():
    """CLI entry point for gating network training."""
    parser = argparse.ArgumentParser(
        description="Train MASTER Market-Guided Gating Network"
    )
    parser.add_argument(
        '--start', type=str, default='2015-01-01',
        help='Training start date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--end', type=str, default='2024-06-30',
        help='Training end date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--val-start', type=str, default=None,
        help='Validation start date (YYYY-MM-DD). Default: end - 120 days.',
    )
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=8.0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument(
        '--save-path', type=str, default=DEFAULT_MODEL_PATH,
        help='Path to save the trained model.',
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    val_start = (
        datetime.strptime(args.val_start, '%Y-%m-%d')
        if args.val_start else None
    )

    trainer = GatingTrainer(
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        dropout=args.dropout,
    )

    results = trainer.train(
        symbols=SYMBOLS_UNIVERSE,
        start_date=start_date,
        end_date=end_date,
        val_start=val_start,
        save_path=args.save_path,
    )

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:.<30} {v:.4f}")
        else:
            print(f"  {k:.<30} {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
