"""
Historical replay and statistical evaluation of oscillator signals.

Two different questions are answered here, and conflating them is the usual
way backtests mislead:

  1. *Does the oscillator carry information?*  Answered by the information
     coefficient and the forward-return-by-decile table — no trading rule, no
     thresholds, no parameters to overfit. If these are flat, nothing built on
     top of the signal will work.

  2. *Does one particular trading rule make money?*  Answered by the replay,
     which is a single path through a large parameter space and should be read
     with corresponding suspicion.

Execution convention: a signal observed at the close of session t is acted on
at session t+1, and costs are charged on every change in position. There is no
peeking — every input to a decision at t is computed from data up to t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import BacktestConfig


def run_backtest(E: dict, cfg: BacktestConfig) -> dict:
    """Replay the oscillator's threshold signals as a long/short strategy."""
    fvo = E["fvo"].values
    ob, os_ = E["ob"].values, E["os_"].values
    conf = E["confidence"].values
    rets = E["rets"][E["target"]].values
    price = E["price"].values
    dates = E["dates"]
    T = len(fvo)

    start = E["lookback"] + int(E["config"].fv_horizon)
    cost = cfg.cost_bps / 1e4

    pos = np.zeros(T)
    strat = np.zeros(T)
    trades: list[dict] = []
    current, entry_i, held, size = 0, -1, 0, 0.0

    for t in range(start, T - 1):
        if current != 0:
            held += 1
            exhausted = held >= cfg.max_hold
            reverted = (current == 1 and fvo[t] >= -cfg.exit_level) or \
                       (current == -1 and fvo[t] <= cfg.exit_level)
            if exhausted or reverted:
                r_trade = current * (np.log(price[t]) - np.log(price[entry_i])) - 2 * cost
                trades.append(dict(
                    entry=dates[entry_i].date(), exit=dates[t].date(),
                    side="LONG" if current == 1 else "SHORT",
                    sessions=held, entry_fvo=float(fvo[entry_i]),
                    exit_fvo=float(fvo[t]),
                    reason="reverted" if reverted else "max hold",
                    ret_pct=100.0 * (np.exp(r_trade) - 1.0)))
                current, size, held = 0, 0.0, 0

        if current == 0 and conf[t] >= cfg.confidence_floor:
            long_ok = np.isfinite(os_[t]) and fvo[t] < os_[t]
            short_ok = cfg.allow_short and np.isfinite(ob[t]) and fvo[t] > ob[t]
            if long_ok or short_ok:
                current = 1 if long_ok else -1
                size = (conf[t] / 100.0) if cfg.size_by_confidence else 1.0
                entry_i, held = t, 0

        pos[t + 1] = current * size

    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * np.nan_to_num(rets) - turnover * cost
    strat[:start] = 0.0

    equity = pd.Series(np.exp(np.cumsum(strat)), index=dates)
    bh_r = np.nan_to_num(rets).copy()
    bh_r[:start] = 0.0
    buyhold = pd.Series(np.exp(np.cumsum(bh_r)), index=dates)

    tr = pd.DataFrame(trades)
    active = strat[start:]
    n_years = max(len(active) / 252.0, 1e-9)

    def _sharpe(x: np.ndarray) -> float:
        s = np.std(x)
        return float(np.mean(x) / s * np.sqrt(252)) if s > 1e-12 else np.nan

    dd = equity / equity.cummax() - 1.0
    wins = tr["ret_pct"] > 0 if len(tr) else pd.Series(dtype=bool)
    gross_win = tr.loc[wins, "ret_pct"].sum() if len(tr) else 0.0
    gross_loss = -tr.loc[~wins, "ret_pct"].sum() if len(tr) else 0.0

    stats = {
        "trades": int(len(tr)),
        "win rate %": 100.0 * float(wins.mean()) if len(tr) else np.nan,
        "avg trade %": float(tr["ret_pct"].mean()) if len(tr) else np.nan,
        "best %": float(tr["ret_pct"].max()) if len(tr) else np.nan,
        "worst %": float(tr["ret_pct"].min()) if len(tr) else np.nan,
        "avg hold (d)": float(tr["sessions"].mean()) if len(tr) else np.nan,
        "profit factor": float(gross_win / gross_loss) if gross_loss > 0 else np.nan,
        "sharpe": _sharpe(active),
        "cagr %": 100.0 * (float(equity.iloc[-1]) ** (1 / n_years) - 1.0),
        "max drawdown %": 100.0 * float(dd.min()),
        "exposure %": 100.0 * float(np.mean(np.abs(pos[start:]) > 0)),
        "bh sharpe": _sharpe(bh_r[start:]),
        "bh cagr %": 100.0 * (float(buyhold.iloc[-1]) ** (1 / n_years) - 1.0),
        "bh max dd %": 100.0 * float((buyhold / buyhold.cummax() - 1.0).min()),
    }

    return dict(equity=equity, buyhold=buyhold, drawdown=dd,
                position=pd.Series(pos, index=dates), trades=tr, stats=stats,
                daily=pd.Series(strat, index=dates), start=start)


# ---------------------------------------------------------------------------
# Parameter-free signal evaluation
# ---------------------------------------------------------------------------
def information_coefficients(E: dict, horizons=(1, 5, 10, 21, 63)) -> pd.DataFrame:
    """Rank correlation between the oscillator and subsequent returns.

    Sign convention: the oscillator is *negated* first, so a positive IC means
    the signal works as intended — a rich reading (high FVO) precedes weak
    returns and a cheap reading precedes strong ones.
    """
    fvo = E["fvo"]
    logp = np.log(E["price"])
    start = E["lookback"] + int(E["config"].fv_horizon)

    rows = []
    for h in horizons:
        fwd = (logp.shift(-h) - logp)
        pair = pd.concat([-fvo, fwd], axis=1).iloc[start:].dropna()
        if len(pair) < 60:
            rows.append({"horizon (d)": h, "IC": np.nan, "p-value": np.nan, "n": len(pair)})
            continue
        ic, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append({"horizon (d)": h, "IC": float(ic), "p-value": float(p),
                     "n": int(len(pair))})
    return pd.DataFrame(rows)


def decile_forward_returns(E: dict, horizon: int = 10, n_buckets: int = 10) -> pd.DataFrame:
    """Mean forward return by oscillator bucket — the signal's shape.

    A monotone, downward-sloping profile (cheap buckets earn more than rich
    ones) is the evidence that the oscillator ranks opportunity. Buckets are
    formed on the full sample, so this is a descriptive study of the signal,
    not a tradable simulation.
    """
    fvo = E["fvo"]
    logp = np.log(E["price"])
    fwd = (logp.shift(-horizon) - logp) * 100.0
    start = E["lookback"] + int(E["config"].fv_horizon)

    df = pd.concat([fvo.rename("fvo"), fwd.rename("fwd")], axis=1).iloc[start:].dropna()
    if len(df) < n_buckets * 10:
        return pd.DataFrame()

    try:
        df["bucket"] = pd.qcut(df["fvo"], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    g = df.groupby("bucket")
    out = pd.DataFrame({
        "bucket": g.size().index.astype(int) + 1,
        "FVO range": [f"{g.get_group(b)['fvo'].min():+.0f} … "
                      f"{g.get_group(b)['fvo'].max():+.0f}" for b in g.size().index],
        "sessions": g.size().values,
        f"mean fwd {horizon}d %": g["fwd"].mean().values,
        "hit rate %": 100.0 * g["fwd"].apply(lambda x: float((x > 0).mean())).values,
    })
    return out.reset_index(drop=True)


def signal_event_study(E: dict, horizon: int = 21) -> pd.DataFrame:
    """Average path of the target after each entry signal fires."""
    logp = np.log(E["price"])
    out = {}
    for name, mask in (("undervalued entry", E["long_entry"]),
                       ("overvalued entry", E["short_entry"]),
                       ("bullish divergence", E["div_bull"]),
                       ("bearish divergence", E["div_bear"])):
        idx = np.flatnonzero(mask.values)
        idx = idx[(idx > E["lookback"]) & (idx < len(logp) - horizon)]
        if len(idx) < 3:
            continue
        paths = np.vstack([logp.values[i:i + horizon + 1] - logp.values[i] for i in idx])
        out[f"{name} (n={len(idx)})"] = 100.0 * paths.mean(axis=0)
    return pd.DataFrame(out, index=range(horizon + 1))
