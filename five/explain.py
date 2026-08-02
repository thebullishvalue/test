"""
Explainability for the fair-value model.

The fair-value map is linear in a transformed feature space, which is a real
advantage: attribution is *exact* rather than approximated. βⱼ·xⱼ decomposes
the model-implied return into a contribution per latent factor and per
orthogonal peer, in basis points per day, with no surrogate model in between.

Three complementary views are produced:

  * **Contribution** — signed, exact, per day. What is pushing fair value?
  * **Permutation importance** — how much explanatory power is lost when a
    feature is scrambled. What does the model *rely* on?
  * **SHAP** — the game-theoretic allocation. For a linear model with the
    interventional/independent-feature convention this coincides with
    βⱼ·(xⱼ − E[xⱼ]); we compute it through ``shap`` when available and fall
    back to the closed form otherwise.

And, because latent factors are unnamed by construction, the loadings are
mapped back onto the instruments and asset classes that define each one — the
step that turns "F3" into "the dollar / commodity axis".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import _residualiser_apply
from .universe import CLASS_OF


def _rebuild_design(E: dict, calib=None, window: str = "train"
                    ) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Reconstruct the exact design matrix a calibration saw.

    ``window='train'`` returns the calibration window, ``'forward'`` the block
    the model was actually responsible for forecasting.
    """
    calib = calib or E["last"]
    Z, others, target = E["Z"], E["others"], E["target"]
    lookback = E["lookback"]

    if window == "forward":
        sl = slice(calib.start, calib.end)
    else:
        sl = slice(calib.start - lookback, calib.start)

    X = Z[others].values[sl]
    S = calib.factors.transform(X)
    R = _residualiser_apply(calib.resid_coef, S, X)
    pos = [others.index(p) for p in calib.peers]
    D = np.column_stack([S, R[:, pos]]) if pos else S
    Ds = (D - calib.design_mean) / calib.design_scale
    y = E["rets"][target].values[sl]
    return Ds, y, list(calib.feature_names), E["dates"][sl]


def linear_contributions(E: dict, n_sessions: int = 60) -> pd.DataFrame:
    """Recent per-feature contributions in basis points per day."""
    contrib = E["contrib"]
    if contrib.empty:
        return contrib
    return contrib.tail(n_sessions)


def contribution_by_class(contrib_row: pd.Series) -> pd.Series:
    """Aggregate one session's contributions up to the asset-class level."""
    groups = [("Latent Factor" if str(c).startswith("F") and str(c)[1:].isdigit()
               else CLASS_OF.get(c, "Other")) for c in contrib_row.index]
    agg = contrib_row.groupby(groups).sum()
    return agg.reindex(agg.abs().sort_values(ascending=False).index)


def permutation_importance(E: dict, n_repeats: int = 5, seed: int = 0) -> pd.Series:
    """Drop in R² when each feature is independently scrambled.

    Computed on the calibration window, so read it as "what this model leans
    on", not as an out-of-sample generalisation claim.
    """
    calib = E["last"]
    D, y, names, _ = _rebuild_design(E, calib, window="train")
    pred = D @ calib.coef + calib.intercept
    sst = float(((y - y.mean()) ** 2).sum()) or 1.0
    base = 1.0 - float(((y - pred) ** 2).sum()) / sst

    rng = np.random.default_rng(seed)
    out = {}
    for j, name in enumerate(names):
        losses = []
        for _ in range(n_repeats):
            Dp = D.copy()
            Dp[:, j] = rng.permutation(Dp[:, j])
            p = Dp @ calib.coef + calib.intercept
            losses.append(base - (1.0 - float(((y - p) ** 2).sum()) / sst))
        out[name] = max(float(np.mean(losses)), 0.0)
    return pd.Series(out).sort_values(ascending=False)


def shap_values(E: dict, n_sessions: int = 60) -> tuple[pd.DataFrame, str]:
    """Per-session SHAP attributions, in basis points per day.

    Returns ``(frame, backend)`` where backend names how it was computed.
    """
    calib = E["last"]
    D, _, names, idx = _rebuild_design(E, calib, window="train")
    D_recent = D[-n_sessions:]
    idx_recent = idx[-n_sessions:]

    try:
        import shap

        explainer = shap.LinearExplainer(
            (calib.coef, calib.intercept), D,
            feature_perturbation="interventional")
        vals = np.asarray(explainer.shap_values(D_recent))
        backend = "shap.LinearExplainer"
    except Exception:
        vals = (D_recent - D.mean(axis=0)) * calib.coef
        backend = "closed form β·(x − E[x])"

    return pd.DataFrame(vals * 1e4, index=idx_recent, columns=names), backend


def factor_interpretation(E: dict, top_n: int = 8) -> pd.DataFrame:
    """Name each latent factor by the instruments that define it.

    Rows: factor. Columns: signed exposure, class tilt, and the strongest
    positive/negative loadings.
    """
    calib = E["last"]
    L = calib.factors.loadings                       # (k, n_instruments)
    others = E["others"]
    ev = np.asarray(calib.factors.explained, dtype="float64")

    rows = []
    for j in range(L.shape[0]):
        w = pd.Series(L[j], index=others)
        top_pos = w.nlargest(top_n)
        top_neg = w.nsmallest(top_n)
        cls_tilt = w.groupby([CLASS_OF.get(s, "Other") for s in others]).sum()
        cls_tilt = cls_tilt.reindex(cls_tilt.abs().sort_values(ascending=False).index)
        rows.append({
            "factor": f"F{j + 1}",
            "variance %": 100.0 * ev[j] if j < len(ev) and np.isfinite(ev[j]) else np.nan,
            "β (bp/day per σ)": calib.coef[j] * 1e4,
            "dominant classes": ", ".join(f"{k} {v:+.2f}" for k, v in cls_tilt.head(3).items()),
            "loads long": ", ".join(top_pos.index[:5]),
            "loads short": ", ".join(top_neg.index[:5]),
        })
    return pd.DataFrame(rows)


def factor_loadings_frame(E: dict, factor: int = 0, top_n: int = 20) -> pd.Series:
    """The ``top_n`` largest absolute loadings on one factor, signed."""
    calib = E["last"]
    w = pd.Series(calib.factors.loadings[factor], index=E["others"])
    return w.reindex(w.abs().sort_values(ascending=False).index).head(top_n)


def coefficient_history(E: dict) -> pd.DataFrame:
    """Latent-factor coefficient paths across calibrations.

    Meaningful only because components are sign- and order-aligned between
    refits; without that alignment this plot would be pure sign flipping.
    """
    k = E["config"].n_factors
    rows, idx = [], []
    for c in E["calibrations"]:
        kk = min(k, len(c.coef))
        rows.append({f"F{i + 1}": c.coef[i] * 1e4 for i in range(kk)})
        idx.append(E["dates"][c.start])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def peer_persistence(E: dict) -> pd.DataFrame:
    """How often each instrument was selected as an orthogonal peer.

    A peer set that churns every refit is a warning sign: the model is
    chasing noise rather than tracking a stable economic relationship.
    """
    counts: dict[str, int] = {}
    for c in E["calibrations"]:
        for p in c.peers:
            counts[p] = counts.get(p, 0) + 1
    total = max(len(E["calibrations"]), 1)
    df = pd.DataFrame({
        "instrument": list(counts.keys()),
        "selected": list(counts.values()),
    })
    if df.empty:
        return df
    df["share %"] = 100.0 * df["selected"] / total
    df["class"] = [CLASS_OF.get(s, "Other") for s in df["instrument"]]
    df["in latest set"] = [s in E["last"].peers for s in df["instrument"]]
    return df.sort_values("selected", ascending=False).reset_index(drop=True)


def driver_table(E: dict, n: int = 15) -> pd.DataFrame:
    """The current session's strongest drivers, ready for display."""
    contrib = E["contrib"]
    if contrib.empty:
        return pd.DataFrame()
    row = contrib.iloc[-1].dropna()
    row = row.reindex(row.abs().sort_values(ascending=False).index).head(n)
    return pd.DataFrame({
        "driver": row.index,
        "class": [("Latent Factor" if str(c).startswith("F") and str(c)[1:].isdigit()
                   else CLASS_OF.get(c, "Other")) for c in row.index],
        "contribution (bp/day)": row.values,
        "direction": ["pushes fair value up" if v > 0 else "pushes fair value down"
                      for v in row.values],
    })
