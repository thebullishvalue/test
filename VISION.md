# ARTHAGATI — Architecture & Design Vision
### @thebullishvalue · Architect's Working Document · v2.6.0

---

## §1  System Purpose

Arthagati answers one question: **"What is the market's current sentiment state, and how confident should I be in that reading?"**

It does this through six steps:

1. Ingest **macro / breadth / valuation** inputs (PE, EY, yields, breadth, policy rates)
2. Compute **correlations** between these variables and the valuation anchors (PE, EY)
3. Construct a **weighted composite sentiment score** — the Mood Score
4. Build a **multi-component oscillator** (MSF Spread) for momentum/structure confirmation
5. Find **historical analogs** with similar market states
6. Show **correlation structure** for full transparency

Every mathematical choice must serve one of these six steps. If a theory does not improve any step, it does not belong.

---

## §2  Candidate Theories — Full Evaluation Table

### A — Correlation Estimation

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| Pearson correlation | Linear association, full sample | **v1.x** — assumes linearity and stationarity |
| **Exponential-decay Spearman** | Monotonic correlation, recency-weighted | ✅ **Adopted** — macro↔valuation relationships are monotonic; regime structure is non-stationary |
| Mutual Information | Captures arbitrary nonlinear dependence | ⚠️ Overkill — relationships are monotonic, not U-shaped |
| DCC-GARCH | Time-varying correlation model | ❌ Heavy, requires MLE, fragile on ~1,500 rows |
| Copulas | Joint distribution modelling | ❌ Overkill for weight estimation |
| Kendall's τ | Concordance-based rank correlation | ⚠️ Similar to Spearman, slower, no clear advantage |

### B — Variable Weighting

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| Correlation magnitude (`\|corr\|`) | Direct weight | **v1.x** — treats noisy high-corr variables the same as stable ones |
| **Shannon entropy penalty** | `weight = \|corr\| × (1 − entropy)` | ✅ **Adopted** — suppresses noisy/random variables |
| Mutual Information weighting | `weight = MI(var, anchor)` | ⚠️ More principled but Spearman + entropy achieves 90% of the benefit |
| LASSO / Ridge regression | Penalised regression weights | ❌ Wrong framing — this is composite construction, not prediction |
| PCA loadings | Variance-explained weights | ❌ Loses interpretability, unclear sign convention |

### C — Historical Positioning (Percentiles)

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| Expanding rank | Percentile against full history | **v1.x** — 2005 data pollutes 2025 percentiles |
| **Decay-weighted empirical CDF** | Recent history weighted more | ✅ **Adopted** — half-life `PCT_HALF_LIFE = 252` trading days |
| Kernel Density Estimation | Full distribution estimate | ⚠️ More than needed — percentile is the sufficient statistic |
| Regime-conditional percentile | Percentile within detected regime | ⚠️ Requires regime detection first — circular dependency |

### D — Score Normalization

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| Global z-score | `(x − μ_all) / σ_all × 30` | **v1.x** — adding one point shifts all of history |
| Expanding z-score | Running mean/std | Better, but still treats history equally |
| **Ornstein-Uhlenbeck** | Mean-reverting diffusion model | ✅ **Adopted** — mood IS mean-reverting; OU gives natural units, equilibrium, reversion speed, and half-life |
| Quantile normalization | Rank → uniform → normal | ❌ Destroys magnitude information |

### E — Smoothing

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| Simple Moving Average | Fixed window average | **v1.x** — arbitrary window, uniform weights |
| EMA | Exponential moving average | Better than SMA, still fixed bandwidth |
| **Kalman filter (1D)** | Adaptive state estimation | ✅ **Adopted** — auto-adjusts smoothing to signal-to-noise ratio; outputs `±KALMAN_CI_Z σ` confidence band |
| Savitzky-Golay | Polynomial smoothing | ⚠️ Preserves peaks but fixed bandwidth |
| Hodrick-Prescott | Trend-cycle decomposition | ❌ Endpoint instability |
| Wavelet denoising | Multi-scale decomposition | ❌ Overkill, hard to interpret |

### F — Oscillator Construction (MSF Spread)

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| Fixed weights | Arbitrary allocation (30/25/25/20) | **v1.x** — no basis |
| **Inverse-variance weighting** | Stable signals receive more weight | ✅ **Adopted** — Markowitz minimum-variance portfolio of signals |
| Fixed regime threshold (0.0033) | One directional threshold | **v1.x** — arbitrary |
| **Adaptive threshold** | Scales with local volatility | ✅ **Adopted** — a move is "directional" only if it exceeds local noise |
| Entropy / Hurst as MSF components | Disorder/persistence as signal inputs | ❌ Dilutes the oscillator's purpose (momentum/structure alignment) |

### G — Diagnostics

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| **Hurst exponent (R/S)** | Trending (H > 0.5) vs mean-reverting (H < 0.5) | ✅ **Adopted** as diagnostic only — tells the user if mood is likely to persist or reverse |
| **Shannon entropy (rolling)** | Market disorder measure | ✅ **Adopted** as diagnostic only — tells the user if the market is choppy |
| **OU half-life** | `ln(2)/θ` — expected time to halve deviation | ✅ **Adopted** as diagnostic only — "this extreme should normalise in ~X days" |
| Fisher Information | `1/variance` — signal confidence | ❌ Just inverse rolling variance; not worth the naming overhead |
| Lyapunov exponents | Chaos detection | ❌ Unstable estimates, requires > 10,000 points |
| Fractal dimension | Path complexity | ❌ Hurst already captures this (`D = 2 − H`) |

### H — Similar Period Matching

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| Manhattan distance (2 features) | Absolute difference | **v1.x** — ignores covariance, too few features |
| **Mahalanobis distance** | Covariance-aware distance on 5-feature vector | ✅ **Adopted** — correlated features don't double-count; weight `SIMILAR_W_MAHA = 0.55` |
| **Cosine similarity on trajectories** | Detrended `TRAJ_WINDOW`-day path shape | ✅ **Adopted** — lightweight DTW approximation; weight `SIMILAR_W_TRAJ = 0.35` |
| **Exponential recency decay** | Prefer recent analogs | ✅ **Adopted** — weight `SIMILAR_W_RECV = 0.10` |
| Dynamic Time Warping (DTW) | Time-elastic sequence matching | ⚠️ More principled but O(n²), too heavy |
| k-NN with rich features | Multi-feature nearest neighbour | Essentially what Mahalanobis achieves |

### I — Data Engineering

| Theory | What It Does | Verdict |
|--------|-------------|---------|
| Raw yields (`IN10Y`, `IN02Y`, etc.) | Yields as independent predictors | **v1.x** — raw yields are correlated; spreads carry orthogonal information |
| **Term spread extraction** | `10Y − 2Y` yield curve slope | ✅ **Adopted** — `IN_TERM_SPREAD` and `US_TERM_SPREAD` derived in `load_data()` |
| Real rate | Nominal yield − inflation | ⚠️ Useful but adds complexity |
| Sovereign spread | `IN10Y − US10Y` | ⚠️ Useful but adds complexity |

---

## §3  Curation Matrix

**Principle: every theory gets exactly one job. No theory appears in two layers.**

| Theory | Verdict | Layer | Job |
|--------|---------|-------|-----|
| Exponential-decay Spearman | ✅ KEEP | Correlation | Replace static Pearson |
| Shannon entropy (variable weighting) | ✅ KEEP | Weighting | Penalise noisy variables |
| Adaptive percentile (decay ECDF) | ✅ KEEP | Positioning | Replace expanding rank |
| Ornstein-Uhlenbeck estimation | ✅ KEEP | Normalization | Physics-based scaling + half-life diagnostic |
| Kalman filter (1D) | ✅ KEEP | Smoothing | Adaptive noise filtering |
| Inverse-variance weighting | ✅ KEEP | MSF Oscillator | Replace fixed weights |
| Adaptive regime threshold | ✅ KEEP | MSF Oscillator | Replace fixed 0.0033 |
| Hurst exponent (rolling) | ✅ KEEP | Diagnostics ONLY | Output: trending vs reverting |
| Shannon entropy (rolling) | ✅ KEEP | Diagnostics ONLY | Output: market disorder |
| OU half-life | ✅ KEEP | Diagnostics ONLY | Output: expected normalisation time |
| Mahalanobis distance | ✅ KEEP | Similar Periods | Covariance-aware state matching |
| Cosine trajectory similarity | ✅ KEEP | Similar Periods | Path shape matching |
| Term spread extraction | ✅ KEEP | Data Engineering | Derive `IN/US_TERM_SPREAD` |
| Entropy gate on mood score | ❌ CUT | — | Over-compresses legitimate signals during volatile regimes |
| Hurst-adjusted classification | ❌ CUT | — | Hurst estimates are noisy (±0.1); dynamic thresholds create instability |
| Entropy / Hurst as MSF components | ❌ CUT | — | Dilutes the oscillator's purpose |
| Fisher Information column | ❌ CUT | — | Just `1/variance`; not actionable |
| Mutual Information | ❌ CUT | — | Spearman + entropy achieves 90% of the benefit at lower cost |
| DCC-GARCH | ❌ CUT | — | Too heavy for a Streamlit app on ~1,500 rows |
| Wavelet / Lyapunov / Fractal | ❌ CUT | — | Overkill; estimates unstable on this data volume |

---

## §4  Final Architecture

### Mood Score Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│  Google Sheets (gviz/tq?tqx=out:csv) → Derive Term Spreads         │
│  IN_TERM_SPREAD = IN10Y − IN02Y  |  US_TERM_SPREAD = US10Y − US02Y │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LAYER 1 — ADAPTIVE CORRELATIONS                        │
│  Exponential-decay weighted Spearman rank correlation               │
│  Each predictor → PE anchor correlation, EY anchor correlation      │
│  Half-life: CORR_HALF_LIFE = 504 days (~2 trading years)           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│         LAYER 2 — INFORMATION-THEORETIC WEIGHTING                   │
│  weight = |correlation| × (1 − normalized_entropy(variable))       │
│  Entropy computed on each variable's returns distribution           │
│  Noisy / random variables suppressed; structured signals amplified  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│           LAYER 3 — ADAPTIVE PERCENTILES                            │
│  Decay-weighted empirical CDF                                       │
│  Half-life: PCT_HALF_LIFE = 252 trading days (~1 year)             │
│  "Where is PE today vs recent-ish history?" — not vs all-time      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│           LAYER 4 — OU NORMALIZATION                                │
│  Fit Ornstein-Uhlenbeck process: dx = θ(μ − x)dt + σdW            │
│  score = (x − μ) / (σ / √2θ) × MOOD_SCALE  →  [−100, +100]       │
│  Diagnostics: half-life = ln(2)/θ,  equilibrium = μ               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│            LAYER 5 — KALMAN SMOOTHING                               │
│  1D Kalman filter with auto-estimated noise parameters              │
│  High noise → more smoothing  |  Low noise → tracks signal         │
│  Output: smoothed score + ±KALMAN_CI_Z σ confidence band           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│          OUTPUT — MOOD SCORE + DIAGNOSTICS                          │
│  Mood Score: [−100, +100] with fixed classification thresholds      │
│  Diagnostics (enriched columns — do NOT feed back into score):      │
│    • Hurst exponent    → trending vs mean-reverting character       │
│    • Market entropy    → ordered vs disordered regime               │
│    • OU half-life      → expected days to normalise from extreme    │
│    • Kalman confidence → ±1.96σ band width                         │
└─────────────────────────────────────────────────────────────────────┘
```

### MSF Spread (Parallel Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Component 1: Momentum  — NIFTY ROC z-score (MSF_ROC_LEN = 14d)   │ ─┐
│  Component 2: Structure — Mood trend divergence + acceleration      │  │ Inverse-Variance
│  Component 3: Regime    — Adaptive-threshold directional count      │  │ Weighting
│  Component 4: Flow      — Breadth participation divergence          │  │ (per MSF_WINDOW = 20)
└─────────────────────────────────────────────────────────────────────┘  │
                                                                          ▼
                                                               MSF Spread [−10, +10]
```

### Similar Periods (Matching Engine)

```
Feature vector = [mood, volatility, NIFTY_ROC, Hurst, entropy]
                              │
               ┌──────────────┴───────────────┐
               ▼                              ▼
    Mahalanobis Distance          Cosine Trajectory Similarity
    state matching                TRAJ_WINDOW = 20-day path shape
    SIMILAR_W_MAHA = 55%          SIMILAR_W_TRAJ = 35%
               │                              │
               └──────────────┬───────────────┘
                              ▼
               Exponential Recency Decay
               SIMILAR_W_RECV = 10%  |  half-life = 365d
                              │
                              ▼
                   Combined Similarity Score
                   → Top-N analogs + forward returns
```

---

## §5  What v2.0 Removed from the Draft Overhaul

| Feature | Reason for removal |
|---------|-------------------|
| Entropy gate on mood scores | Compresses legitimate signals during volatile regimes. If the market IS chaotic, the mood score should reflect that chaos — not be silenced. |
| Hurst-adjusted classification thresholds | Hurst estimates carry ±0.1 variance. Using them to dynamically shift classification thresholds makes the system unstable. A "Bullish" reading flickering to "Neutral" because Hurst jittered is worse than fixed thresholds. |
| Entropy as MSF component | The MSF measures momentum ↔ sentiment alignment. Entropy dilutes this. Entropy belongs in diagnostics only. |
| Hurst as MSF component | Same reasoning. Persistence is a meta-property of the signal, not a signal component. |
| Fisher Information output | It is `1/rolling_variance`. Naming it, computing it, and surfacing it adds complexity for something the user can eyeball from the volatility column. |
| 12 primitive functions | Reduced to 11. Each has exactly one callsite and one purpose. |

---

## §6  Key Design Decisions

**Q: Why fixed mood classification thresholds (±20, ±60) instead of adaptive?**

Stability. When a user sees "Bullish" today and "Bullish" yesterday, those labels should mean the same thing. Adaptive thresholds change the meaning of "Bullish" over time. The OU normalization already handles scale adaptation — the classification layer should remain stable.

---

**Q: Why Hurst and entropy as diagnostics only, not score inputs?**

They inform the *interpretation* of the mood score, not its *value*. "Mood is +45 (Bullish), Hurst is 0.62 (trending), entropy is low (ordered)" tells the user: "This bullish reading is in a trending, ordered market — trust it." Embedding Hurst into the score would remove this interpretive layer entirely.

---

**Q: Why inverse-variance weighting for MSF but not for mood?**

The mood score is a correlation-weighted composite of percentiles — not a portfolio of signals. Its weights come from correlation strength × entropy quality, which is the correct weighting for that construction. The MSF components are independent technical indicators that *are* a signal portfolio, and minimum-variance allocation (Markowitz) is the natural fit.

---

**Q: Why term spreads but not real rates or sovereign spreads?**

The 10Y−2Y spread is the single most studied and validated macro signal in financial economics — every US recession since 1960 was preceded by inversion. It adds genuine orthogonal information relative to raw yields. Real rates and sovereign spreads add marginal value but would double the number of derived features without proportional benefit.

---

**Q: Why `gviz/tq?tqx=out:csv` for the Sheets fetch?**

The Google Visualization endpoint returns a well-structured CSV with no authentication overhead. The sheet must be set to "Anyone with the link can view" — no OAuth, no service accounts, no `st.secrets`. Sheet coordinates (`ARTHAGATI_SHEET_ID`, `ARTHAGATI_SHEET_GID`) are read from environment variables, keeping the codebase free of hardcoded credentials and simplifying deployment across any environment (local, Streamlit Cloud, Docker, etc.).

The architecture is:
- `_fetch_sheet_csv()` — fetches via gviz endpoint with retry + exponential backoff
- `load_data()` — parses and transforms, knows nothing about auth
- Environment variables — the only configuration; `.gitignore` ensures `.env` is never committed

---

**Q: How is memory managed during rapid UI interactions?**

Streamlit's `@st.cache_data` is used for all heavy mathematical pipelines. To prevent unbounded RAM growth when users frequently change predictor configurations, caches are strictly bounded using `max_entries=5`. This ensures the server holds only the most recent analytical states, automatically evicting old DataFrames.

---

**Q: Why use WebGL (`go.Scattergl`) with interleaved `None` for vertical lines instead of Plotly's `add_vline`?**

Rendering hundreds of regime transitions via `add_vline` creates individual SVG layout shapes, causing severe DOM bloat and browser lag. Grouping lines by color and interleaving coordinates with `None` in a single WebGL trace offloads rendering to the GPU, making interactions (panning/zooming) instantaneous even over 15-year timeframes.

---

*© 2026 Arthagati · @thebullishvalue*
