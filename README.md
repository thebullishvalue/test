# FVE — Fair Value Engine

A market-relative valuation model. It estimates what one asset *should* be
worth given the state of everything else, and publishes the deviation as a
bounded, adaptively-normalised oscillator.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first live download of a few hundred instruments takes several minutes and
is cached to disk (`.fve_cache/`), so later runs start immediately. **Demo
mode** in the sidebar runs instantly offline against a synthetic market.

---

## What the model says

At each session the cross-section of 200+ instruments is compressed into a
handful of latent factors, plus a set of **orthogonal peers** — instruments
that still say something about the target *after* the common factors are
removed. A walk-forward, regime-weighted ridge regression maps that market
state onto the target's return.

Fair value is that implied return rolled forward from an anchor `H` sessions
back:

$$FV_t = P_{t-H}\cdot\exp\Big(\textstyle\sum_{s=t-H+1}^{t}\hat{y}_s\Big)$$

so the mispricing

$$m_t = \log(P_t/FV_t) = \underbrace{\text{realised } H\text{-period return}}_{\text{what the asset did}} - \underbrace{\text{implied } H\text{-period return}}_{\text{what the market implied}}$$

is exactly the target's cumulative excess move against the rest of the world
over the last `H` sessions.

**Why anchored rather than cumulative.** The obvious alternative — compound
implied returns from a single fixed start date — makes the gap an integral of
residuals. That is a random walk: it drifts arbitrarily far from price, its
variance grows without bound, and "20% overvalued" ends up meaning "the model
has been accumulating small errors for three years". The anchored form is
stationary by construction and reads as a valuation at every point.

The oscillator is then

$$FVO_t = 100\cdot\tanh\!\big(z_t/\kappa\big),\qquad z_t = \frac{m_t - \text{med}_w(m)}{1.4826\cdot\text{MAD}_w(m)}$$

Adaptive normalisation is what makes a 2019 reading comparable to a 2020 one.
`tanh` rather than a hard clip, because clipping destroys all information past
the boundary — every extreme prints the same −100 — while `tanh` compresses
monotonically.

---

## Choosing a target

The **Asset class** selector drives what you can value:

* **US Stocks** and **India Stocks** take a typed ticker. The symbol is
  resolved against the provider before the run is allowed — India tries
  `SYMBOL.NS` then `SYMBOL.BO` (NSE wins for dual listings), US uppercases and
  maps `.` to `-` (`BRK.B` → `BRK-B`). An unresolvable symbol is rejected with
  the candidates it tried, rather than failing several minutes later mid-run.
* **Every other class** — indices, sector/style/regional ETFs, India indices,
  commodities, FX, crypto, rates, credit, volatility, real assets — is a
  dropdown drawn from the explanatory universe itself.

There is deliberately no curated list of individual US company stocks. A fixed
basket of a hundred names is simultaneously too long to browse and too short to
contain whatever you actually wanted; free-form entry covers every listed
equity on either market.

**Nothing runs until you press RUN ANALYSIS.** Streamlit reruns the whole
script on every widget interaction, so binding the engine directly to the
controls would fire a multi-minute download and a full recalibration each time
you touched one — including part-way through choosing. Selections are staged
and committed as one atomic set. Change something after a run and the previous
results stay on screen under a notice that they are stale.

---

## Architecture

```
app.py                 Streamlit dashboard — presentation and wiring only
fve/
  universe.py          237 explanatory instruments, target categories, palette
  config.py            EngineConfig / DataConfig / BacktestConfig
  cache.py             two-tier TTL cache with last-good-snapshot fallback
  circuit_breaker.py   circuit breaker + retry-with-backoff
  data.py              fault-tolerant fetch, calendar alignment, simulator
  features.py          returns, causal standardisation, sample weights
  factors.py           PCA / ICA / FactorAnalysis / Autoencoder + alignment
  regimes.py           walk-forward k-means regime classification
  oscillator.py        normalisation, Kalman/EMA smoothing, OU, divergences
  engine.py            the walk-forward calibration loop
  explain.py           attribution, permutation importance, SHAP, factor naming
  backtest.py          signal replay + parameter-free evidence
  viz.py / theme.py    Plotly builders and CSS shell
```

### The universe is aggregates, not single names

The explanatory cross-section is built from indices, sector/style/regional
ETFs, rates, credit, commodities, FX, crypto, volatility and real assets — 237
instruments across 13 classes. Individual company stocks are deliberately
absent from it: a basket of them adds little that a sector ETF does not already
carry, while multiplying idiosyncratic noise and download time.

The **India Equity** block (Nifty sectoral indices, NSE ETFs, US-listed India
funds) exists because an India stock target needs local conditioning. Pricing
an NSE name purely off US macro would be a materially worse model, and an
India target also switches the calendar anchor to the NSE session so local
holidays are not silently dropped.

### Fetching (adapted from the Tattva system)

Every provider call is wrapped in four layers:

1. **Retry with backoff** for transient failures (1.5s → 3s → 6s).
2. **A circuit breaker** for sustained ones. After five consecutive failures
   the circuit opens and calls fail instantly instead of burning 30 seconds
   each; one probe is allowed through after the recovery timeout.
3. **A two-tier cache** (memory + disk, TTL + versioned keys). Expired entries
   are *retained*, so `get_stale` can serve last-good data during an outage —
   a naive cache is fresh-or-nothing and shows an error page instead.
4. **Partial-success completion.** A batch is one call, but Yahoo rate-limits a
   few tickers per batch, so it returns incomplete. The missing symbols get one
   targeted re-fetch; whatever is still absent is backfilled from the newest
   snapshot — unless that snapshot's own last observation is more than 10
   sessions stale, in which case the column is dropped rather than
   forward-filled flat across weeks.

The **panel and the target are fetched separately**. The panel is
target-agnostic and caches as one unit keyed by (symbols, years); a free-form
equity is a one-symbol fetch joined on afterwards. Folding the target into the
batch would give every new ticker a different cache key and force a full
re-download each time — the single most expensive mistake available here.

Provider health, cache hit rates and any snapshot backfills are surfaced in the
**Universe & Data** tab rather than hidden in logs.

Everything in `fve/` is pure Python with no Streamlit dependency, so the engine
can be driven from a notebook or a scheduler:

```python
from fve.config import EngineConfig, DataConfig
from fve.data import load_universe, resolve_symbol
from fve.engine import run_engine, snapshot
from fve.universe import symbols_for_tier

ticker, exchange = resolve_symbol("RELIANCE", "india")     # -> ('RELIANCE.NS', 'NSE')
symbols = symbols_for_tier("core") + [ticker]
bundle = load_universe(symbols, DataConfig(years=8), target=ticker)
E = run_engine(bundle.prices, EngineConfig(target=ticker))
print(snapshot(E))
```

---

## Five things that are easy to get wrong, and how they're handled

**1. Look-ahead in the regime labels.** Clustering the market state over the
full sample is the standard shortcut, and it leaks the future into every
regime-conditioned weight and every backtest downstream. Here the classifier is
refit walk-forward on trailing data only.

**2. Factor sign flips.** PCA components carry an arbitrary sign and ICA an
arbitrary order. Refit every 21 sessions without alignment and "factor 3" means
something different each time — coefficient paths become noise and stability
diagnostics measure nothing but sign flips. Successive loading matrices are
matched by optimal assignment and sign-corrected.

**3. Scalers that leak.** A factor model fitted on standardised data must be
applied using the *fit window's* mean and scale. Recomputing them on the
forecast window is a subtle look-ahead; the scaler is frozen inside the fitted
object.

**4. Near-perfect proxies.** Explain QQQ using `^NDX` and fair value collapses
onto price — the oscillator becomes pure noise with an impressive R². Anything
correlated above a configurable threshold (default 0.99) is excluded from the
explanatory set. The test deliberately runs on **raw** returns: measured in the
standardised feature space, rolling volatility scaling attenuates correlation
by 2-3 points, which is enough for a 0.9994-correlated index proxy to slip
under a 0.99 threshold. On live QQQ, testing in the wrong space admitted `^NDX`
and inflated out-of-sample R² from 0.90 to 0.95.

Tightening the threshold trades fit for orthogonality. At 0.99 the peer set for
QQQ keeps `SPY` and `^GSPC` — which is the point, since "QQQ rich against the
S&P" is a genuine relative-value statement. Around 0.90 the index proxies drop
out entirely and the peers become sector and factor exposures.

**5. The trading calendar.** Crypto trades weekends. Take the union of all
instrument calendars and you insert rows that are empty for every other asset,
then smear crypto weekend moves across the panel on forward-fill. The panel is
aligned to a liquid equity reference instead.

---

## Reading the output

| Output | Meaning |
|---|---|
| **Fair value** | Price implied by the market state, anchored `H` sessions back |
| **Fair value gap** | `Price − FV`, and the same as a percentage |
| **FVO** | Bounded ±100. `0` = fairly valued, `> 0` rich, `< 0` cheap |
| **Dynamic thresholds** | Trailing empirical quantiles, lagged one session |
| **Confidence** | Blend of out-of-sample R², coefficient and loading stability, regime settledness, data coverage |
| **Residual risk** | Idiosyncratic volatility the cross-section cannot explain |
| **Mean-reversion P** | `P(|z|` inside ±0.5σ within `h)` under a fitted OU process — analytic, from the normal CDF, not a tuned logistic |
| **Regime** | One of RISK-ON, RISK-OFF, TREND, MEAN-REV, HIGH-VOL |

The **Signal Evidence** tab leads with parameter-free diagnostics — the
information coefficient and the forward-return-by-bucket profile. Neither has a
threshold or a trading rule to overfit. If those are flat, no strategy built on
the oscillator will work, however good the equity curve underneath happens to
look.

---

## Notes, honestly

- **PCA and ICA give near-identical fair values.** This is correct, not a bug:
  L2 ridge is rotation-invariant and ICA spans the same subspace as PCA. ICA is
  useful for *interpreting* factors, not for changing the fit. The autoencoder
  is the option that genuinely changes the function class.
- **Demo mode should show roughly zero signal.** The simulator is a pure factor
  model with i.i.d. idiosyncratic noise — it contains no mean-reverting
  mispricing by construction. The engine reports near-zero IC and a losing
  backtest on it. That is the engine being honest, and it doubles as a control:
  if it ever finds strong signal in demo mode, something is leaking.
- **Overlapping observations inflate significance.** Forward returns at a daily
  frequency overlap heavily, so IC p-values are optimistic. Treat |IC| < 0.03
  as noise regardless of what the p-value says.
- **The backtest is one path through a large parameter space.** The sidebar
  exposes enough degrees of freedom to fit almost any curve.
- The engine models *relative* value against a market cross-section. It has no
  view on cash flows, earnings, or whether the entire cross-section is
  mispriced together.

Research tool. Not investment advice.
