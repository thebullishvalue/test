# 📜 CHANGELOG
### Sanket | The Pulse Engine Evolution

All notable changes to the **Sanket** platform are documented here.

---

## [v2.2.0] · 2026-05-05
### Institutional Pulse Engine & Sync

**"The Obsidian Quant Transformation"**

This version marks the final transition of Sanket into an institutional-grade terminal, achieving perfect synchronization between the Python backend and the TradingView indicator.

#### 🚀 Features & Enhancements
- **The Pulse Engine**: Implemented "Abnormal Acceleration" logic—calculating 5D Velocity modulated by 20D Volatility Z-Score.
- **Mathematical Parity**: Unified HMA, WMA, and Linear Regression endpoint calculations across both `sanket.py` and `wrci.pine` to achieve 1:1 signal precision.
- **Obsidian Quant UI**: Fully applied the premium "Obsidian" design language across all terminal components, featuring sleek SVG icons and high-contrast data tables.
- **Fractal Multi-Timeframe (MTF)**: Integrated Daily and Weekly macro-anchoring for tactical signals, reducing "noise" in volatile regimes.
- **Universal Asset Mapping**: Expanded support for Global Macro Bond ETFs and advanced Commodity futures tracking.

#### 🔧 Internal Optimizations
- **VWAP Accuracy**: Refactored VWAP logic to use true ratio-of-sums instead of discrete averaging.
- **Signal Gating**: Hardened signal logic (Sets A–D) to ensure mutual exclusivity and non-redundant firing.
- **Anti-Clustering**: Enhanced the Analog Engine with anti-clustering logic to prevent redundant matches in pattern analysis.

---

## [v2.1.0] · 2026-04-30
### WRCI Foundation

- **Initial WRCI Release**: Introduced the Wave-Regime Composite Index as the core analytical engine.
- **Squeeze Engine**: Integrated Bollinger/Keltner volatility compression analysis.
- **Analog Matching**: First iteration of the Cosine Similarity-based pattern matcher.

---

## [v2.0.0] · 2026-04-15
### Pragyam Family Rebirth

- **Architectural Shift**: Transitioned from a standalone script to the modular Pragyam architecture.
- **Streamlit Integration**: Launched the web-based terminal interface.
- **Multi-Universe Support**: Introduced constituent scraping for NSE, NASDAQ, and S&P 500.

---

*For detailed technical specifications, refer to the [README.md](README.md).*
