# 🌌 SANKET | Market Signal Screener
### The Institutional Pulse Engine · Obsidian Quant Terminal

**Sanket** is a high-fidelity quantitative terminal designed for institutional-grade market analysis. As a core member of the **Pragyam Product Family**, it utilizes the **Wave-Regime Composite Index (WRCI)** to identify abnormal market acceleration ("Pulse") and fractal resonance across global asset classes.

The system provides 1:1 mathematical parity between the Python-based terminal and the TradingView Pine Script indicator, ensuring that tactical signals generated in the screener match the visual execution on the charts.

---

## ⚡ Key Intelligence Features

- **WRCI Pulse Engine**: Measures abnormal acceleration by modulating 5-day velocity with 20-day statistical intensity (Z-Score). It filters out "market noise" to identify true regime shifts.
- **Fractal Resonance Logic**: A multi-dimensional signal engine that synchronizes tactical momentum with structural trend alignment.
- **Obsidian Quant UI**: A premium, high-contrast dark-mode terminal built on Streamlit, designed for maximum information density with zero visual friction.
- **Universal Asset Scanning**: Native support for NSE (F&O, Indices), US Equities, Crypto, Commodities, Currencies, and Global Macro Bond ETFs.
- **Analog Engine**: A historical pattern-matching system using cosine similarity to identify high-probability price trajectories.

---

## 🛠 Tech Stack

- **Backend**: Python 3.10+
- **Frontend**: Streamlit (with Custom Obsidian CSS)
- **Signal Logic**: Pine Script v6 (Mathematical Mirror)
- **Data Pipeline**: YFinance API / NSE India
- **Visuals**: Plotly Quant Themes

---

## 🚦 Signal Hierarchy (Sets A–D)

Sanket classifies every market signal into a strict hierarchical system to ensure non-redundant trading execution:

| Signal Set | Type | Logic | Use Case |
| :--- | :--- | :--- | :--- |
| **Set A** | **Momentum** | WRCI Line/Signal Crossover | Tactical trend following in liquid regimes. |
| **Set B** | **Contrarian** | WRCI Cross in OS/OB Zones | High-probability reversal identification. |
| **Set C** | **Threshold** | Zone Entry/Exit Gates | Identifying volatility regime transitions. |
| **Set D** | **Squeeze** | BB/KC Volatility Breakout | Capitalizing on imminent explosive expansion. |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher installed.
- Git.

### 2. Installation
```bash
git clone https://github.com/manthan/Sanket.git
cd Sanket
pip install -r requirements.txt
```

### 3. Launch the Terminal
```bash
streamlit run sanket.py
```

---

## 📂 Project Structure

```
├── sanket.py           # Main Intelligence Engine & Streamlit UI
├── priority_engine.py  # Directional Priority Ranking Logic
├── wrci.pine           # TradingView Indicator (Mathematical 1:1)
├── logger.py           # Professional Console Logging System
├── ui/
│   ├── theme.py        # Obsidian Quant Design System
│   └── components.py   # Reusable UI Elements (SVG/Tables)
└── requirements.txt    # Project Dependencies
```

---

## ⚖️ License

Institutional usage only. See the `LICENSE` file for full terms.
