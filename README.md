# Monday Range (Sweep-and-Fade) — Research + Backtest Framework

A structured mini-project for researching and backtesting a widely discussed weekly liquidity pattern commonly referred to as the **Monday Range sweep-and-fade** idea.

The repo is organised into two layers:

1. **Research module**: empirically studies what tends to happen *after* Monday range sweeps (hit rates, conditional outcomes, distributions).
2. **Backtest module**: implements a **risk-managed**, parameterised strategy informed by the research.

The emphasis is on:
- clear signal definitions (avoid “hand-wavy” rules),
- reproducible experiments (same inputs → same outputs),
- risk-first position sizing,
- transparent reporting (trade logs + metrics + plots).

> **Note:** This project is for educational/research purposes only and is not trading advice.

---

## Concept

### Define the Monday Range

For each week:

- **Monday High**: \(H_M\)  
- **Monday Low**: \(L_M\)  
- **Range size**: \(R = H_M - L_M\)  
- **Midpoint**: \(M = (H_M + L_M) / 2\)

This creates a weekly reference “box” used as a liquidity boundary.

### Sweep-and-Fade Signal

From **Tuesday onward**, look for a *sweep* beyond the Monday range followed by rejection back inside:

**Long setup (sweep below)**
- A candle trades **below** \(L_M\)
- Then closes **back above** \(L_M\)
- → enter **long** on the next bar open (or on the close, depending on the chosen convention)

**Short setup (sweep above)**
- A candle trades **above** \(H_M\)
- Then closes **back below** \(H_M\)
- → enter **short** on the next bar open

Interpretation: price runs external liquidity (stops), then mean-reverts back into the range.

---

## Repository Structure

Typical structure:

```
.
├── research_monday_range.py        # empirical analysis of sweep outcomes
├── backtest_monday_range.py        # strategy backtest informed by research
├── src/                            # reusable utilities (data, signals, sizing, metrics)
├── results/
│   ├── research/                   # research CSV/JSON outputs
│   ├── backtests/                  # trade logs + metrics JSON
│   └── plots/                      # equity curves + diagnostic plots
├── requirements.txt
└── README.md
```

(Exact filenames may differ slightly depending on the version it is using.)

---

## Research Module

### What it measures

`research_monday_range.py` is meant to answer questions like:

- How often does a **Monday sweep** occur?
- Conditional probabilities of reaching:
  - **midpoint** \(M\),
  - the **opposite boundary** (full reversal),
  - or failing and continuing the breakout.
- Distributions of:
  - **maximum adverse excursion (MAE)**,
  - **maximum favourable excursion (MFE)**,
  - time-to-target (how many bars until TP is hit).
- Outcomes measured in **units of Monday range** (e.g. 0.5R, 1.0R).

The key goal is to quantify:
> “If a sweep happens, what is the empirical chance of mean reversion to M / opposite side?”

### Multi-symbol support

The research module supports multiple instruments, e.g.:

- `BTC-USD`
- `^GSPC` (S&P 500)
- `^DJI` (Dow Jones)
- `^IXIC` (Nasdaq)
- `EURUSD=X` (FX via Yahoo tickers)
- any other Yahoo Finance symbol with sufficient history

**Tip:** when adding more indices, keep the intervals consistent (e.g. `1h` or `4h`) to compare like-for-like.

### Example usage

```bash
python research_monday_range.py --symbol BTC-USD --interval 4h --period 2y
python research_monday_range.py --symbol ^GSPC   --interval 1h --period 5y
```

Outputs are saved under `results/` (CSV + summary JSON).

---

## Backtest Module

`backtest_monday_range.py` implements a basic event-driven backtest loop with:

- signal generation (sweep-and-fade),
- entries/exits,
- position sizing via **risk per trade**,
- trade logging and performance reporting.

### Initial capital

The backtest starts with a configurable initial equity, e.g.:

- `initial_capital = 100_000`

### Risk per trade (position sizing)

Rather than “buy 1 unit”, the strategy sizes positions based on a fixed fraction of equity:

- `risk_fraction = 0.01`  (1% of equity per trade)

Then:

- **risk amount** = `equity * risk_fraction`
- **stop distance** = (entry price − stop loss) in price units
- **position size** = `risk_amount / stop_distance`

This is the standard **risk parity per trade** idea used in systematic strategies.

### Stops and targets

Stops/targets are typically expressed relative to the Monday range:

- Stop loss beyond sweep extreme (optionally with a multiplier)
- Take-profit levels tied to:
  - TP1: midpoint \(M\)
  - TP2: opposite boundary (full reversal)

### Layering (partial exits)

The strategy supports **layering**, i.e. partial exits:

- At **TP1**, close a fraction of the position:
  - `tp1_fraction = 0.5` means take 50% off at midpoint
- The remainder runs toward **TP2**

Why this is useful:
- reduces variance,
- locks in partial profits,
- keeps upside if the full reversal happens.

### Trade management knobs

Key configurable parameters include:

- `symbol`, `interval`, `period`
- `initial_capital`
- `risk_fraction`
- `tp1_fraction`
- stop placement rules (range-multiple, fixed, ATR-based)
- end-of-week exit (e.g. close on Friday)

### Outputs

Backtests produce:
- `equity_curve.png`
- `trades.csv` (trade-by-trade log)
- `metrics.json` (summary stats)

Typical summary metrics:
- total return
- max drawdown
- win rate
- average win / average loss
- profit factor
- Sharpe (if using bar returns)
- exposure and trade count

---

## What this project demonstrates

This repo is intentionally *simple* but “quant-structured”:

- separation of **research** vs **execution/backtest** code,
- risk-based sizing rather than fixed position units,
- layered exits (a realistic trade-management technique),
- explicit parameterisation,
- repeatable output artefacts.

---

## Important caveat: strategy saturation and regime dependence

The Monday range idea is widely known. In modern markets:

- simple rules often degrade as they become crowded,
- edge is frequently **regime-dependent** (trend vs mean-reversion),
- transaction costs/slippage can eliminate apparent backtest edge.

So a “vanilla” sweep-and-fade strategy is unlikely to be robust in current conditions without additional structure.

---

## Limitations

This project is not a production trading engine:

- no realistic slippage model by default,
- limited transaction cost modelling,
- depends on Yahoo Finance data quality,
- no corporate actions nuance for some instruments,
- assumes clean fills at bar boundaries.

This is expected for a portfolio/research demo.

---

## Extensions / Improvements

To evolve this into something genuinely research-grade:

### 1) Regime filters (time-series analysis)
Only trade when conditions are favourable, e.g.:

- moving average slope filter (trend/mean-reversion regime),
- volatility filter (e.g. ATR percentile),
- VIX filter (for indices),
- range size filter (trade only when Monday range is “normal”).

### 2) Better exits and dynamic targets
- ATR-based stops/targets
- trailing stops
- time stops (exit after N bars if no follow-through)
- multiple staggered take-profits beyond just TP1/TP2

### 3) Parameter robustness checks
- walk-forward testing
- sensitivity analysis (heatmaps over TP1 fraction, risk fraction, stop multiplier)
- sub-period performance (year-by-year)

### 4) Cost-based optimisation (more quant-relevant)
Instead of maximising F1/win-rate, optimise **expected value**:

- include per-trade fee/slippage assumptions,
- maximise expectancy or risk-adjusted return under constraints.

### 5) ML-assisted filtering (optional, and harder)
Use ML *not* to “predict price”, but to decide **when the pattern is worth trading**.

Example:
- Build a dataset where each sweep event is a row.
- Features: Monday range size, volatility, trend indicators, day/time, macro proxies.
- Label: whether midpoint/opposite boundary was reached.
- Train a classifier (logistic regression / gradient boosting) to output a probability of success.
- Trade only if probability exceeds a threshold (and size positions proportionally).

This often works better than trying to predict raw returns.

---

## Takeaways

- Splitting research and backtesting code makes the process clearer and more defensible.
- In imbalanced/conditional setups, it’s critical to quantify hit rates and failure modes.
- A well-known discretionary pattern is rarely robust “as-is”; regime filters and execution realism matter.
- The right next step is **time-series aware validation** (walk-forward) and **robustness checks**, not just adding complexity.

---

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run research:

```bash
python research_monday_range.py --symbol BTC-USD --interval 4h --period 2y
```

Run backtest:

```bash
python backtest_monday_range.py --symbol BTC-USD --interval 4h --period 2y
```

---

## Disclaimer

This repository is provided for educational purposes only. Past performance does not guarantee future results.
