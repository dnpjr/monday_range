# Monday Range (Sweep-and-Fade) — Research + Backtest

This repository contains a small **research + backtesting** pipeline for a "Monday range" idea:

1. **Research (`research_monday_range.py`)**: quantify how often a weekly signal occurs and what happens after entry (hit mid-range, hit opposite side of range, drawdowns in units of Monday range, etc.).
2. **Backtest (`backtest_monday_range.py`)**: implement a simple strategy informed by the research (risk sizing, stop placement, targets, and basic risk metrics).

The goal is **clarity + reproducibility**: single-command scripts, clean outputs, and a small set of standard performance metrics.

---

## Strategy idea (high-level)

- Define **Monday range** = Monday high − Monday low; **mid** = (high + low)/2.
- From Tuesday onward, look for a **sweep**:
  - **Long setup:** a bar trades below Monday low and closes back inside (close > Monday low) → enter long next bar open.
  - **Short setup:** a bar trades above Monday high and closes back inside (close < Monday high) → enter short next bar open.
- Exits: hard stop, partial take-profit at mid, final take-profit near the opposite side of the Monday range, and a Friday close-out (optional).

---

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Run research
```bash
python research_monday_range.py --symbol BTC-USD --interval 4h --period 2y
```

### 3) Run backtest
```bash
python backtest_monday_range.py --symbol BTC-USD --interval 4h --period 2y
```

Outputs are saved under `results/`.

---

## Notes
- Data is fetched via `yfinance` for convenience.
- This is a toy research project for learning backtesting hygiene (data → signal → evaluation). It is **not** investment advice.
