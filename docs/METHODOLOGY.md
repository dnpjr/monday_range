# Backtest methodology

This document describes the canonical accounting and execution model used by `src/backtest.py` and `run_backtest.py` from Phase 1 onward.

## Strategy timing

The Monday range strategy is unchanged. A completed bar is evaluated for a sweep and rejection. When a signal exists, the strategy enters at the next bar open after applying adverse entry slippage. The full Monday range is only used after Monday is complete. The current strategy takes at most one entry per ISO week and permits entries from Tuesday through Thursday.

## Portfolio accounting

The engine uses an account ledger suitable for symmetric long and short research:

- `cash` is initial capital plus realised gross P&L minus all charged fees. Entry notional is tracked separately rather than deducted as a spot purchase.
- `open_qty` is the remaining absolute quantity. `position` is signed quantity.
- `position_value` is signed quantity multiplied by the current mark. `gross_notional` is its absolute value.
- `unrealized_pnl` is the remaining position's P&L from its slipped entry price to the current mark.
- `realized_pnl` is `cash - initial_capital`, so it includes realised trading P&L and fees already charged.
- `equity` is `cash + unrealized_pnl`.

Open positions are marked at every OHLC bar close. Total return, CAGR, drawdown, volatility, Sharpe, and Sortino use this marked equity series. Volatility, Sharpe, and Sortino use the interval-derived bars-per-year value. CAGR uses actual elapsed timestamps and a 365-day crypto year.

By default, entry risk capital is current marked equity multiplied by `risk_fraction`. Set `risk_base=initial_capital` to use a fixed initial-capital fraction. Direct calls can leave `risk_fraction=None` and supply `risk_per_trade` for an explicit legacy fixed-dollar mode. Entry quantity is the smaller of stop-risk quantity and `current_equity * max_leverage / entry_price`. The corrected research default is `max_leverage=1.0`.

## Execution conventions

Stops and targets are evaluated only after entry at the next bar open. Fees are charged on every entry and exit fill, including partial, gap, Friday, and end-of-data exits. Slippage is adverse for the trade direction.

When a bar opens through a stop, the raw fill is the bar open because the stop level is no longer available. When a bar opens through a target, the raw fill is the bar open, allowing the observable opening improvement. Slippage is then applied to that raw fill.

OHLC bars do not reveal whether an intrabar stop or target occurred first. `intrabar_policy=conservative_stop_first` is the canonical research default. `target_first` exists for explicitly labelled sensitivity analysis. If a partial target moves the stop, the same bar's already-observed high/low is not reused against the new stop after an opening-gap target fill.

A remaining position on the final bar is closed at that bar's close using normal exit slippage and fees. Its exit reason is `END_OF_DATA`.

## Stop names

- `swept_boundary`: the stop is exactly the swept Monday boundary: Monday low for a long and Monday high for a short.
- `swept_boundary_offset`: the stop is `stop_range_fraction` Monday-range widths beyond the swept boundary. This preserves the canonical Monday repository's historical behavior.
- `entry_fixed_pct`: the stop is `stop_pct` away from the slipped entry price.

Saved names `opposite_boundary`, `range_fraction`, and `fixed_pct` remain readable and emit deprecation warnings. Both of the first two map to `swept_boundary_offset`; direct legacy `opposite_boundary` calls use `stop_mult`, while saved runner configs preserve the runner's historical one-range-width behavior.

## Legacy results

Anything already under `results/`, `data/backtests/`, `data/sweeps/`, or other previously generated output directories was produced before marked equity, current-equity sizing, leverage caps, gap-aware fills, explicit intrabar policy, and terminal closure. Those artifacts remain historical records and must not be interpreted as results from the corrected methodology. Phase 1 does not overwrite or regenerate them.
