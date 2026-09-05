from __future__ import annotations

import numpy as np
import pandas as pd


def equity_metrics(equity: pd.Series, bars_per_year: float) -> dict:
    """Compute portfolio metrics from marked bar-close equity.

    Return volatility, Sharpe and Sortino use ``bars_per_year``. CAGR uses actual
    elapsed calendar time for a DatetimeIndex and a 365-day crypto year; it falls
    back to the observed bar count when timestamps are unavailable.
    """
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return {}

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] != 0 else np.nan
    peak = eq.cummax()
    drawdown = (eq - peak) / peak.replace(0.0, np.nan)
    max_drawdown = float(drawdown.min()) if drawdown.notna().any() else 0.0

    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    period_vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annualized_volatility = period_vol * float(np.sqrt(bars_per_year)) if bars_per_year > 0 else 0.0
    mean_return = float(returns.mean()) if len(returns) else 0.0
    sharpe = (mean_return / period_vol) * float(np.sqrt(bars_per_year)) if period_vol > 0 and bars_per_year > 0 else 0.0

    downside = returns[returns < 0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    sortino = (
        (mean_return / downside_deviation) * float(np.sqrt(bars_per_year))
        if downside_deviation > 0 and bars_per_year > 0
        else 0.0
    )

    elapsed_years = 0.0
    if isinstance(eq.index, pd.DatetimeIndex) and len(eq) > 1:
        elapsed_seconds = float((eq.index[-1] - eq.index[0]).total_seconds())
        elapsed_years = elapsed_seconds / (365.0 * 24.0 * 3600.0)
    elif bars_per_year > 0 and len(eq) > 1:
        elapsed_years = (len(eq) - 1) / float(bars_per_year)

    if elapsed_years > 0 and eq.iloc[0] > 0 and eq.iloc[-1] > 0:
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / elapsed_years) - 1.0)
    else:
        cagr = 0.0

    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return": total_return,
        "annualized_return": cagr,
        "cagr": cagr,
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "elapsed_years": float(elapsed_years),
        "annualization_periods_per_year": float(bars_per_year),
    }


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "num_trades": 0,
            "win_rate": None,
            "avg_pnl": None,
            "avg_trade_return": None,
            "median_pnl": None,
            "profit_factor": None,
            "total_net_pnl": 0.0,
            "total_fees": 0.0,
            "by_reason": {},
        }

    pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None

    return {
        "num_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
        "avg_pnl": float(pnl.mean()) if len(pnl) else None,
        "avg_trade_return": (
            float(pd.to_numeric(trades["return_pct"], errors="coerce").mean())
            if "return_pct" in trades.columns
            else None
        ),
        "median_pnl": float(pnl.median()) if len(pnl) else None,
        "profit_factor": (None if profit_factor is None else float(profit_factor)),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "total_net_pnl": float(pnl.sum()),
        "total_fees": (
            float(pd.to_numeric(trades["fees_paid"], errors="coerce").fillna(0.0).sum())
            if "fees_paid" in trades.columns
            else None
        ),
        "by_reason": trades["reason"].value_counts().to_dict(),
    }
