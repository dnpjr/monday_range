from __future__ import annotations
import pandas as pd
import numpy as np

def equity_metrics(equity: pd.Series, bars_per_year: float) -> dict:
    eq = equity.dropna()
    if len(eq) < 3:
        return {}

    rets = eq.pct_change().fillna(0.0)

    vol = rets.std()
    sharpe = (rets.mean() / vol) * np.sqrt(bars_per_year) if vol > 0 else 0.0

    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min())

    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
    }

def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"num_trades": 0}

    pnl = trades["pnl"]
    return {
        "num_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "profit_factor": float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl[pnl < 0].sum() != 0) else float("inf"),
        "by_reason": trades["reason"].value_counts().to_dict(),
    }
