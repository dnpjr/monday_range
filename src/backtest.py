from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry: float
    exit: float
    qty: float
    pnl: float
    reason: str
    week_id: tuple[int, int]

def backtest_sweep_fade(
    df: pd.DataFrame,
    *,
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 100.0,
    stop_mult: float = 1.0,      # stop distance = stop_mult * mon_range beyond monday boundary
    tp1_at_mid: bool = True,
    tp1_frac: float = 0.5,       # fraction to take at mid
    tp2_to_full: float = 1.0,    # 1.0 means opposite side (Mon high for long, Mon low for short)
    exit_friday_close: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event-driven backtest: one trade per week, first signal only.

    Position sizing: fixed $ risk per trade using distance to stop.

    Returns:
      - df_out: original df plus equity/position columns
      - trades: trade log
    """
    df = df.copy()
    df["equity"] = np.nan
    df["position"] = 0.0
    df["trade_pnl"] = 0.0

    equity = float(initial_capital)
    df.iloc[0, df.columns.get_loc("equity")] = equity

    trades: List[Trade] = []

    in_pos = False
    side = None
    entry = 0.0
    qty = 0.0
    stop = 0.0
    tp1 = 0.0
    tp2 = 0.0
    tp1_taken = False
    week_id = None
    has_traded_week = False

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        row = df.iloc[i]
        ts = row.name

        curr_week = (int(row["iso_year"]), int(row["iso_week"]))
        if curr_week != week_id:
            week_id = curr_week
            has_traded_week = False

        mon_high = row["mon_high"]
        mon_low = row["mon_low"]
        mon_mid = row["mon_mid"]
        rng = row["mon_range"]

        if pd.isna(mon_high) or pd.isna(mon_low) or rng <= 0:
            df.at[ts, "equity"] = equity
            continue

        # Friday close-out heuristic: last bar of Friday in the dataset (or before weekend)
        is_friday = (row["weekday"] == 4)
        next_weekday = df.iloc[i+1]["weekday"] if i < len(df) - 1 else None
        is_last_friday_bar = is_friday and (next_weekday != 4)

        # --- Entry: first valid signal, Tue-Thu only ---
        if (not in_pos) and row["is_tradeable"] and (not has_traded_week) and (row["weekday"] < 4):
            long_signal = (prev["low"] < mon_low) and (prev["close"] > mon_low)
            short_signal = (prev["high"] > mon_high) and (prev["close"] < mon_high)

            if long_signal or short_signal:
                side = "LONG" if long_signal else "SHORT"
                entry = float(row["open"])

                if side == "LONG":
                    stop = float(mon_low - stop_mult * rng)
                    tp1 = float(mon_mid)
                    tp2 = float(mon_high if tp2_to_full >= 1.0 else mon_mid + tp2_to_full * (mon_high - mon_mid))
                    dist = abs(entry - stop)
                else:
                    stop = float(mon_high + stop_mult * rng)
                    tp1 = float(mon_mid)
                    tp2 = float(mon_low if tp2_to_full >= 1.0 else mon_mid - tp2_to_full * (mon_mid - mon_low))
                    dist = abs(stop - entry)

                qty = (risk_per_trade / dist) if dist > 0 else 0.0
                if qty > 0:
                    in_pos = True
                    tp1_taken = False
                    has_traded_week = True

        # --- Manage exits ---
        if in_pos and qty > 0:
            # Stop check (intrabar)
            hit_stop = (row["low"] <= stop) if side == "LONG" else (row["high"] >= stop)
            if hit_stop:
                exit_px = stop
                pnl = (exit_px - entry) * qty if side == "LONG" else (entry - exit_px) * qty
                equity += pnl
                trades.append(Trade(entry_time=df.index[i], exit_time=ts, side=side, entry=entry, exit=exit_px, qty=qty, pnl=pnl, reason="STOP", week_id=week_id))
                df.at[ts, "trade_pnl"] += pnl
                in_pos = False
                qty = 0.0

            # Partial TP1 at mid
            if in_pos and tp1_at_mid and (not tp1_taken):
                hit_tp1 = (row["high"] >= tp1) if side == "LONG" else (row["low"] <= tp1)
                if hit_tp1:
                    exit_qty = qty * tp1_frac
                    exit_px = tp1
                    pnl = (exit_px - entry) * exit_qty if side == "LONG" else (entry - exit_px) * exit_qty
                    equity += pnl
                    df.at[ts, "trade_pnl"] += pnl
                    qty = qty - exit_qty
                    tp1_taken = True

                    # optional: move stop to breakeven after TP1
                    stop = entry

            # Final TP2
            if in_pos and qty > 0:
                hit_tp2 = (row["high"] >= tp2) if side == "LONG" else (row["low"] <= tp2)
                if hit_tp2:
                    exit_px = tp2
                    pnl = (exit_px - entry) * qty if side == "LONG" else (entry - exit_px) * qty
                    equity += pnl
                    trades.append(Trade(entry_time=df.index[i], exit_time=ts, side=side, entry=entry, exit=exit_px, qty=qty, pnl=pnl, reason="TP2", week_id=week_id))
                    df.at[ts, "trade_pnl"] += pnl
                    in_pos = False
                    qty = 0.0

            # Friday close-out
            if in_pos and qty > 0 and exit_friday_close and is_last_friday_bar:
                exit_px = float(row["close"])
                pnl = (exit_px - entry) * qty if side == "LONG" else (entry - exit_px) * qty
                equity += pnl
                trades.append(Trade(entry_time=df.index[i], exit_time=ts, side=side, entry=entry, exit=exit_px, qty=qty, pnl=pnl, reason="FRIDAY", week_id=week_id))
                df.at[ts, "trade_pnl"] += pnl
                in_pos = False
                qty = 0.0

        df.at[ts, "position"] = (qty if in_pos else 0.0) * (1 if side == "LONG" else (-1 if side == "SHORT" else 0))
        df.at[ts, "equity"] = equity

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return df, trades_df
