from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
from .signals import sweep_rejection_signal
from .execution_helpers import (
    apply_slippage,
    calculate_entry_fee,
    calculate_exit_fee,
    calculate_tp_levels,
    is_stop_hit,
    is_tp_hit,
    is_friday_cutoff_bar,
)

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry: float
    exit: float
    qty: float
    pnl: float
    return_pct: float
    reason: str
    week_id: tuple[int, int]
    net_pnl: float | None = None
    gross_pnl: float | None = None
    fees_paid: float | None = None
    holding_hours: float | None = None
    entry_weekday_utc: int | None = None
    entry_hour_utc: int | None = None
    mon_range: float | None = None
    mon_mid: float | None = None
    mon_range_pct: float | None = None

def backtest_sweep_fade(
    df: pd.DataFrame,
    *,
    strategy: str = "current_monday_range",
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 100.0,
    stop_mult: float = 1.0,      # stop distance = stop_mult * mon_range beyond monday boundary
    tp1_at_mid: bool = True,
    tp1_frac: float = 0.5,       # fraction to take at mid
    tp2_to_full: float = 1.0,    # 1.0 means opposite side (Mon high for long, Mon low for short)
    exit_friday_close: bool = True,
    friday_cutoff_hour_utc: int = 23,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    min_range_pct: float | None = None,
    max_range_pct: float | None = None,
    direction: str = "both",
    max_entry_day_utc: int | None = None,
    max_entry_hour_utc: int | None = None,
    sma_period: int | None = None,
    tp1_range_fraction: float | None = None,
    tp2_range_fraction: float | None = None,
    tp1_to_mid: float = 1.0,
    tp1_close_fraction: float | None = None,
    stop_mode: str = "opposite_boundary",
    stop_range_fraction: float = 1.0,
    stop_pct: float = 0.01,
    move_stop_to_breakeven_after_tp1: bool = True,
    breakeven_includes_fees: bool = False,
    single_target_mode: bool = False,
    single_target_level: str = "tp2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event-driven backtest: one trade per week, first signal only.

    Position sizing: fixed $ risk per trade using distance to stop.

    Returns:
      - df_out: original df plus equity/position columns
      - trades: trade log
    """
    valid_directions = {"both", "long_only", "short_only"}
    valid_strategies = {"current_monday_range", "sweep_retest"}
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {sorted(valid_strategies)}")
    if direction not in valid_directions:
        raise ValueError(f"direction must be one of {sorted(valid_directions)}")
    if max_entry_day_utc is not None and not (0 <= int(max_entry_day_utc) <= 6):
        raise ValueError("max_entry_day_utc must be between 0 (Monday) and 6 (Sunday).")
    if max_entry_hour_utc is not None and not (0 <= int(max_entry_hour_utc) <= 23):
        raise ValueError("max_entry_hour_utc must be between 0 and 23.")
    if sma_period is not None and int(sma_period) < 1:
        raise ValueError("sma_period must be >= 1 when provided.")
    if min_range_pct is not None and max_range_pct is not None and float(min_range_pct) > float(max_range_pct):
        raise ValueError("min_range_pct cannot be greater than max_range_pct.")
    valid_stop_modes = {"opposite_boundary", "range_fraction", "fixed_pct"}
    if stop_mode not in valid_stop_modes:
        raise ValueError(f"stop_mode must be one of {sorted(valid_stop_modes)}")
    if tp1_to_mid < 0:
        raise ValueError("tp1_to_mid must be >= 0.")
    if tp1_range_fraction is not None and tp1_range_fraction < 0:
        raise ValueError("tp1_range_fraction must be >= 0.")
    if tp2_range_fraction is not None and tp2_range_fraction < 0:
        raise ValueError("tp2_range_fraction must be >= 0.")
    if tp1_close_fraction is None:
        tp1_close_fraction = float(tp1_frac)
    if not (0.0 <= float(tp1_close_fraction) <= 1.0):
        raise ValueError("tp1_close_fraction must be between 0 and 1.")
    if stop_range_fraction <= 0:
        raise ValueError("stop_range_fraction must be > 0.")
    if stop_pct <= 0:
        raise ValueError("stop_pct must be > 0.")
    valid_single_target_levels = {"tp1", "tp2"}
    if single_target_level not in valid_single_target_levels:
        raise ValueError(f"single_target_level must be one of {sorted(valid_single_target_levels)}")

    df = df.copy()
    df["equity"] = np.nan
    df["position"] = 0.0
    df["trade_pnl"] = 0.0
    df["fees_paid"] = 0.0
    if sma_period is not None:
        df["_sma"] = df["close"].rolling(window=int(sma_period), min_periods=int(sma_period)).mean()

    equity = float(initial_capital)
    df.iloc[0, df.columns.get_loc("equity")] = equity

    trades: List[Trade] = []

    in_pos = False
    side = None
    entry_time = None
    entry = 0.0
    entry_notional = 0.0
    qty = 0.0
    stop = 0.0
    tp1 = 0.0
    tp2 = 0.0
    tp1_taken = False
    trade_pnl_accum = 0.0
    gross_pnl_accum = 0.0
    fees_paid_accum = 0.0
    entry_weekday_utc = None
    entry_hour_utc = None
    entry_mon_range = None
    entry_mon_mid = None
    entry_mon_range_pct = None
    week_id = None
    has_traded_week = False
    first_signal_seen_week = False
    signal_count = 0

    freq_delta = (df.index[1] - df.index[0]) if len(df.index) > 1 else pd.Timedelta(hours=1)

    def net_pnl_for_exit(exit_px_raw: float, exit_qty: float, side_: str) -> tuple[float, float, float, float]:
        exit_px = apply_slippage(exit_px_raw, side_, action="exit", slippage_bps=slippage_bps)
        gross = (exit_px - entry) * exit_qty if side_ == "LONG" else (entry - exit_px) * exit_qty
        fee = calculate_exit_fee(exit_px, exit_qty, fee_bps=fee_bps)
        return gross - fee, exit_px, fee, gross

    def entry_cutoff_allows(ts_utc: pd.Timestamp) -> bool:
        if max_entry_day_utc is None and max_entry_hour_utc is None:
            return True
        weekday = int(ts_utc.weekday())
        hour = int(ts_utc.hour)
        if max_entry_day_utc is not None:
            if weekday > int(max_entry_day_utc):
                return False
            if weekday < int(max_entry_day_utc):
                return True
        if max_entry_hour_utc is not None:
            return hour <= int(max_entry_hour_utc)
        return True

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        row = df.iloc[i]
        ts = row.name

        curr_week = (int(row["iso_year"]), int(row["iso_week"]))
        if curr_week != week_id:
            week_id = curr_week
            has_traded_week = False
            first_signal_seen_week = False

        mon_high = row["mon_high"]
        mon_low = row["mon_low"]
        mon_mid = row["mon_mid"]
        rng = row["mon_range"]

        if pd.isna(mon_high) or pd.isna(mon_low) or rng <= 0:
            df.at[ts, "equity"] = equity
            continue

        ts_utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        bar_open = ts_utc
        bar_close = ts_utc + freq_delta
        friday_cutoff_hit = is_friday_cutoff_bar(bar_open, bar_close, friday_cutoff_hour_utc=friday_cutoff_hour_utc)
        range_pct = (float(rng) / abs(float(mon_mid))) if (pd.notna(mon_mid) and float(mon_mid) != 0.0) else np.nan

        # --- Entry ---
        if (not in_pos) and row["is_tradeable"] and (not has_traded_week):
            signal_side = None
            if strategy == "current_monday_range":
                # Existing strategy behavior: first valid entry Tuesday-Thursday.
                if row["weekday"] < 4:
                    if min_range_pct is not None and (pd.isna(range_pct) or float(range_pct) < float(min_range_pct)):
                        signal_side = None
                    elif max_range_pct is not None and (pd.isna(range_pct) or float(range_pct) > float(max_range_pct)):
                        signal_side = None
                    elif not entry_cutoff_allows(ts_utc):
                        signal_side = None
                    else:
                        signal_side = sweep_rejection_signal(prev, mon_low=mon_low, mon_high=mon_high)
            else:
                # Legacy sweep/retest behavior: first actual sweep signal of the week only.
                if not first_signal_seen_week:
                    raw_signal_side = sweep_rejection_signal(prev, mon_low=mon_low, mon_high=mon_high)
                    if raw_signal_side is not None:
                        first_signal_seen_week = True
                        signal_count += 1
                        if min_range_pct is not None and (pd.isna(range_pct) or float(range_pct) < float(min_range_pct)):
                            signal_side = None
                        elif max_range_pct is not None and (pd.isna(range_pct) or float(range_pct) > float(max_range_pct)):
                            signal_side = None
                        elif not entry_cutoff_allows(ts_utc):
                            signal_side = None
                        else:
                            signal_side = raw_signal_side

            if signal_side is not None and direction == "long_only" and signal_side != "LONG":
                signal_side = None
            if signal_side is not None and direction == "short_only" and signal_side != "SHORT":
                signal_side = None

            if signal_side is not None and sma_period is not None:
                prev_sma = prev.get("_sma", np.nan)
                if pd.isna(prev_sma):
                    signal_side = None
                elif signal_side == "LONG" and not (float(prev["close"]) > float(prev_sma)):
                    signal_side = None
                elif signal_side == "SHORT" and not (float(prev["close"]) < float(prev_sma)):
                    signal_side = None

            if signal_side is not None:
                if strategy == "current_monday_range":
                    signal_count += 1
                side = signal_side
                entry_time = ts
                entry = apply_slippage(float(row["open"]), side, action="entry", slippage_bps=slippage_bps)

                if stop_mode == "opposite_boundary":
                    if side == "LONG":
                        stop = float(mon_low - stop_mult * rng)
                    else:
                        stop = float(mon_high + stop_mult * rng)
                elif stop_mode == "range_fraction":
                    if side == "LONG":
                        stop = float(mon_low - stop_range_fraction * rng)
                    else:
                        stop = float(mon_high + stop_range_fraction * rng)
                else:  # fixed_pct
                    if side == "LONG":
                        stop = float(entry * (1.0 - stop_pct))
                    else:
                        stop = float(entry * (1.0 + stop_pct))

                tp1, tp2 = calculate_tp_levels(
                    side,
                    mon_low=float(mon_low),
                    mon_mid=float(mon_mid),
                    mon_high=float(mon_high),
                    tp2_range_fraction=(None if tp2_range_fraction is None else float(tp2_range_fraction)),
                    tp1_range_fraction=(None if tp1_range_fraction is None else float(tp1_range_fraction)),
                    tp2_to_full=float(tp2_to_full),
                    tp1_to_mid=float(tp1_to_mid),
                )
                dist = abs(entry - stop)

                qty = (risk_per_trade / dist) if dist > 0 else 0.0
                if qty > 0:
                    entry_notional = abs(entry * qty)
                    entry_fee = calculate_entry_fee(entry, qty, fee_bps=fee_bps)
                    equity -= entry_fee
                    df.at[ts, "fees_paid"] += entry_fee
                    trade_pnl_accum = -entry_fee
                    gross_pnl_accum = 0.0
                    fees_paid_accum = entry_fee
                    entry_weekday_utc = int(ts_utc.weekday())
                    entry_hour_utc = int(ts_utc.hour)
                    entry_mon_range = float(rng)
                    entry_mon_mid = float(mon_mid)
                    entry_mon_range_pct = float(range_pct) if pd.notna(range_pct) else None
                    in_pos = True
                    tp1_taken = False
                    has_traded_week = True

        # --- Manage exits ---
        if in_pos and qty > 0:
            # Stop check (intrabar)
            hit_stop = is_stop_hit(side, candle_high=float(row["high"]), candle_low=float(row["low"]), stop_price=float(stop))
            if hit_stop:
                pnl, exit_px, fee, gross = net_pnl_for_exit(stop, qty, side)
                equity += pnl
                trade_pnl_accum += pnl
                gross_pnl_accum += gross
                fees_paid_accum += fee
                ret = (trade_pnl_accum / entry_notional) if entry_notional > 0 else 0.0
                holding_hours = float((ts - entry_time).total_seconds() / 3600.0) if entry_time is not None else None
                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=ts,
                        side=side,
                        entry=entry,
                        exit=exit_px,
                        qty=qty,
                        pnl=trade_pnl_accum,
                        return_pct=ret,
                        reason="STOP",
                        week_id=week_id,
                        net_pnl=trade_pnl_accum,
                        gross_pnl=gross_pnl_accum,
                        fees_paid=fees_paid_accum,
                        holding_hours=holding_hours,
                        entry_weekday_utc=entry_weekday_utc,
                        entry_hour_utc=entry_hour_utc,
                        mon_range=entry_mon_range,
                        mon_mid=entry_mon_mid,
                        mon_range_pct=entry_mon_range_pct,
                    )
                )
                df.at[ts, "trade_pnl"] += pnl
                df.at[ts, "fees_paid"] += fee
                in_pos = False
                qty = 0.0

            # Partial TP1 at mid
            tp1_enabled = (not single_target_mode) and tp1_at_mid and (float(tp1_close_fraction) > 0.0)
            if in_pos and tp1_enabled and (not tp1_taken):
                hit_tp1 = is_tp_hit(side, candle_high=float(row["high"]), candle_low=float(row["low"]), tp_price=float(tp1))
                if hit_tp1:
                    exit_qty = qty * float(tp1_close_fraction)
                    pnl, exit_px, fee, gross = net_pnl_for_exit(tp1, exit_qty, side)
                    equity += pnl
                    df.at[ts, "trade_pnl"] += pnl
                    df.at[ts, "fees_paid"] += fee
                    trade_pnl_accum += pnl
                    gross_pnl_accum += gross
                    fees_paid_accum += fee
                    qty = qty - exit_qty
                    tp1_taken = True

                    if move_stop_to_breakeven_after_tp1:
                        if breakeven_includes_fees and fee_bps > 0:
                            fee_rate = fee_bps / 10_000.0
                            stop = entry * (1.0 + 2.0 * fee_rate) if side == "LONG" else entry * (1.0 - 2.0 * fee_rate)
                        else:
                            stop = entry

            # Final target (TP2 default; TP1 when single-target TP1 mode is enabled)
            if in_pos and qty > 0:
                target_price = float(tp1) if (single_target_mode and single_target_level == "tp1") else float(tp2)
                target_reason = "TP1" if (single_target_mode and single_target_level == "tp1") else "TP2"
                hit_target = is_tp_hit(side, candle_high=float(row["high"]), candle_low=float(row["low"]), tp_price=target_price)
                if hit_target:
                    pnl, exit_px, fee, gross = net_pnl_for_exit(target_price, qty, side)
                    equity += pnl
                    trade_pnl_accum += pnl
                    gross_pnl_accum += gross
                    fees_paid_accum += fee
                    ret = (trade_pnl_accum / entry_notional) if entry_notional > 0 else 0.0
                    holding_hours = float((ts - entry_time).total_seconds() / 3600.0) if entry_time is not None else None
                    trades.append(
                        Trade(
                            entry_time=entry_time,
                            exit_time=ts,
                            side=side,
                            entry=entry,
                            exit=exit_px,
                            qty=qty,
                            pnl=trade_pnl_accum,
                            return_pct=ret,
                            reason=target_reason,
                            week_id=week_id,
                            net_pnl=trade_pnl_accum,
                            gross_pnl=gross_pnl_accum,
                            fees_paid=fees_paid_accum,
                            holding_hours=holding_hours,
                            entry_weekday_utc=entry_weekday_utc,
                            entry_hour_utc=entry_hour_utc,
                            mon_range=entry_mon_range,
                            mon_mid=entry_mon_mid,
                            mon_range_pct=entry_mon_range_pct,
                        )
                    )
                    df.at[ts, "trade_pnl"] += pnl
                    df.at[ts, "fees_paid"] += fee
                    in_pos = False
                    qty = 0.0

            # Friday close-out
            if in_pos and qty > 0 and exit_friday_close and friday_cutoff_hit:
                pnl, exit_px, fee, gross = net_pnl_for_exit(float(row["close"]), qty, side)
                equity += pnl
                trade_pnl_accum += pnl
                gross_pnl_accum += gross
                fees_paid_accum += fee
                ret = (trade_pnl_accum / entry_notional) if entry_notional > 0 else 0.0
                holding_hours = float((ts - entry_time).total_seconds() / 3600.0) if entry_time is not None else None
                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=ts,
                        side=side,
                        entry=entry,
                        exit=exit_px,
                        qty=qty,
                        pnl=trade_pnl_accum,
                        return_pct=ret,
                        reason="FRIDAY",
                        week_id=week_id,
                        net_pnl=trade_pnl_accum,
                        gross_pnl=gross_pnl_accum,
                        fees_paid=fees_paid_accum,
                        holding_hours=holding_hours,
                        entry_weekday_utc=entry_weekday_utc,
                        entry_hour_utc=entry_hour_utc,
                        mon_range=entry_mon_range,
                        mon_mid=entry_mon_mid,
                        mon_range_pct=entry_mon_range_pct,
                    )
                )
                df.at[ts, "trade_pnl"] += pnl
                df.at[ts, "fees_paid"] += fee
                in_pos = False
                qty = 0.0

        df.at[ts, "position"] = (qty if in_pos else 0.0) * (1 if side == "LONG" else (-1 if side == "SHORT" else 0))
        df.at[ts, "equity"] = equity

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    df.attrs["strategy"] = strategy
    df.attrs["signal_count"] = int(signal_count)
    return df, trades_df
