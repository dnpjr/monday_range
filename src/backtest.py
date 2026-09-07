from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .execution_helpers import (
    apply_slippage,
    calculate_entry_fee,
    calculate_exit_fee,
    calculate_tp_levels,
    is_friday_cutoff_bar,
    is_stop_hit,
    is_tp_hit,
)
from .signals import sweep_rejection_signal


CANONICAL_STOP_MODES = {"swept_boundary", "swept_boundary_offset", "entry_fixed_pct"}
LEGACY_STOP_MODE_MAP = {
    "opposite_boundary": "swept_boundary_offset",
    "range_fraction": "swept_boundary_offset",
    "fixed_pct": "entry_fixed_pct",
}
INTRABAR_POLICIES = {"conservative_stop_first", "target_first"}
RISK_BASES = {"current_equity", "initial_capital"}


def canonical_stop_mode(stop_mode: str) -> str:
    """Return the explicit stop name, accepting saved legacy configuration names."""
    if stop_mode in LEGACY_STOP_MODE_MAP:
        warnings.warn(
            f"stop_mode={stop_mode!r} is deprecated; use {LEGACY_STOP_MODE_MAP[stop_mode]!r}",
            DeprecationWarning,
            stacklevel=2,
        )
        return LEGACY_STOP_MODE_MAP[stop_mode]
    if stop_mode not in CANONICAL_STOP_MODES:
        valid = sorted(CANONICAL_STOP_MODES | set(LEGACY_STOP_MODE_MAP))
        raise ValueError(f"stop_mode must be one of {valid}")
    return stop_mode


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
    entry_slippage_cost: float | None = None
    exit_slippage_cost: float | None = None
    total_slippage_cost: float | None = None
    combined_execution_cost: float | None = None
    holding_hours: float | None = None
    entry_weekday_utc: int | None = None
    entry_hour_utc: int | None = None
    mon_range: float | None = None
    mon_mid: float | None = None
    mon_range_pct: float | None = None
    initial_qty: float | None = None
    entry_notional: float | None = None
    risk_capital: float | None = None
    max_notional: float | None = None
    sizing_limited_by: str | None = None
    initial_stop: float | None = None
    tp1_price: float | None = None
    tp2_price: float | None = None


def _gap_aware_raw_fill(
    side: str,
    *,
    trigger_kind: str,
    bar_open: float,
    trigger_price: float,
) -> float:
    """Choose the first executable raw price when the bar opens through a trigger."""
    if trigger_kind == "stop":
        opened_through = bar_open <= trigger_price if side == "LONG" else bar_open >= trigger_price
    elif trigger_kind == "target":
        opened_through = bar_open >= trigger_price if side == "LONG" else bar_open <= trigger_price
    else:
        raise ValueError("trigger_kind must be 'stop' or 'target'")
    return float(bar_open if opened_through else trigger_price)


def _opened_through(side: str, *, trigger_kind: str, bar_open: float, trigger_price: float) -> bool:
    return _gap_aware_raw_fill(
        side,
        trigger_kind=trigger_kind,
        bar_open=bar_open,
        trigger_price=trigger_price,
    ) == float(bar_open) and float(bar_open) != float(trigger_price)


def backtest_sweep_fade(
    df: pd.DataFrame,
    *,
    strategy: str = "current_monday_range",
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 100.0,
    risk_fraction: float | None = None,
    risk_base: str = "current_equity",
    max_leverage: float = 1.0,
    stop_mult: float = 1.0,
    tp1_at_mid: bool = True,
    tp1_frac: float = 0.5,
    tp2_to_full: float = 1.0,
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
    stop_mode: str = "swept_boundary_offset",
    stop_range_fraction: float = 1.0,
    stop_pct: float = 0.01,
    move_stop_to_breakeven_after_tp1: bool = True,
    breakeven_includes_fees: bool = False,
    single_target_mode: bool = False,
    single_target_level: str = "tp2",
    intrabar_policy: str = "conservative_stop_first",
    close_open_position_at_end: bool = True,
    entry_start_utc: str | pd.Timestamp | None = None,
    entry_end_exclusive_utc: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Monday-range sweep/fade strategy with marked portfolio accounting.

    Signals are calculated from the completed previous bar and entries occur at the
    next bar open. Open positions are marked at every bar close. ``cash`` is the
    realised account balance after fees; it does not reserve spot notional.
    ``equity`` equals cash plus unrealised P&L.

    Passing ``risk_fraction`` uses either current marked equity (the research
    default) or initial capital according to ``risk_base``. Leaving it as ``None``
    retains the explicit legacy fixed-dollar ``risk_per_trade`` mode.
    """
    if df.empty:
        raise ValueError("df must contain at least one OHLC row")

    valid_directions = {"both", "long_only", "short_only"}
    valid_strategies = {"current_monday_range", "sweep_retest"}
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {sorted(valid_strategies)}")
    if direction not in valid_directions:
        raise ValueError(f"direction must be one of {sorted(valid_directions)}")
    if risk_base not in RISK_BASES:
        raise ValueError(f"risk_base must be one of {sorted(RISK_BASES)}")
    if risk_fraction is not None and float(risk_fraction) <= 0:
        raise ValueError("risk_fraction must be > 0 when provided")
    if risk_fraction is None and float(risk_per_trade) <= 0:
        raise ValueError("risk_per_trade must be > 0 in fixed-dollar mode")
    if float(max_leverage) <= 0:
        raise ValueError("max_leverage must be > 0")
    if intrabar_policy not in INTRABAR_POLICIES:
        raise ValueError(f"intrabar_policy must be one of {sorted(INTRABAR_POLICIES)}")
    if max_entry_day_utc is not None and not (0 <= int(max_entry_day_utc) <= 6):
        raise ValueError("max_entry_day_utc must be between 0 (Monday) and 6 (Sunday).")
    if max_entry_hour_utc is not None and not (0 <= int(max_entry_hour_utc) <= 23):
        raise ValueError("max_entry_hour_utc must be between 0 and 23.")
    entry_start = pd.Timestamp(entry_start_utc) if entry_start_utc is not None else None
    entry_end = pd.Timestamp(entry_end_exclusive_utc) if entry_end_exclusive_utc is not None else None
    if entry_start is not None:
        entry_start = entry_start.tz_localize("UTC") if entry_start.tzinfo is None else entry_start.tz_convert("UTC")
    if entry_end is not None:
        entry_end = entry_end.tz_localize("UTC") if entry_end.tzinfo is None else entry_end.tz_convert("UTC")
    if entry_start is not None and entry_end is not None and entry_end <= entry_start:
        raise ValueError("entry_end_exclusive_utc must be later than entry_start_utc")
    if sma_period is not None and int(sma_period) < 1:
        raise ValueError("sma_period must be >= 1 when provided.")
    if min_range_pct is not None and max_range_pct is not None and float(min_range_pct) > float(max_range_pct):
        raise ValueError("min_range_pct cannot be greater than max_range_pct.")

    requested_stop_mode = stop_mode
    resolved_stop_mode = canonical_stop_mode(stop_mode)
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

    # The legacy name used stop_mult; all explicit offset modes use stop_range_fraction.
    effective_stop_range_fraction = float(stop_mult) if requested_stop_mode == "opposite_boundary" else float(stop_range_fraction)

    df = df.copy()
    for column, default in {
        "equity": np.nan,
        "cash": np.nan,
        "realized_pnl": np.nan,
        "unrealized_pnl": 0.0,
        "position": 0.0,
        "open_qty": 0.0,
        "entry_price": np.nan,
        "mark_price": np.nan,
        "position_value": 0.0,
        "gross_notional": 0.0,
        "trade_pnl": 0.0,
        "fees_paid": 0.0,
        "slippage_cost": 0.0,
        "combined_execution_cost": 0.0,
        "turnover": 0.0,
        "cumulative_turnover": 0.0,
        "exposed": 0.0,
    }.items():
        df[column] = default
    if sma_period is not None:
        df["_sma"] = df["close"].rolling(window=int(sma_period), min_periods=int(sma_period)).mean()

    cash = float(initial_capital)
    cumulative_turnover = 0.0
    trades: List[Trade] = []

    in_pos = False
    side: str | None = None
    entry_time: pd.Timestamp | None = None
    entry_week_id: tuple[int, int] | None = None
    entry = 0.0
    entry_notional = 0.0
    qty = 0.0
    initial_qty = 0.0
    stop = 0.0
    initial_stop = 0.0
    tp1 = 0.0
    tp2 = 0.0
    tp1_taken = False
    trade_pnl_accum = 0.0
    gross_pnl_accum = 0.0
    fees_paid_accum = 0.0
    entry_slippage_cost_accum = 0.0
    exit_slippage_cost_accum = 0.0
    trade_risk_capital = 0.0
    trade_max_notional = 0.0
    sizing_limited_by = "risk"
    entry_weekday_utc: int | None = None
    entry_hour_utc: int | None = None
    entry_mon_range: float | None = None
    entry_mon_mid: float | None = None
    entry_mon_range_pct: float | None = None
    week_id: tuple[int, int] | None = None
    has_traded_week = False
    first_signal_seen_week = False
    signal_count = 0

    freq_delta = (df.index[1] - df.index[0]) if len(df.index) > 1 else pd.Timedelta(hours=1)

    def current_unrealized(mark: float) -> float:
        if not in_pos or qty <= 0 or side is None:
            return 0.0
        return float((mark - entry) * qty if side == "LONG" else (entry - mark) * qty)

    def record_portfolio(ts: pd.Timestamp, mark: float, bar_turnover: float, bar_exposed: bool = False) -> None:
        nonlocal cumulative_turnover
        unrealized = current_unrealized(mark)
        signed_qty = qty * (1.0 if side == "LONG" else -1.0) if in_pos else 0.0
        cumulative_turnover += float(bar_turnover)
        df.at[ts, "cash"] = cash
        df.at[ts, "realized_pnl"] = cash - float(initial_capital)
        df.at[ts, "unrealized_pnl"] = unrealized
        df.at[ts, "equity"] = cash + unrealized
        df.at[ts, "position"] = signed_qty
        df.at[ts, "open_qty"] = qty if in_pos else 0.0
        df.at[ts, "entry_price"] = entry if in_pos else np.nan
        df.at[ts, "mark_price"] = float(mark)
        df.at[ts, "position_value"] = signed_qty * float(mark)
        df.at[ts, "gross_notional"] = abs(qty * float(mark)) if in_pos else 0.0
        df.at[ts, "turnover"] = float(bar_turnover)
        df.at[ts, "cumulative_turnover"] = cumulative_turnover
        df.at[ts, "exposed"] = 1.0 if bar_exposed else 0.0

    first_ts = df.index[0]
    record_portfolio(first_ts, float(df.iloc[0]["close"]), 0.0)

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
        return max_entry_hour_utc is None or hour <= int(max_entry_hour_utc)

    def protocol_window_allows(ts_utc: pd.Timestamp) -> bool:
        if entry_start is not None and ts_utc < entry_start:
            return False
        if entry_end is not None and ts_utc >= entry_end:
            return False
        return True

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        row = df.iloc[i]
        ts = row.name
        bar_turnover = 0.0
        bar_exposed = bool(in_pos)

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
            record_portfolio(ts, float(row["close"]), bar_turnover, bar_exposed)
            continue

        ts_utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        bar_open_ts = ts_utc
        bar_close_ts = ts_utc + freq_delta
        friday_cutoff_hit = is_friday_cutoff_bar(
            bar_open_ts,
            bar_close_ts,
            friday_cutoff_hour_utc=friday_cutoff_hour_utc,
        )
        range_pct = (float(rng) / abs(float(mon_mid))) if (pd.notna(mon_mid) and float(mon_mid) != 0.0) else np.nan

        # Signal is based only on the completed previous bar; execution is this bar's open.
        if (not in_pos) and row["is_tradeable"] and (not has_traded_week) and protocol_window_allows(ts_utc):
            signal_side = None
            if strategy == "current_monday_range":
                if row["weekday"] < 4:
                    if min_range_pct is not None and (pd.isna(range_pct) or float(range_pct) < float(min_range_pct)):
                        signal_side = None
                    elif max_range_pct is not None and (pd.isna(range_pct) or float(range_pct) > float(max_range_pct)):
                        signal_side = None
                    elif not entry_cutoff_allows(ts_utc):
                        signal_side = None
                    else:
                        signal_side = sweep_rejection_signal(prev, mon_low=mon_low, mon_high=mon_high)
            elif not first_signal_seen_week:
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
                candidate_side = signal_side
                candidate_entry = apply_slippage(
                    float(row["open"]),
                    candidate_side,
                    action="entry",
                    slippage_bps=slippage_bps,
                )

                if resolved_stop_mode == "swept_boundary":
                    candidate_stop = float(mon_low if candidate_side == "LONG" else mon_high)
                elif resolved_stop_mode == "swept_boundary_offset":
                    candidate_stop = float(
                        mon_low - effective_stop_range_fraction * rng
                        if candidate_side == "LONG"
                        else mon_high + effective_stop_range_fraction * rng
                    )
                else:
                    candidate_stop = float(
                        candidate_entry * (1.0 - stop_pct)
                        if candidate_side == "LONG"
                        else candidate_entry * (1.0 + stop_pct)
                    )

                candidate_tp1, candidate_tp2 = calculate_tp_levels(
                    candidate_side,
                    mon_low=float(mon_low),
                    mon_mid=float(mon_mid),
                    mon_high=float(mon_high),
                    tp2_range_fraction=(None if tp2_range_fraction is None else float(tp2_range_fraction)),
                    tp1_range_fraction=(None if tp1_range_fraction is None else float(tp1_range_fraction)),
                    tp2_to_full=float(tp2_to_full),
                    tp1_to_mid=float(tp1_to_mid),
                )
                stop_distance = abs(candidate_entry - candidate_stop)
                equity_for_sizing = cash  # Entries are only allowed while flat.
                risk_basis_value = equity_for_sizing if risk_base == "current_equity" else float(initial_capital)
                risk_capital = (
                    risk_basis_value * float(risk_fraction)
                    if risk_fraction is not None
                    else float(risk_per_trade)
                )
                risk_sized_qty = risk_capital / stop_distance if stop_distance > 0 and risk_capital > 0 else 0.0
                allowed_notional = max(0.0, equity_for_sizing) * float(max_leverage)
                leverage_sized_qty = allowed_notional / abs(candidate_entry) if candidate_entry != 0 else 0.0
                candidate_qty = min(risk_sized_qty, leverage_sized_qty)

                if candidate_qty > 0:
                    side = candidate_side
                    entry_time = ts
                    entry_week_id = curr_week
                    entry = candidate_entry
                    stop = candidate_stop
                    initial_stop = candidate_stop
                    tp1 = float(candidate_tp1)
                    tp2 = float(candidate_tp2)
                    qty = candidate_qty
                    initial_qty = candidate_qty
                    entry_notional = abs(entry * qty)
                    trade_risk_capital = risk_capital
                    trade_max_notional = allowed_notional
                    sizing_limited_by = "max_leverage" if leverage_sized_qty < risk_sized_qty else "stop_risk"
                    entry_fee = calculate_entry_fee(entry, qty, fee_bps=fee_bps)
                    entry_slippage_cost = abs(candidate_entry - float(row["open"])) * qty
                    cash -= entry_fee
                    bar_turnover += entry_notional
                    df.at[ts, "fees_paid"] += entry_fee
                    df.at[ts, "slippage_cost"] += entry_slippage_cost
                    df.at[ts, "combined_execution_cost"] += entry_fee + entry_slippage_cost
                    trade_pnl_accum = -entry_fee
                    gross_pnl_accum = 0.0
                    fees_paid_accum = entry_fee
                    entry_slippage_cost_accum = entry_slippage_cost
                    exit_slippage_cost_accum = 0.0
                    entry_weekday_utc = int(ts_utc.weekday())
                    entry_hour_utc = int(ts_utc.hour)
                    entry_mon_range = float(rng)
                    entry_mon_mid = float(mon_mid)
                    entry_mon_range_pct = float(range_pct) if pd.notna(range_pct) else None
                    in_pos = True
                    bar_exposed = True
                    tp1_taken = False
                    has_traded_week = True

        def exit_quantity(exit_qty: float, raw_price: float) -> tuple[float, float, float, float]:
            nonlocal cash, qty, trade_pnl_accum, gross_pnl_accum, fees_paid_accum, exit_slippage_cost_accum, bar_turnover
            assert side is not None
            exit_px = apply_slippage(raw_price, side, action="exit", slippage_bps=slippage_bps)
            gross = (exit_px - entry) * exit_qty if side == "LONG" else (entry - exit_px) * exit_qty
            fee = calculate_exit_fee(exit_px, exit_qty, fee_bps=fee_bps)
            exit_slippage_cost = abs(exit_px - float(raw_price)) * exit_qty
            net = gross - fee
            cash += net
            qty -= exit_qty
            if abs(qty) < 1e-12:
                qty = 0.0
            trade_pnl_accum += net
            gross_pnl_accum += gross
            fees_paid_accum += fee
            exit_slippage_cost_accum += exit_slippage_cost
            bar_turnover += abs(exit_px * exit_qty)
            df.at[ts, "trade_pnl"] += net
            df.at[ts, "fees_paid"] += fee
            df.at[ts, "slippage_cost"] += exit_slippage_cost
            df.at[ts, "combined_execution_cost"] += fee + exit_slippage_cost
            return net, exit_px, fee, gross

        def finish_trade(reason: str, raw_price: float) -> None:
            nonlocal in_pos, qty
            assert side is not None and entry_time is not None and entry_week_id is not None
            final_qty = qty
            _, exit_px, _, _ = exit_quantity(final_qty, raw_price)
            ret = (trade_pnl_accum / entry_notional) if entry_notional > 0 else 0.0
            holding_hours = float((ts - entry_time).total_seconds() / 3600.0)
            trades.append(
                Trade(
                    entry_time=entry_time,
                    exit_time=ts,
                    side=side,
                    entry=entry,
                    exit=exit_px,
                    qty=final_qty,
                    pnl=trade_pnl_accum,
                    return_pct=ret,
                    reason=reason,
                    week_id=entry_week_id,
                    net_pnl=trade_pnl_accum,
                    gross_pnl=gross_pnl_accum,
                    fees_paid=fees_paid_accum,
                    entry_slippage_cost=entry_slippage_cost_accum,
                    exit_slippage_cost=exit_slippage_cost_accum,
                    total_slippage_cost=entry_slippage_cost_accum + exit_slippage_cost_accum,
                    combined_execution_cost=fees_paid_accum + entry_slippage_cost_accum + exit_slippage_cost_accum,
                    holding_hours=holding_hours,
                    entry_weekday_utc=entry_weekday_utc,
                    entry_hour_utc=entry_hour_utc,
                    mon_range=entry_mon_range,
                    mon_mid=entry_mon_mid,
                    mon_range_pct=entry_mon_range_pct,
                    initial_qty=initial_qty,
                    entry_notional=entry_notional,
                    risk_capital=trade_risk_capital,
                    max_notional=trade_max_notional,
                    sizing_limited_by=sizing_limited_by,
                    initial_stop=initial_stop,
                    tp1_price=tp1,
                    tp2_price=tp2,
                )
            )
            in_pos = False
            qty = 0.0

        def take_tp1(raw_price: float) -> None:
            nonlocal qty, tp1_taken, stop
            exit_qty = qty * float(tp1_close_fraction)
            if exit_qty <= 0:
                return
            if exit_qty >= qty - 1e-12:
                tp1_taken = True
                finish_trade("TP1", raw_price)
                return
            exit_quantity(exit_qty, raw_price)
            tp1_taken = True
            if move_stop_to_breakeven_after_tp1:
                if breakeven_includes_fees and fee_bps > 0:
                    fee_rate = fee_bps / 10_000.0
                    stop = entry * (1.0 + 2.0 * fee_rate) if side == "LONG" else entry * (1.0 - 2.0 * fee_rate)
                else:
                    stop = entry

        if in_pos and qty > 0 and side is not None:
            bar_open = float(row["open"])
            candle_high = float(row["high"])
            candle_low = float(row["low"])
            tp1_enabled = (not single_target_mode) and tp1_at_mid and float(tp1_close_fraction) > 0.0
            final_target = float(tp1) if (single_target_mode and single_target_level == "tp1") else float(tp2)
            final_reason = "TP1" if (single_target_mode and single_target_level == "tp1") else "TP2"

            stop_open_gap = _opened_through(
                side,
                trigger_kind="stop",
                bar_open=bar_open,
                trigger_price=float(stop),
            )
            tp1_open_gap = tp1_enabled and (not tp1_taken) and _opened_through(
                side,
                trigger_kind="target",
                bar_open=bar_open,
                trigger_price=float(tp1),
            )
            final_open_gap = _opened_through(
                side,
                trigger_kind="target",
                bar_open=bar_open,
                trigger_price=final_target,
            )

            def process_targets() -> None:
                if not in_pos or qty <= 0:
                    return
                if tp1_enabled and (not tp1_taken) and is_tp_hit(
                    side,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    tp_price=float(tp1),
                ):
                    raw = _gap_aware_raw_fill(
                        side,
                        trigger_kind="target",
                        bar_open=bar_open,
                        trigger_price=float(tp1),
                    )
                    take_tp1(raw)
                if in_pos and qty > 0 and is_tp_hit(
                    side,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    tp_price=final_target,
                ):
                    raw = _gap_aware_raw_fill(
                        side,
                        trigger_kind="target",
                        bar_open=bar_open,
                        trigger_price=final_target,
                    )
                    finish_trade(final_reason, raw)

            def process_stop() -> None:
                if in_pos and qty > 0 and is_stop_hit(
                    side,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    stop_price=float(stop),
                ):
                    raw = _gap_aware_raw_fill(
                        side,
                        trigger_kind="stop",
                        bar_open=bar_open,
                        trigger_price=float(stop),
                    )
                    finish_trade("STOP", raw)

            # Gaps occur at the bar open and therefore precede ambiguous intrabar touches.
            if stop_open_gap:
                process_stop()
            elif tp1_open_gap or final_open_gap:
                # An opening target gap is known to occur before the bar's range.
                # Do not reuse the same OHLC low/high against a newly moved stop.
                process_targets()
            elif intrabar_policy == "conservative_stop_first":
                process_stop()
                process_targets()
            else:
                process_targets()
                process_stop()

            if in_pos and qty > 0 and exit_friday_close and friday_cutoff_hit:
                finish_trade("FRIDAY", float(row["close"]))

            if in_pos and qty > 0 and close_open_position_at_end and i == len(df) - 1:
                finish_trade("END_OF_DATA", float(row["close"]))

        record_portfolio(ts, float(row["close"]), bar_turnover, bar_exposed)

    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
    df.attrs.update(
        {
            "strategy": strategy,
            "accounting_version": "marked_equity_v1",
            "signal_count": int(signal_count),
            "mark_price": "bar_close",
            "risk_mode": "fraction" if risk_fraction is not None else "fixed_dollar",
            "risk_base": risk_base if risk_fraction is not None else "fixed_dollar",
            "max_leverage": float(max_leverage),
            "intrabar_policy": intrabar_policy,
            "stop_mode": resolved_stop_mode,
            "legacy_stop_mode": requested_stop_mode if requested_stop_mode in LEGACY_STOP_MODE_MAP else None,
            "close_open_position_at_end": bool(close_open_position_at_end),
            "entry_start_utc": None if entry_start is None else entry_start.isoformat(),
            "entry_end_exclusive_utc": None if entry_end is None else entry_end.isoformat(),
            "short_position_interpretation": "synthetic_research_position_on_binance_spot_price_series",
        }
    )
    return df, trades_df
