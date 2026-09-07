from __future__ import annotations

import pandas as pd


def apply_slippage(price: float, side: str, action: str, slippage_bps: float) -> float:
    slip_rate = slippage_bps / 10_000.0
    if slip_rate <= 0:
        return float(price)

    if action == "entry":
        return float(price) * (1.0 + slip_rate) if side == "LONG" else float(price) * (1.0 - slip_rate)
    if action == "exit":
        return float(price) * (1.0 - slip_rate) if side == "LONG" else float(price) * (1.0 + slip_rate)
    raise ValueError("action must be 'entry' or 'exit'")


def calculate_entry_fee(entry_price: float, qty: float, fee_bps: float) -> float:
    fee_rate = fee_bps / 10_000.0
    return abs(float(entry_price) * float(qty)) * fee_rate


def calculate_exit_fee(exit_price: float, qty: float, fee_bps: float) -> float:
    fee_rate = fee_bps / 10_000.0
    return abs(float(exit_price) * float(qty)) * fee_rate


def calculate_tp_levels(
    side: str,
    mon_low: float,
    mon_mid: float,
    mon_high: float,
    tp2_range_fraction: float | None = None,
    tp1_range_fraction: float | None = None,
    tp2_to_full: float | None = None,
    tp1_to_mid: float | None = None,
) -> tuple[float, float]:
    # Backward-compatible mapping from legacy parameters:
    # - tp1_to_mid=1.0 (legacy default) => tp1_range_fraction=0.5 (midpoint)
    # - tp2_to_full=1.0 (legacy default) => tp2_range_fraction=1.0 (opposite boundary)
    if tp1_range_fraction is None:
        legacy_tp1 = 1.0 if tp1_to_mid is None else float(tp1_to_mid)
        tp1_range_fraction = 0.5 * legacy_tp1
    if tp2_range_fraction is None:
        legacy_tp2 = 1.0 if tp2_to_full is None else float(tp2_to_full)
        tp2_range_fraction = 1.0 if legacy_tp2 >= 1.0 else (0.5 + 0.5 * legacy_tp2)

    rng = float(mon_high) - float(mon_low)
    if side == "LONG":
        tp1 = float(mon_low + float(tp1_range_fraction) * rng)
        tp2 = float(mon_low + float(tp2_range_fraction) * rng)
    else:
        tp1 = float(mon_high - float(tp1_range_fraction) * rng)
        tp2 = float(mon_high - float(tp2_range_fraction) * rng)
    return tp1, tp2


def is_stop_hit(side: str, candle_high: float, candle_low: float, stop_price: float) -> bool:
    return bool(candle_low <= stop_price) if side == "LONG" else bool(candle_high >= stop_price)


def is_tp_hit(side: str, candle_high: float, candle_low: float, tp_price: float) -> bool:
    return bool(candle_high >= tp_price) if side == "LONG" else bool(candle_low <= tp_price)


def is_friday_cutoff_bar(
    bar_open: pd.Timestamp,
    bar_close: pd.Timestamp,
    friday_cutoff_hour_utc: int = 23,
) -> bool:
    ts_utc = bar_open.tz_convert("UTC") if bar_open.tzinfo is not None else bar_open.tz_localize("UTC")
    bar_close_utc = bar_close.tz_convert("UTC") if bar_close.tzinfo is not None else bar_close.tz_localize("UTC")
    cutoff_ts = ts_utc.normalize() + pd.Timedelta(hours=friday_cutoff_hour_utc)
    return bool((ts_utc.weekday() == 4) and (ts_utc <= cutoff_ts < bar_close_utc))
