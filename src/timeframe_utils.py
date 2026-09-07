from __future__ import annotations

import re
from typing import Literal

import pandas as pd


_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([mhdMHD])\s*$")


def timeframe_to_minutes(interval: str) -> int:
    match = _TIMEFRAME_RE.match(str(interval))
    if not match:
        raise ValueError(f"Unsupported timeframe: {interval!r}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    if value <= 0:
        raise ValueError(f"Timeframe must be positive: {interval!r}")
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 24 * 60
    raise ValueError(f"Unsupported timeframe unit in {interval!r}")


def timeframe_to_timedelta(interval: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=timeframe_to_minutes(interval))


def compare_timeframes(a: str, b: str) -> Literal[-1, 0, 1]:
    a_min = timeframe_to_minutes(a)
    b_min = timeframe_to_minutes(b)
    if a_min < b_min:
        return -1
    if a_min > b_min:
        return 1
    return 0


def can_resample(base_interval: str, target_interval: str) -> bool:
    base_min = timeframe_to_minutes(base_interval)
    target_min = timeframe_to_minutes(target_interval)
    if target_min < base_min:
        return False
    return (target_min % base_min) == 0


def _resample_rule(interval: str) -> str:
    match = _TIMEFRAME_RE.match(str(interval))
    if not match:
        raise ValueError(f"Unsupported timeframe: {interval!r}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "m":
        return f"{value}min"
    if unit == "h":
        return f"{value}h"
    if unit == "d":
        return f"{value}D"
    raise ValueError(f"Unsupported timeframe: {interval!r}")


def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    elif "open_time" in df.columns:
        out = df.copy()
        out["open_time"] = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
        out = out.dropna(subset=["open_time"]).set_index("open_time")
    else:
        raise ValueError("OHLCV frame must have DatetimeIndex or open_time column.")

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for c in ("open", "high", "low", "close"):
        if c not in out.columns:
            raise ValueError(f"Missing required OHLC column: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"])


def resample_ohlcv(
    df: pd.DataFrame,
    *,
    base_interval: str,
    target_interval: str,
    drop_incomplete_final: bool = True,
) -> pd.DataFrame:
    out = _normalize_ohlcv_frame(df)
    if out.empty:
        return out
    if not can_resample(base_interval, target_interval):
        raise ValueError(
            f"Cannot resample from {base_interval} to {target_interval}. "
            "Target timeframe must be an integer multiple of base timeframe."
        )
    if compare_timeframes(base_interval, target_interval) == 0:
        return out

    rule = _resample_rule(target_interval)
    base_min = timeframe_to_minutes(base_interval)
    target_min = timeframe_to_minutes(target_interval)
    expected_count = int(target_min / base_min)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in out.columns:
        agg["volume"] = "sum"
    if "close_time" in out.columns:
        agg["close_time"] = "last"

    grouped = out.resample(
        rule,
        label="left",
        closed="left",
        origin="start_day",
    )
    bars = grouped.agg(agg)
    counts = grouped["open"].count()
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars = bars[counts.reindex(bars.index).fillna(0).astype(int) > 0]

    if drop_incomplete_final and not bars.empty:
        counts = counts.reindex(bars.index).fillna(0).astype(int)
        first_times = grouped["open"].apply(lambda series: series.index.min())
        last_times = grouped["open"].apply(lambda series: series.index.max())
        expected_last = bars.index + timeframe_to_timedelta(base_interval) * (expected_count - 1)
        complete = (
            (counts == expected_count)
            & (first_times.reindex(bars.index) == bars.index)
            & (last_times.reindex(bars.index) == expected_last)
        )
        dropped = [ts.isoformat() for ts in bars.index[~complete]]
        bars = bars.loc[complete]
        bars.attrs["dropped_incomplete_periods"] = dropped

    if "close_time" not in bars.columns:
        delta = timeframe_to_timedelta(target_interval)
        bars["close_time"] = bars.index + delta - pd.Timedelta(milliseconds=1)
    return bars.sort_index()
