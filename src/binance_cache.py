from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .binance_data import INTERVAL_MS


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


def to_utc_timestamp(value: Any, field_name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def resolve_time_window(
    *,
    now_utc: pd.Timestamp,
    lookback_days: int,
    start: str | None,
    end: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_utc = to_utc_timestamp(end, "end") if end is not None else now_utc
    if start is not None or end is not None:
        start_utc = to_utc_timestamp(start, "start") if start is not None else (end_utc - pd.Timedelta(days=lookback_days))
    else:
        start_utc = end_utc - pd.Timedelta(days=lookback_days)

    if end_utc <= start_utc:
        raise ValueError("end must be later than start")
    return start_utc, end_utc


def _interval_delta(interval: str) -> pd.Timedelta:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval!r}")
    return pd.Timedelta(milliseconds=INTERVAL_MS[interval])


def load_cache_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS)
    df = pd.read_csv(p)
    for col in ("open_time", "close_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "num_trades" in df.columns:
        df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").fillna(0).astype(int)
    return df


def save_cache_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in KLINE_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[KLINE_COLUMNS]
    out.to_csv(p, index=False)


def merge_and_clean_candles(existing: pd.DataFrame, incoming: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, int, int | None]:
    frames = []
    if existing is not None and len(existing):
        frames.append(existing.copy())
    if incoming is not None and len(incoming):
        frames.append(incoming.copy())
    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS), 0, None

    combined = pd.concat(frames, ignore_index=True)
    dup_count = int(combined["open_time"].duplicated().sum()) if "open_time" in combined.columns else 0
    combined = combined.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    missing = detect_missing_candles(combined, interval)
    return combined, dup_count, missing


def detect_missing_candles(candles: pd.DataFrame, interval: str) -> int | None:
    if candles is None or candles.empty or "open_time" not in candles.columns or len(candles) < 2:
        return None
    delta = _interval_delta(interval)
    ts = candles["open_time"].sort_values().reset_index(drop=True)
    total_span = ts.iloc[-1] - ts.iloc[0]
    if delta <= pd.Timedelta(0):
        return None
    expected = int(total_span / delta) + 1
    return max(0, expected - len(ts))


def build_incremental_fetch_ranges(
    *,
    cache_df: pd.DataFrame,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    interval: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if end_utc <= start_utc:
        return []
    if cache_df is None or cache_df.empty or "open_time" not in cache_df.columns:
        return [(start_utc, end_utc)]

    delta = _interval_delta(interval)
    cache_sorted = cache_df.sort_values("open_time")
    first_open = cache_sorted["open_time"].iloc[0]
    last_open = cache_sorted["open_time"].iloc[-1]
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    if start_utc < first_open:
        ranges.append((start_utc, first_open))

    tail_start = max(last_open + delta, start_utc)
    if tail_start < end_utc:
        ranges.append((tail_start, end_utc))

    return ranges
