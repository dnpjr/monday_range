from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import warnings

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT"}
SUPPORTED_INTERVALS = {"15m", "1h", "4h", "1d"}
INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def _cache_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "binance"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_utc_ms(value: str | pd.Timestamp | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.value // 1_000_000)


def _validate_inputs(symbol: str, interval: str, start: Any, end: Any) -> tuple[int, int]:
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol '{symbol}'. Supported: {sorted(SUPPORTED_SYMBOLS)}")
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {sorted(SUPPORTED_INTERVALS)}")
    if start is None or end is None:
        raise ValueError("start and end are required")

    start_ms = _to_utc_ms(start)
    end_ms = _to_utc_ms(end)
    if end_ms <= start_ms:
        raise ValueError("end must be later than start")
    return start_ms, end_ms


def _parse_klines_response(rows: Iterable[Iterable[Any]]) -> pd.DataFrame:
    cols = [
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
        "ignore",
    ]

    df = pd.DataFrame(list(rows), columns=cols)
    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in float_cols:
        df[col] = df[col].astype(float)

    df["num_trades"] = df["num_trades"].astype(int)

    df = df.drop(columns=["ignore"]).sort_values("open_time")
    df = df.drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    return df


def _cache_path(symbol: str, interval: str, start_ms: int, end_ms: int, fmt: str) -> Path:
    return _cache_dir() / f"{symbol}_{interval}_{start_ms}_{end_ms}.{fmt}"


def _load_cached(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["open_time", "close_time"])

    if df.empty:
        return df

    if str(df["open_time"].dtype) != "datetime64[ns, UTC]":
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    if str(df["close_time"].dtype) != "datetime64[ns, UTC]":
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)

    return df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)


def _save_cached(df: pd.DataFrame, path: Path) -> Path:
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Parquet save failed ({exc}); falling back to CSV.")
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return csv_path

    df.to_csv(path, index=False)
    return path


def download_klines(
    symbol: str,
    interval: str,
    start: str | pd.Timestamp | int | float,
    end: str | pd.Timestamp | int | float,
    *,
    use_cache: bool = True,
    cache_format: str = "parquet",
) -> pd.DataFrame:
    """Download Binance public klines with 1000-row pagination.

    Returns a DataFrame sorted by open_time with UTC timestamps and float OHLCV fields.
    """
    start_ms, end_ms = _validate_inputs(symbol, interval, start, end)

    cache_format = cache_format.lower()
    if cache_format not in {"csv", "parquet"}:
        raise ValueError("cache_format must be 'csv' or 'parquet'")

    cache_path = _cache_path(symbol, interval, start_ms, end_ms, cache_format)
    if use_cache and cache_path.exists():
        return _load_cached(cache_path)

    all_rows: list[list[Any]] = []
    cursor = start_ms
    step_ms = INTERVAL_MS[interval]

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = int(rows[-1][0])
        next_cursor = last_open_time + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(rows) < 1000:
            break

    df = _parse_klines_response(all_rows)
    if not df.empty:
        df = df[(df["open_time"] >= pd.to_datetime(start_ms, unit="ms", utc=True)) & (df["open_time"] < pd.to_datetime(end_ms, unit="ms", utc=True))]
        df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)

    if use_cache:
        _save_cached(df, cache_path)

    return df
