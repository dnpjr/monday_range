from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.binance_data import download_klines, SUPPORTED_SYMBOLS, SUPPORTED_INTERVALS
from src.binance_cache import (
    resolve_time_window,
    load_cache_csv,
    save_cache_csv,
    merge_and_clean_candles,
    build_incremental_fetch_ranges,
)


def run_cache_update(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    lookback_days: int | None = 1825,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    cache_dir: str = "data/binance",
) -> dict:
    now_utc = pd.Timestamp.now(tz="UTC")
    lb_days = int(lookback_days) if lookback_days is not None else 1825
    start_utc, end_utc = resolve_time_window(
        now_utc=now_utc,
        lookback_days=lb_days,
        start=start,
        end=end,
    )

    cache_path = Path(cache_dir) / f"{symbol}_{interval}.csv"
    existing = load_cache_csv(cache_path)

    incoming_parts: list[pd.DataFrame] = []
    if force:
        incoming_parts.append(
            download_klines(
                symbol=symbol,
                interval=interval,
                start=start_utc,
                end=end_utc,
                use_cache=False,
                cache_format="parquet",
            )
        )
        base = pd.DataFrame(columns=existing.columns if len(existing.columns) else None)
    else:
        ranges = build_incremental_fetch_ranges(
            cache_df=existing,
            start_utc=start_utc,
            end_utc=end_utc,
            interval=interval,
        )
        for s, e in ranges:
            incoming_parts.append(
                download_klines(
                    symbol=symbol,
                    interval=interval,
                    start=s,
                    end=e,
                    use_cache=False,
                    cache_format="parquet",
                )
            )
        base = existing

    incoming = pd.concat(incoming_parts, ignore_index=True) if incoming_parts else pd.DataFrame()
    merged, duplicates_removed, missing_count = merge_and_clean_candles(base, incoming, interval)
    save_cache_csv(merged, cache_path)

    first_candle = merged["open_time"].iloc[0].isoformat() if len(merged) and "open_time" in merged.columns else "N/A"
    last_candle = merged["open_time"].iloc[-1].isoformat() if len(merged) and "open_time" in merged.columns else "N/A"
    missing_txt = str(missing_count) if missing_count is not None else "N/A"

    return {
        "symbol": symbol,
        "interval": interval,
        "rows_saved": int(len(merged)),
        "first_candle": first_candle,
        "last_candle": last_candle,
        "duplicates_removed": int(duplicates_removed),
        "missing_candle_count": missing_txt,
        "cache_file": str(cache_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/update public Binance candle cache.")
    ap.add_argument("--symbol", default="BTCUSDT", choices=sorted(SUPPORTED_SYMBOLS))
    ap.add_argument("--interval", default="1h", choices=sorted(SUPPORTED_INTERVALS))
    ap.add_argument("--lookback_days", type=int, default=1825)
    ap.add_argument("--start", default=None, help="Optional start date/time (UTC), e.g. 2021-01-01")
    ap.add_argument("--end", default=None, help="Optional end date/time (UTC), e.g. 2026-01-01")
    ap.add_argument("--force", action="store_true", help="Force full refresh for the selected window.")
    args = ap.parse_args()

    result = run_cache_update(
        symbol=args.symbol,
        interval=args.interval,
        lookback_days=args.lookback_days,
        start=args.start,
        end=args.end,
        force=args.force,
        cache_dir="data/binance",
    )

    print("=== Binance cache update summary ===")
    print(f"symbol: {result['symbol']}")
    print(f"interval: {result['interval']}")
    print(f"rows_saved: {result['rows_saved']}")
    print(f"first_candle: {result['first_candle']}")
    print(f"last_candle: {result['last_candle']}")
    print(f"duplicates_removed: {result['duplicates_removed']}")
    print(f"missing_candle_count: {result['missing_candle_count']}")
    print(f"cache_file: {result['cache_file']}")


if __name__ == "__main__":
    main()
