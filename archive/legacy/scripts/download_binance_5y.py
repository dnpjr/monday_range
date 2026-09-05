from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.binance_data import download_klines


def main() -> None:
    ap = argparse.ArgumentParser(description="Download last 5 years of Binance BTCUSDT 1h klines.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--cache_format", choices=["csv", "parquet"], default="parquet")
    args = ap.parse_args()

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.DateOffset(years=5)

    df = download_klines(
        symbol=args.symbol,
        interval=args.interval,
        start=start,
        end=end,
        use_cache=True,
        cache_format=args.cache_format,
    )

    root = Path(__file__).resolve().parent
    out_dir = root / "data" / "binance"
    print(f"Downloaded {len(df)} candles into cache under: {out_dir}")
    if not df.empty:
        print(f"Range: {df['open_time'].min()} -> {df['open_time'].max()}")


if __name__ == "__main__":
    main()
