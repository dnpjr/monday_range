from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.binance_cache import (
    resolve_time_window,
    merge_and_clean_candles,
    build_incremental_fetch_ranges,
)


class BinanceCacheTests(unittest.TestCase):
    def test_resolve_time_window_default_lookback(self) -> None:
        now = pd.Timestamp("2026-01-01T00:00:00+00:00")
        start, end = resolve_time_window(now_utc=now, lookback_days=30, start=None, end=None)
        self.assertEqual(end, now)
        self.assertEqual(start, now - pd.Timedelta(days=30))

    def test_resolve_time_window_start_end_override(self) -> None:
        now = pd.Timestamp("2026-01-01T00:00:00+00:00")
        start, end = resolve_time_window(now_utc=now, lookback_days=365, start="2024-01-01", end="2024-06-01")
        self.assertEqual(start, pd.Timestamp("2024-01-01", tz="UTC"))
        self.assertEqual(end, pd.Timestamp("2024-06-01", tz="UTC"))

    def test_resolve_time_window_invalid_date(self) -> None:
        now = pd.Timestamp("2026-01-01T00:00:00+00:00")
        with self.assertRaises(ValueError):
            resolve_time_window(now_utc=now, lookback_days=30, start="bad-date", end=None)

    def test_merge_and_deduplicate(self) -> None:
        existing = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    ["2024-01-01 00:00:00+00:00", "2024-01-01 01:00:00+00:00"], utc=True
                ),
                "close_time": pd.to_datetime(
                    ["2024-01-01 00:59:59.999+00:00", "2024-01-01 01:59:59.999+00:00"], utc=True
                ),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1.0, 1.0],
                "quote_asset_volume": [1.0, 1.0],
                "num_trades": [1, 1],
                "taker_buy_base_asset_volume": [1.0, 1.0],
                "taker_buy_quote_asset_volume": [1.0, 1.0],
            }
        )
        incoming = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    ["2024-01-01 01:00:00+00:00", "2024-01-01 02:00:00+00:00"], utc=True
                ),
                "close_time": pd.to_datetime(
                    ["2024-01-01 01:59:59.999+00:00", "2024-01-01 02:59:59.999+00:00"], utc=True
                ),
                "open": [111.0, 102.0],  # updated duplicate row at 01:00
                "high": [112.0, 103.0],
                "low": [110.0, 101.0],
                "close": [111.5, 102.5],
                "volume": [2.0, 1.0],
                "quote_asset_volume": [2.0, 1.0],
                "num_trades": [2, 1],
                "taker_buy_base_asset_volume": [2.0, 1.0],
                "taker_buy_quote_asset_volume": [2.0, 1.0],
            }
        )

        merged, dup_removed, missing = merge_and_clean_candles(existing, incoming, "1h")
        self.assertEqual(dup_removed, 1)
        self.assertEqual(len(merged), 3)
        self.assertAlmostEqual(float(merged.loc[merged["open_time"] == pd.Timestamp("2024-01-01 01:00:00+00:00"), "open"].iloc[0]), 111.0)
        self.assertEqual(missing, 0)

    def test_incremental_fetch_ranges(self) -> None:
        cache_df = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    ["2024-01-01 00:00:00+00:00", "2024-01-01 01:00:00+00:00"], utc=True
                )
            }
        )
        start_utc = pd.Timestamp("2023-12-31 22:00:00+00:00")
        end_utc = pd.Timestamp("2024-01-01 04:00:00+00:00")
        ranges = build_incremental_fetch_ranges(
            cache_df=cache_df,
            start_utc=start_utc,
            end_utc=end_utc,
            interval="1h",
        )
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0][0], start_utc)
        self.assertEqual(ranges[0][1], pd.Timestamp("2024-01-01 00:00:00+00:00"))
        self.assertEqual(ranges[1][0], pd.Timestamp("2024-01-01 02:00:00+00:00"))
        self.assertEqual(ranges[1][1], end_utc)


if __name__ == "__main__":
    unittest.main()
