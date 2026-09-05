from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.timeframe_utils import (
    timeframe_to_minutes,
    can_resample,
    compare_timeframes,
    resample_ohlcv,
)


class TimeframeUtilsTests(unittest.TestCase):
    def test_timeframe_parsing(self) -> None:
        self.assertEqual(timeframe_to_minutes("15m"), 15)
        self.assertEqual(timeframe_to_minutes("1h"), 60)
        self.assertEqual(timeframe_to_minutes("2h"), 120)
        self.assertEqual(timeframe_to_minutes("1d"), 1440)
        with self.assertRaises(ValueError):
            timeframe_to_minutes("bad")

    def test_compare_and_resample_permissions(self) -> None:
        self.assertEqual(compare_timeframes("1h", "4h"), -1)
        self.assertEqual(compare_timeframes("4h", "1h"), 1)
        self.assertTrue(can_resample("1h", "2h"))
        self.assertTrue(can_resample("1h", "1d"))
        self.assertFalse(can_resample("1h", "15m"))
        self.assertFalse(can_resample("4h", "1h"))
        self.assertFalse(can_resample("1d", "4h"))

    def test_resample_1h_to_4h_aggregation(self) -> None:
        idx = pd.date_range("2024-01-01 00:00:00+00:00", periods=4, freq="1h")
        df = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 13.0],
                "high": [11.0, 12.0, 14.0, 13.5],
                "low": [9.0, 10.0, 11.0, 12.0],
                "close": [10.5, 11.5, 13.5, 13.0],
                "volume": [1.0, 2.0, 3.0, 4.0],
            },
            index=idx,
        )
        out = resample_ohlcv(df, base_interval="1h", target_interval="4h")
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertAlmostEqual(float(row["open"]), 10.0, places=10)
        self.assertAlmostEqual(float(row["high"]), 14.0, places=10)
        self.assertAlmostEqual(float(row["low"]), 9.0, places=10)
        self.assertAlmostEqual(float(row["close"]), 13.0, places=10)
        self.assertAlmostEqual(float(row["volume"]), 10.0, places=10)

    def test_resample_1h_to_1d_aggregation(self) -> None:
        idx = pd.date_range("2024-01-01 00:00:00+00:00", periods=24, freq="1h")
        df = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(24)],
                "high": [101.0 + i for i in range(24)],
                "low": [99.0 + i for i in range(24)],
                "close": [100.5 + i for i in range(24)],
                "volume": [1.0] * 24,
            },
            index=idx,
        )
        out = resample_ohlcv(df, base_interval="1h", target_interval="1d")
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertAlmostEqual(float(row["open"]), 100.0, places=10)
        self.assertAlmostEqual(float(row["high"]), 124.0, places=10)
        self.assertAlmostEqual(float(row["low"]), 99.0, places=10)
        self.assertAlmostEqual(float(row["close"]), 123.5, places=10)
        self.assertAlmostEqual(float(row["volume"]), 24.0, places=10)

    def test_invalid_downsample_request_raises(self) -> None:
        idx = pd.date_range("2024-01-01 00:00:00+00:00", periods=4, freq="1h")
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [2.0, 3.0, 4.0, 5.0],
                "low": [0.5, 1.5, 2.5, 3.5],
                "close": [1.5, 2.5, 3.5, 4.5],
                "volume": [1.0, 1.0, 1.0, 1.0],
            },
            index=idx,
        )
        with self.assertRaises(ValueError):
            resample_ohlcv(df, base_interval="1h", target_interval="15m")


if __name__ == "__main__":
    unittest.main()
