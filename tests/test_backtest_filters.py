from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.backtest import backtest_sweep_fade
from src.features import add_monday_range


def _long_signal_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-02 00:00:00+00:00",  # signal candle
            "2024-01-02 01:00:00+00:00",  # entry candle
            "2024-01-02 02:00:00+00:00",  # stop candle
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 100.0, 100.0],
            "high": [110.0, 108.0, 96.0, 101.0, 100.0],
            "low": [90.0, 92.0, 89.0, 99.0, 69.0],
            "close": [100.0, 101.0, 91.0, 100.0, 75.0],
        },
        index=idx,
    )


def _short_signal_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-02 00:00:00+00:00",  # signal candle
            "2024-01-02 01:00:00+00:00",  # entry candle
            "2024-01-02 02:00:00+00:00",  # tp candle
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 105.0, 105.0, 105.0],
            "high": [110.0, 108.0, 111.0, 106.0, 106.0],
            "low": [90.0, 92.0, 104.0, 104.0, 89.0],
            "close": [100.0, 101.0, 109.0, 105.0, 90.0],
        },
        index=idx,
    )


def _thur_entry_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-03 23:00:00+00:00",  # signal candle (Wednesday)
            "2024-01-04 00:00:00+00:00",  # entry candle (Thursday)
            "2024-01-04 01:00:00+00:00",  # stop candle
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 100.0, 100.0],
            "high": [110.0, 108.0, 96.0, 101.0, 100.0],
            "low": [90.0, 92.0, 89.0, 99.0, 69.0],
            "close": [100.0, 101.0, 91.0, 100.0, 75.0],
        },
        index=idx,
    )


class BacktestFilterTests(unittest.TestCase):
    def test_default_filters_preserve_behavior(self) -> None:
        df = add_monday_range(_long_signal_df())
        _, trades_default = backtest_sweep_fade(df, initial_capital=10_000.0, risk_per_trade=100.0)
        _, trades_explicit = backtest_sweep_fade(
            df,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            min_range_pct=None,
            max_range_pct=None,
            direction="both",
            max_entry_day_utc=None,
            max_entry_hour_utc=None,
            sma_period=None,
        )
        self.assertEqual(len(trades_default), len(trades_explicit))
        self.assertEqual(trades_default.iloc[0]["side"], trades_explicit.iloc[0]["side"])
        self.assertEqual(trades_default.iloc[0]["reason"], trades_explicit.iloc[0]["reason"])

    def test_range_filter_blocks_trade(self) -> None:
        df = add_monday_range(_long_signal_df())
        _, trades_min = backtest_sweep_fade(df, min_range_pct=0.25)
        _, trades_max = backtest_sweep_fade(df, max_range_pct=0.15)
        self.assertEqual(len(trades_min), 0)
        self.assertEqual(len(trades_max), 0)

    def test_direction_filter(self) -> None:
        df_long = add_monday_range(_long_signal_df())
        _, trades_long_only = backtest_sweep_fade(df_long, direction="long_only")
        _, trades_short_only = backtest_sweep_fade(df_long, direction="short_only")
        self.assertEqual(len(trades_long_only), 1)
        self.assertEqual(trades_long_only.iloc[0]["side"], "LONG")
        self.assertEqual(len(trades_short_only), 0)

        df_short = add_monday_range(_short_signal_df())
        _, trades_short_only_pass = backtest_sweep_fade(df_short, direction="short_only")
        _, trades_long_only_block = backtest_sweep_fade(df_short, direction="long_only")
        self.assertEqual(len(trades_short_only_pass), 1)
        self.assertEqual(trades_short_only_pass.iloc[0]["side"], "SHORT")
        self.assertEqual(len(trades_long_only_block), 0)

    def test_max_entry_time_filter(self) -> None:
        df = add_monday_range(_thur_entry_df())
        _, trades_no_cutoff = backtest_sweep_fade(df)
        _, trades_cutoff = backtest_sweep_fade(df, max_entry_day_utc=2, max_entry_hour_utc=23)
        self.assertEqual(len(trades_no_cutoff), 1)
        self.assertEqual(len(trades_cutoff), 0)

    def test_sma_filter(self) -> None:
        df = add_monday_range(_long_signal_df())
        _, trades_no_sma = backtest_sweep_fade(df)
        _, trades_with_sma = backtest_sweep_fade(df, sma_period=3)
        self.assertEqual(len(trades_no_sma), 1)
        self.assertEqual(len(trades_with_sma), 0)


if __name__ == "__main__":
    unittest.main()
