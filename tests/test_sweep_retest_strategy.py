from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.features import add_monday_range
from src.backtest import backtest_sweep_fade
from src.research import analyze_weekly_sweep_signals


def _two_week_signal_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            # Week 1 Monday
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            # Week 1 Tuesday signal/entry/path
            "2024-01-02 00:00:00+00:00",  # signal candle (LONG)
            "2024-01-02 01:00:00+00:00",  # entry long
            "2024-01-02 02:00:00+00:00",
            "2024-01-02 03:00:00+00:00",  # hits TP1(mid)
            # Week 2 Monday
            "2024-01-08 00:00:00+00:00",
            "2024-01-08 01:00:00+00:00",
            # Week 2 Tuesday signal/entry/path
            "2024-01-09 00:00:00+00:00",  # signal candle (SHORT)
            "2024-01-09 01:00:00+00:00",  # entry short
            "2024-01-09 02:00:00+00:00",
            "2024-01-09 03:00:00+00:00",  # hits TP1(mid)
        ]
    )
    return pd.DataFrame(
        {
            "open": [110, 112, 95, 100, 102, 104, 210, 212, 225, 220, 218, 216],
            "high": [130, 128, 100, 101, 107, 111, 230, 228, 231, 221, 219, 217],
            "low": [90, 92, 89, 99, 100, 103, 190, 192, 220, 219, 213, 209],
            "close": [110, 111, 95, 100, 106, 110, 210, 211, 225, 220, 214, 210],
        },
        index=idx,
    )


def _no_signal_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-02 00:00:00+00:00",
            "2024-01-02 01:00:00+00:00",
            "2024-01-03 00:00:00+00:00",
        ]
    )
    return pd.DataFrame(
        {
            "open": [100, 100, 101, 101, 102],
            "high": [110, 109, 108, 109, 110],
            "low": [90, 91, 92, 93, 94],
            "close": [100, 100, 101, 102, 103],
        },
        index=idx,
    )


def _stop_hit_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-02 00:00:00+00:00",  # signal candle (LONG)
            "2024-01-02 01:00:00+00:00",  # entry candle
            "2024-01-02 02:00:00+00:00",  # stop hit
        ]
    )
    return pd.DataFrame(
        {
            "open": [110, 112, 95, 100, 100],
            "high": [130, 128, 100, 101, 101],
            "low": [90, 92, 89, 99, 69],  # long stop at 70 when stop_range_fraction=0.5
            "close": [110, 111, 95, 100, 70],
        },
        index=idx,
    )


class SweepRetestStrategyTests(unittest.TestCase):
    def test_signal_count_matches_legacy_and_uses_first_signal_per_week(self) -> None:
        df = add_monday_range(_two_week_signal_df())
        legacy_signals = analyze_weekly_sweep_signals(df)
        df_out, trades = backtest_sweep_fade(
            df,
            strategy="sweep_retest",
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mode="range_fraction",
            stop_range_fraction=0.5,
            single_target_mode=True,
            single_target_level="tp1",
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        self.assertEqual(int(df_out.attrs.get("signal_count", -1)), len(legacy_signals))
        self.assertEqual(len(trades), len(legacy_signals))
        # One trade per signaled week; does not emit both long+short candidates per week.
        self.assertEqual(len(trades), 2)

    def test_uses_only_actual_legacy_signals(self) -> None:
        df = add_monday_range(_no_signal_df())
        legacy_signals = analyze_weekly_sweep_signals(df)
        df_out, trades = backtest_sweep_fade(
            df,
            strategy="sweep_retest",
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mode="range_fraction",
            stop_range_fraction=0.5,
        )
        self.assertEqual(len(legacy_signals), 0)
        self.assertEqual(int(df_out.attrs.get("signal_count", -1)), 0)
        self.assertEqual(len(trades), 0)

    def test_entry_timing_matches_legacy_next_bar_open(self) -> None:
        df = add_monday_range(_two_week_signal_df())
        legacy_signals = analyze_weekly_sweep_signals(df)
        _, trades = backtest_sweep_fade(
            df,
            strategy="sweep_retest",
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mode="range_fraction",
            stop_range_fraction=0.5,
            single_target_mode=True,
            single_target_level="tp1",
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        self.assertEqual(len(trades), len(legacy_signals))
        for (_, legacy_row), (_, trade_row) in zip(legacy_signals.iterrows(), trades.iterrows()):
            self.assertEqual(pd.Timestamp(legacy_row["entry_time"]), pd.Timestamp(trade_row["entry_time"]))
            self.assertAlmostEqual(float(legacy_row["entry_price"]), float(trade_row["entry"]), places=10)

    def test_single_target_tp1_midpoint_semantics(self) -> None:
        df = add_monday_range(_two_week_signal_df())
        _, trades = backtest_sweep_fade(
            df,
            strategy="sweep_retest",
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mode="range_fraction",
            stop_range_fraction=0.5,
            single_target_mode=True,
            single_target_level="tp1",
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        first = trades.iloc[0]
        self.assertEqual(str(first["reason"]), "TP1")
        self.assertAlmostEqual(float(first["exit"]), 110.0, places=10)  # Week-1 midpoint

    def test_range_fraction_stop_handling(self) -> None:
        df = add_monday_range(_stop_hit_df())
        _, trades = backtest_sweep_fade(
            df,
            strategy="sweep_retest",
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mode="range_fraction",
            stop_range_fraction=0.5,
            single_target_mode=True,
            single_target_level="tp2",
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(str(trades.iloc[0]["reason"]), "STOP")
        self.assertAlmostEqual(float(trades.iloc[0]["exit"]), 70.0, places=10)


if __name__ == "__main__":
    unittest.main()
