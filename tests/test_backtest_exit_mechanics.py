from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.backtest import backtest_sweep_fade
from src.features import add_monday_range


def _tp1_then_stop_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-02 00:00:00+00:00",  # signal candle
            "2024-01-03 00:00:00+00:00",  # entry candle
            "2024-01-03 01:00:00+00:00",  # no event
            "2024-01-03 02:00:00+00:00",  # stop-reversal
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 100.0, 96.0, 96.0],
            "high": [110.0, 108.0, 96.0, 96.0, 96.0, 96.0],
            "low": [90.0, 92.0, 89.0, 95.0, 95.0, 69.0],
            "close": [100.0, 101.0, 91.0, 96.0, 96.0, 70.0],
        },
        index=idx,
    )


def _stop_mode_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-02 00:00:00+00:00",  # signal candle
            "2024-01-03 00:00:00+00:00",  # entry candle
            "2024-01-03 01:00:00+00:00",  # 80 low (hits tight stops only)
            "2024-01-05 23:00:00+00:00",  # Friday close
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 100.0, 94.0, 82.0],
            "high": [110.0, 108.0, 96.0, 94.0, 95.0, 83.0],
            "low": [90.0, 92.0, 89.0, 93.0, 80.0, 81.0],
            "close": [100.0, 101.0, 91.0, 94.0, 82.0, 82.0],
        },
        index=idx,
    )


def _breakeven_fee_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-02 00:00:00+00:00",  # signal candle
            "2024-01-03 00:00:00+00:00",  # entry + TP1 hit when tp1_to_mid=0.5
            "2024-01-03 01:00:00+00:00",  # low just above entry
            "2024-01-05 23:00:00+00:00",  # Friday close fallback
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 100.0, 100.5, 100.5],
            "high": [110.0, 108.0, 96.0, 96.0, 101.0, 101.0],
            "low": [90.0, 92.0, 89.0, 95.0, 100.05, 100.0],
            "close": [100.0, 101.0, 91.0, 96.0, 100.5, 100.5],
        },
        index=idx,
    )


class BacktestExitMechanicsTests(unittest.TestCase):
    def test_defaults_match_legacy_equivalent(self) -> None:
        df = add_monday_range(_tp1_then_stop_df())
        _, t_default = backtest_sweep_fade(df, initial_capital=10_000.0, risk_per_trade=100.0)
        _, t_legacy = backtest_sweep_fade(
            df,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_at_mid=True,
            tp1_frac=0.5,
            tp1_to_mid=1.0,
            tp1_close_fraction=0.5,
            stop_mode="opposite_boundary",
            move_stop_to_breakeven_after_tp1=True,
            breakeven_includes_fees=False,
            single_target_mode=False,
        )
        self.assertEqual(len(t_default), len(t_legacy))
        self.assertAlmostEqual(float(t_default.iloc[0]["pnl"]), float(t_legacy.iloc[0]["pnl"]), places=10)
        self.assertEqual(t_default.iloc[0]["reason"], t_legacy.iloc[0]["reason"])

    def test_tp1_to_mid_changes_behavior(self) -> None:
        df = add_monday_range(_tp1_then_stop_df())
        _, t_no_tp1 = backtest_sweep_fade(df, tp1_to_mid=1.0, initial_capital=10_000.0, risk_per_trade=100.0)
        _, t_with_tp1 = backtest_sweep_fade(df, tp1_to_mid=0.5, initial_capital=10_000.0, risk_per_trade=100.0)
        self.assertEqual(len(t_no_tp1), 1)
        self.assertEqual(len(t_with_tp1), 1)
        self.assertGreater(float(t_with_tp1.iloc[0]["pnl"]), float(t_no_tp1.iloc[0]["pnl"]))

    def test_tp1_close_fraction_changes_behavior(self) -> None:
        df = add_monday_range(_tp1_then_stop_df())
        _, t_small = backtest_sweep_fade(
            df,
            tp1_to_mid=0.5,
            tp1_close_fraction=0.25,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        _, t_large = backtest_sweep_fade(
            df,
            tp1_to_mid=0.5,
            tp1_close_fraction=0.75,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        # The entry bar opens beyond TP1, so gap-aware execution fills the partial
        # at the open. A larger partial leaves less quantity for the later adverse gap.
        self.assertGreater(float(t_large.iloc[0]["pnl"]), float(t_small.iloc[0]["pnl"]))

    def test_stop_mode_range_fraction(self) -> None:
        df = add_monday_range(_stop_mode_df())
        _, t_default = backtest_sweep_fade(df, initial_capital=10_000.0, risk_per_trade=100.0)
        _, t_range = backtest_sweep_fade(
            df,
            stop_mode="range_fraction",
            stop_range_fraction=0.25,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        self.assertEqual(t_default.iloc[0]["reason"], "FRIDAY")
        self.assertEqual(t_range.iloc[0]["reason"], "STOP")

    def test_stop_mode_fixed_pct(self) -> None:
        df = add_monday_range(_stop_mode_df())
        _, t_fixed = backtest_sweep_fade(
            df,
            stop_mode="fixed_pct",
            stop_pct=0.02,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        self.assertEqual(t_fixed.iloc[0]["reason"], "STOP")

    def test_move_stop_to_breakeven_after_tp1(self) -> None:
        df = add_monday_range(_tp1_then_stop_df())
        _, t_move = backtest_sweep_fade(
            df,
            tp1_to_mid=0.5,
            move_stop_to_breakeven_after_tp1=True,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        _, t_no_move = backtest_sweep_fade(
            df,
            tp1_to_mid=0.5,
            move_stop_to_breakeven_after_tp1=False,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        self.assertGreater(float(t_move.iloc[0]["pnl"]), float(t_no_move.iloc[0]["pnl"]))

    def test_breakeven_includes_fees(self) -> None:
        df = add_monday_range(_breakeven_fee_df())
        _, t_no_fee_be = backtest_sweep_fade(
            df,
            tp1_to_mid=0.5,
            fee_bps=100.0,
            move_stop_to_breakeven_after_tp1=True,
            breakeven_includes_fees=False,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        _, t_with_fee_be = backtest_sweep_fade(
            df,
            tp1_to_mid=0.5,
            fee_bps=100.0,
            move_stop_to_breakeven_after_tp1=True,
            breakeven_includes_fees=True,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        self.assertNotEqual(str(t_no_fee_be.iloc[0]["exit_time"]), str(t_with_fee_be.iloc[0]["exit_time"]))

    def test_single_target_mode(self) -> None:
        df = add_monday_range(_tp1_then_stop_df())
        _, t_partial = backtest_sweep_fade(
            df,
            tp1_to_mid=0.5,
            single_target_mode=False,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        _, t_single = backtest_sweep_fade(
            df,
            tp1_to_mid=0.5,
            single_target_mode=True,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
        )
        self.assertLess(float(t_single.iloc[0]["pnl"]), float(t_partial.iloc[0]["pnl"]))


if __name__ == "__main__":
    unittest.main()
