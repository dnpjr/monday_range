from __future__ import annotations

import unittest
import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.features import add_monday_range
from src.backtest import backtest_sweep_fade
from src.research import analyze_weekly_sweep_signals


def _sample_week_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00",  # Monday
            "2024-01-01 04:00:00",  # Monday
            "2024-01-02 00:00:00",  # Tuesday sweep bar (both long+short true)
            "2024-01-02 04:00:00",  # Tuesday entry bar
            "2024-01-02 08:00:00",  # Tuesday stop-to-BE bar
        ]
    )

    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 100.0, 101.0],
            "high": [110.0, 108.0, 112.0, 105.0, 102.0],
            "low": [90.0, 92.0, 89.0, 99.0, 100.0],
            "close": [100.0, 101.0, 95.0, 102.0, 100.0],
        },
        index=idx,
    )


class StrategyRegressionTests(unittest.TestCase):
    def test_backtest_trade_shape_and_values(self) -> None:
        df = add_monday_range(_sample_week_df())
        df_out, trades = backtest_sweep_fade(
            df,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_frac=0.5,
        )

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade["side"], "LONG")
        self.assertEqual(trade["reason"], "STOP")
        self.assertAlmostEqual(float(trade["entry"]), 100.0)
        self.assertAlmostEqual(float(trade["exit"]), 100.0)
        self.assertAlmostEqual(float(trade["pnl"]), 0.0)
        self.assertAlmostEqual(float(df_out["equity"].dropna().iloc[-1]), 10_000.0)

    def test_research_long_precedence_when_both_true(self) -> None:
        df = add_monday_range(_sample_week_df())
        signals = analyze_weekly_sweep_signals(df)

        self.assertEqual(len(signals), 1)
        row = signals.iloc[0]
        self.assertEqual(row["side"], "LONG")
        self.assertAlmostEqual(float(row["entry_price"]), 100.0)


if __name__ == "__main__":
    unittest.main()
