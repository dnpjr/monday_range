from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.features import add_monday_range
from src.backtest import backtest_sweep_fade
from src.metrics import trade_metrics


def _stop_scenario_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00",  # Monday
            "2024-01-01 04:00:00",  # Monday
            "2024-01-02 00:00:00",  # Tuesday sweep signal bar
            "2024-01-02 04:00:00",  # Tuesday entry bar
            "2024-01-02 08:00:00",  # Tuesday stop bar
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 100.0, 95.0],
            "high": [110.0, 108.0, 96.0, 99.0, 80.0],
            "low": [90.0, 92.0, 89.0, 95.0, 69.0],
            "close": [100.0, 101.0, 91.0, 96.0, 75.0],
        },
        index=idx,
    )


class AccountingTests(unittest.TestCase):
    def test_no_fee_no_slippage_stop_is_minus_risk(self) -> None:
        df = add_monday_range(_stop_scenario_df())
        df_out, trades = backtest_sweep_fade(
            df,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_at_mid=False,
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertAlmostEqual(float(trade["pnl"]), -100.0, places=8)
        self.assertAlmostEqual(float(df_out["equity"].dropna().iloc[-1]), 9_900.0, places=8)

    def test_fee_and_slippage_adjust_net_pnl(self) -> None:
        fee_bps = 10.0
        slippage_bps = 10.0
        fee_rate = fee_bps / 10_000.0
        slip_rate = slippage_bps / 10_000.0

        df = add_monday_range(_stop_scenario_df())
        df_out, trades = backtest_sweep_fade(
            df,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_at_mid=False,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]

        # Reproduce the internal accounting exactly for this scenario.
        entry_raw = 100.0
        stop_raw = 70.0
        entry = entry_raw * (1.0 + slip_rate)  # long entry adverse slippage
        qty = 100.0 / (entry - stop_raw)
        entry_fee = entry * qty * fee_rate
        exit_px = stop_raw * (1.0 - slip_rate)  # long exit adverse slippage
        gross = (exit_px - entry) * qty
        exit_fee = exit_px * qty * fee_rate
        expected_pnl = -entry_fee + gross - exit_fee
        expected_ret = expected_pnl / (entry * qty)

        self.assertAlmostEqual(float(trade["pnl"]), expected_pnl, places=8)
        self.assertAlmostEqual(float(trade["return_pct"]), expected_ret, places=8)
        self.assertAlmostEqual(float(df_out["equity"].dropna().iloc[-1]), 10_000.0 + expected_pnl, places=8)

        tm = trade_metrics(trades)
        self.assertAlmostEqual(float(tm["avg_trade_return"]), expected_ret, places=8)


if __name__ == "__main__":
    unittest.main()
