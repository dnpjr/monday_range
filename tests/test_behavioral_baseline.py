from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.backtest import backtest_sweep_fade
from src.execution_helpers import apply_slippage, calculate_entry_fee, calculate_exit_fee, calculate_tp_levels
from src.features import add_monday_range
from src.signals import sweep_rejection_signal


def _representative_week() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-02 00:00:00+00:00",  # both-sided sweep; LONG has precedence
            "2024-01-02 01:00:00+00:00",  # entry at this bar's open
            "2024-01-03 00:00:00+00:00",  # TP1 only
            "2024-01-05 23:00:00+00:00",  # Friday close of remainder
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 95.0, 98.0, 101.0],
            "high": [110.0, 108.0, 112.0, 99.0, 101.0, 103.0],
            "low": [90.0, 92.0, 89.0, 94.0, 96.0, 100.5],
            "close": [100.0, 101.0, 95.0, 96.0, 99.0, 102.0],
        },
        index=index,
    )


class BehavioralBaselineTests(unittest.TestCase):
    def test_representative_strategy_and_execution_contract(self) -> None:
        fee_bps = 10.0
        slippage_bps = 5.0
        featured = add_monday_range(_representative_week())

        signal_row = featured.iloc[2]
        self.assertEqual(
            sweep_rejection_signal(signal_row, mon_low=90.0, mon_high=110.0),
            "LONG",
        )

        _, trades = backtest_sweep_fade(
            featured,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mode="opposite_boundary",
            stop_mult=1.0,
            tp1_to_mid=1.0,
            tp1_close_fraction=0.5,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        entry_raw = 95.0
        entry = apply_slippage(entry_raw, "LONG", action="entry", slippage_bps=slippage_bps)
        stop = 90.0 - (1.0 * 20.0)
        tp1, tp2 = calculate_tp_levels(
            "LONG", mon_low=90.0, mon_mid=100.0, mon_high=110.0, tp1_to_mid=1.0, tp2_to_full=1.0
        )
        initial_qty = 100.0 / abs(entry - stop)
        partial_qty = initial_qty * 0.5
        remaining_qty = initial_qty - partial_qty
        tp1_fill = apply_slippage(tp1, "LONG", action="exit", slippage_bps=slippage_bps)
        friday_fill = apply_slippage(102.0, "LONG", action="exit", slippage_bps=slippage_bps)
        fees = (
            calculate_entry_fee(entry, initial_qty, fee_bps=fee_bps)
            + calculate_exit_fee(tp1_fill, partial_qty, fee_bps=fee_bps)
            + calculate_exit_fee(friday_fill, remaining_qty, fee_bps=fee_bps)
        )
        gross = ((tp1_fill - entry) * partial_qty) + ((friday_fill - entry) * remaining_qty)

        self.assertEqual(trade["entry_time"], featured.index[3])
        self.assertEqual(trade["exit_time"], featured.index[5])
        self.assertEqual(trade["side"], "LONG")
        self.assertEqual(trade["reason"], "FRIDAY")
        self.assertAlmostEqual(float(trade["entry"]), entry)
        self.assertAlmostEqual(float(trade["exit"]), friday_fill)
        self.assertAlmostEqual(float(trade["qty"]), remaining_qty)
        self.assertAlmostEqual(float(trade["gross_pnl"]), gross)
        self.assertAlmostEqual(float(trade["fees_paid"]), fees)
        self.assertAlmostEqual(float(trade["net_pnl"]), gross - fees)
        self.assertAlmostEqual(stop, 70.0)
        self.assertAlmostEqual(tp1, 100.0)
        self.assertAlmostEqual(tp2, 110.0)


if __name__ == "__main__":
    unittest.main()
