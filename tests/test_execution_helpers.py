from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.execution_helpers import (
    apply_slippage,
    calculate_entry_fee,
    calculate_exit_fee,
    calculate_tp_levels,
    is_stop_hit,
    is_tp_hit,
    is_friday_cutoff_bar,
)


class ExecutionHelperTests(unittest.TestCase):
    def test_apply_slippage_entry_and_exit(self) -> None:
        self.assertAlmostEqual(apply_slippage(100.0, "LONG", "entry", 10.0), 100.1, places=10)
        self.assertAlmostEqual(apply_slippage(100.0, "SHORT", "entry", 10.0), 99.9, places=10)
        self.assertAlmostEqual(apply_slippage(100.0, "LONG", "exit", 10.0), 99.9, places=10)
        self.assertAlmostEqual(apply_slippage(100.0, "SHORT", "exit", 10.0), 100.1, places=10)
        self.assertAlmostEqual(apply_slippage(100.0, "LONG", "entry", 0.0), 100.0, places=10)

    def test_fee_calculations(self) -> None:
        self.assertAlmostEqual(calculate_entry_fee(100.0, 2.0, 10.0), 0.2, places=10)
        self.assertAlmostEqual(calculate_exit_fee(105.0, 2.0, 10.0), 0.21, places=10)

    def test_calculate_tp_levels(self) -> None:
        # New range-fraction semantics.
        tp1, tp2 = calculate_tp_levels("LONG", mon_low=100.0, mon_mid=105.0, mon_high=110.0, tp1_range_fraction=0.5, tp2_range_fraction=1.0)
        self.assertAlmostEqual(tp1, 105.0, places=10)
        self.assertAlmostEqual(tp2, 110.0, places=10)

        tp1, tp2 = calculate_tp_levels("SHORT", mon_low=100.0, mon_mid=105.0, mon_high=110.0, tp1_range_fraction=0.5, tp2_range_fraction=1.0)
        self.assertAlmostEqual(tp1, 105.0, places=10)
        self.assertAlmostEqual(tp2, 100.0, places=10)

        tp1, _ = calculate_tp_levels("LONG", mon_low=100.0, mon_mid=105.0, mon_high=110.0, tp1_range_fraction=0.25, tp2_range_fraction=1.0)
        self.assertAlmostEqual(tp1, 102.5, places=10)
        tp1, _ = calculate_tp_levels("SHORT", mon_low=100.0, mon_mid=105.0, mon_high=110.0, tp1_range_fraction=0.25, tp2_range_fraction=1.0)
        self.assertAlmostEqual(tp1, 107.5, places=10)

    def test_calculate_tp_levels_legacy_alias_mapping(self) -> None:
        # Legacy defaults map to midpoint/full boundary.
        tp1_legacy, tp2_legacy = calculate_tp_levels("LONG", mon_low=90.0, mon_mid=100.0, mon_high=110.0, tp2_to_full=1.0, tp1_to_mid=1.0)
        tp1_new, tp2_new = calculate_tp_levels("LONG", mon_low=90.0, mon_mid=100.0, mon_high=110.0, tp1_range_fraction=0.5, tp2_range_fraction=1.0)
        self.assertAlmostEqual(tp1_legacy, tp1_new, places=10)
        self.assertAlmostEqual(tp2_legacy, tp2_new, places=10)

    def test_stop_and_tp_hits(self) -> None:
        self.assertTrue(is_stop_hit("LONG", candle_high=101.0, candle_low=89.0, stop_price=90.0))
        self.assertFalse(is_stop_hit("LONG", candle_high=101.0, candle_low=91.0, stop_price=90.0))
        self.assertTrue(is_stop_hit("SHORT", candle_high=111.0, candle_low=100.0, stop_price=110.0))

        self.assertTrue(is_tp_hit("LONG", candle_high=106.0, candle_low=99.0, tp_price=105.0))
        self.assertFalse(is_tp_hit("LONG", candle_high=104.0, candle_low=99.0, tp_price=105.0))
        self.assertTrue(is_tp_hit("SHORT", candle_high=106.0, candle_low=94.0, tp_price=95.0))

    def test_is_friday_cutoff_bar(self) -> None:
        bar_open = pd.Timestamp("2024-01-05 22:00:00+00:00")
        bar_close = pd.Timestamp("2024-01-05 23:00:00+00:00")
        self.assertFalse(is_friday_cutoff_bar(bar_open, bar_close, friday_cutoff_hour_utc=23))

        bar_open = pd.Timestamp("2024-01-05 23:00:00+00:00")
        bar_close = pd.Timestamp("2024-01-06 00:00:00+00:00")
        self.assertTrue(is_friday_cutoff_bar(bar_open, bar_close, friday_cutoff_hour_utc=23))


if __name__ == "__main__":
    unittest.main()
