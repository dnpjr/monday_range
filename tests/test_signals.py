from __future__ import annotations

import unittest
import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.signals import sweep_rejection_signal


class SweepSignalTests(unittest.TestCase):
    def test_long_signal(self) -> None:
        prev = pd.Series({"low": 89.0, "high": 105.0, "close": 91.0})
        self.assertEqual(sweep_rejection_signal(prev, mon_low=90.0, mon_high=110.0), "LONG")

    def test_short_signal(self) -> None:
        prev = pd.Series({"low": 95.0, "high": 111.0, "close": 109.0})
        self.assertEqual(sweep_rejection_signal(prev, mon_low=90.0, mon_high=110.0), "SHORT")

    def test_no_signal(self) -> None:
        prev = pd.Series({"low": 95.0, "high": 105.0, "close": 100.0})
        self.assertIsNone(sweep_rejection_signal(prev, mon_low=90.0, mon_high=110.0))

    def test_long_precedence_when_both_true(self) -> None:
        prev = pd.Series({"low": 89.0, "high": 111.0, "close": 95.0})
        self.assertEqual(sweep_rejection_signal(prev, mon_low=90.0, mon_high=110.0), "LONG")


if __name__ == "__main__":
    unittest.main()
