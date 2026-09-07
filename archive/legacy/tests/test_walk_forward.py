from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from run_walk_forward import (
    build_rolling_year_folds,
    build_yearly_mode_folds,
    generate_sweep_retest_grid,
    run_walk_forward,
)


def _synthetic_ohlc(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="1D", tz="UTC")
    base = pd.Series(range(len(idx)), dtype="float64")
    df = pd.DataFrame(index=idx)
    df["open"] = 100.0 + (base % 25.0)
    df["high"] = df["open"] + 2.0
    df["low"] = df["open"] - 2.0
    df["close"] = df["open"] + ((base % 3.0) - 1.0)
    return df


class WalkForwardTests(unittest.TestCase):
    def test_build_yearly_mode_folds_default(self) -> None:
        folds = build_yearly_mode_folds("2026-05-20")
        self.assertEqual(len(folds), 4)
        self.assertEqual(folds[0]["train_start"], "2021-05-21")
        self.assertEqual(folds[0]["train_end"], "2022-12-31")
        self.assertEqual(folds[0]["test_start"], "2023-01-01")
        self.assertEqual(folds[0]["test_end"], "2023-12-31")
        self.assertEqual(folds[-1]["train_start"], "2024-01-01")
        self.assertEqual(folds[-1]["test_end"], "2026-05-20")

    def test_generate_sweep_retest_grid_defaults(self) -> None:
        grid = generate_sweep_retest_grid()
        self.assertEqual(len(grid), 5 * 4 * 4 * 1 * 2 * 2)
        self.assertIn("tp1_range_fraction", grid[0])
        self.assertIn("stop_range_fraction", grid[0])
        self.assertIn("tp1_close_fraction", grid[0])
        self.assertIn("tp2_range_fraction", grid[0])
        self.assertIn("friday_cutoff_hour_utc", grid[0])
        self.assertIn("exit_style", grid[0])

    def test_build_rolling_year_folds_train_test_separation(self) -> None:
        folds = build_rolling_year_folds(
            available_years=[2020, 2021, 2022, 2023, 2024],
            train_window=2,
            test_window=1,
        )
        self.assertEqual(len(folds), 3)
        for fold in folds:
            train_end = pd.Timestamp(fold["train_end"], tz="UTC")
            test_start = pd.Timestamp(fold["test_start"], tz="UTC")
            self.assertLess(train_end, test_start)

    def test_run_walk_forward_writes_outputs(self) -> None:
        ohlc = _synthetic_ohlc("2021-01-01", "2023-12-31")
        with tempfile.TemporaryDirectory() as tmp, patch("run_walk_forward.load_cached_ohlc", return_value=ohlc):
            result = run_walk_forward(
                strategy="sweep_retest",
                symbol="BTCUSDT",
                interval="1h",
                yearly_mode=False,
                train_window=1,
                test_window=1,
                fee_bps=10.0,
                slippage_bps=5.0,
                top_n=1,
                output_dir=tmp,
                risk_fraction=0.01,
                tp1_range_fractions=[0.5],
                tp2_range_fractions=[1.0],
                friday_cutoff_hours_utc=[23],
                exit_styles=["partial_tp2"],
                tp1_close_fractions=[0.5],
                stop_range_fractions=[1.0],
            )

            out_dir = pathlib.Path(result["out_dir"])
            self.assertTrue((out_dir / "walk_forward_results.csv").exists())
            self.assertTrue((out_dir / "fold_train_results.csv").exists())
            self.assertTrue((out_dir / "fold_test_results.csv").exists())
            self.assertTrue((out_dir / "config.json").exists())

            wf = pd.read_csv(out_dir / "walk_forward_results.csv")
            self.assertEqual(len(wf), 1)
            self.assertIn("percentage_positive_out_of_sample", wf.columns)
            self.assertIn("worst_out_of_sample_return_pct", wf.columns)
            self.assertEqual(result["config"]["dataset_version"], "btcusdt_binance_spot_1h_v1")
            self.assertEqual(len(result["config"]["dataset_sha256"]), 64)

    def test_run_walk_forward_calls_train_then_test_per_fold(self) -> None:
        ohlc = _synthetic_ohlc("2021-01-01", "2025-12-31")
        eval_calls: list[tuple[str, str]] = []

        train_row = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "start": "2021-01-01",
            "end": "2021-12-31",
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "risk_fraction": 0.01,
            "tp2_to_full": 0.5,
            "friday_cutoff_hour_utc": 23,
            "min_range_pct": None,
            "max_range_pct": None,
            "direction": "both",
            "max_entry_day_utc": None,
            "max_entry_hour_utc": None,
            "tp1_close_fraction": 0.5,
            "move_stop_to_breakeven_after_tp1": True,
            "stop_mode": "range_fraction",
            "stop_range_fraction": 1.0,
            "exit_style": "partial_tp2",
            "single_target_level": "tp2",
            "total_return_pct": 1.0,
            "return_after_costs_pct": 1.0,
            "max_drawdown": -0.1,
            "win_rate": 0.5,
            "num_trades": 2,
            "average_trade_pnl": 1.0,
            "best_trade": 2.0,
            "worst_trade": -1.0,
            "total_fees_paid": 2.0,
            "cost_drag_pct": 0.02,
            "score": 0.5,
            "train_rank": 1,
        }
        test_row = dict(train_row)
        test_row["total_return_pct"] = 0.5
        test_row["return_after_costs_pct"] = 0.5

        def fake_window_features(_ohlc, _start: str, _end: str) -> pd.DataFrame:
            idx = pd.date_range("2021-01-01", periods=3, freq="1D", tz="UTC")
            return pd.DataFrame({"open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0]}, index=idx)

        def fake_eval(**kwargs):
            start = str(kwargs["start"])
            end = str(kwargs["end"])
            eval_calls.append((start, end))
            if len(eval_calls) % 2 == 1:
                return pd.DataFrame([train_row])
            return pd.DataFrame([test_row])

        with tempfile.TemporaryDirectory() as tmp, patch("run_walk_forward.load_cached_ohlc", return_value=ohlc), patch(
            "run_walk_forward._window_features", side_effect=fake_window_features
        ), patch("run_walk_forward.generate_sweep_retest_grid", return_value=[{"risk_fraction": 0.01}]), patch(
            "run_walk_forward._evaluate_grid_on_features", side_effect=fake_eval
        ):
            run_walk_forward(
                strategy="sweep_retest",
                symbol="BTCUSDT",
                interval="1h",
                yearly_mode=True,
                fee_bps=10.0,
                slippage_bps=5.0,
                top_n=1,
                output_dir=tmp,
            )

        self.assertEqual(len(eval_calls), 6)
        for i in range(0, len(eval_calls), 2):
            train_start, train_end = eval_calls[i]
            test_start, test_end = eval_calls[i + 1]
            self.assertLess(pd.Timestamp(train_end, tz="UTC"), pd.Timestamp(test_start, tz="UTC"))
            self.assertLess(pd.Timestamp(train_start, tz="UTC"), pd.Timestamp(train_end, tz="UTC"))
            self.assertLess(pd.Timestamp(test_start, tz="UTC"), pd.Timestamp(test_end, tz="UTC"))


if __name__ == "__main__":
    unittest.main()
