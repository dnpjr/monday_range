from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from run_parameter_sweep import (
    build_sweep_row,
    generate_grid,
    parse_float_grid,
    parse_int_grid,
    parse_optional_float_grid,
    parse_optional_int_grid,
    parse_direction_grid,
    run_sweep,
    write_sweep_outputs,
)


class ParameterSweepTests(unittest.TestCase):
    def test_grid_generation(self) -> None:
        risk_fractions = [0.005, 0.01]
        tp2_vals = [0.5, 1.0]
        cutoff_hours = [22, 23]
        grid = generate_grid(
            risk_fractions=risk_fractions,
            tp2_to_full_values=tp2_vals,
            friday_cutoff_hours_utc=cutoff_hours,
        )
        self.assertEqual(len(grid), 8)
        self.assertEqual(grid[0]["risk_fraction"], 0.005)
        self.assertEqual(grid[-1]["friday_cutoff_hour_utc"], 23)

    def test_summary_row_construction(self) -> None:
        summary = {
            "total_return_pct": 12.5,
            "max_drawdown": -0.2,
            "win_rate": 0.6,
            "num_trades": 7,
            "average_trade_pnl": 15.0,
            "best_trade": 50.0,
            "worst_trade": -20.0,
            "total_fees_paid": 9.5,
            "start_used": "2024-01-01",
            "end_used": "2024-01-31",
        }
        row = build_sweep_row(
            summary=summary,
            symbol="BTCUSDT",
            interval="1h",
            start=None,
            end=None,
            risk_fraction=0.01,
            tp1_range_fraction=0.5,
            tp2_range_fraction=1.0,
            tp2_to_full=0.75,
            friday_cutoff_hour_utc=23,
            fee_bps=10.0,
            slippage_bps=5.0,
            initial_cash=10_000.0,
            min_range_pct=None,
            max_range_pct=None,
            direction="both",
            max_entry_day_utc=None,
            max_entry_hour_utc=None,
        )
        self.assertEqual(row["symbol"], "BTCUSDT")
        self.assertEqual(row["fee_bps"], 10.0)
        self.assertEqual(row["slippage_bps"], 5.0)
        self.assertEqual(row["risk_fraction"], 0.01)
        self.assertEqual(row["tp1_range_fraction"], 0.5)
        self.assertEqual(row["tp2_range_fraction"], 1.0)
        self.assertEqual(row["tp2_to_full"], 0.75)
        self.assertEqual(row["friday_cutoff_hour_utc"], 23)
        self.assertAlmostEqual(row["cost_drag_pct"], 0.095)
        self.assertEqual(row["return_after_costs_pct"], 12.5)
        self.assertIn("total_fees_paid", row)
        self.assertIn("score", row)
        self.assertGreater(row["score"], -1000)

    def test_write_outputs(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "start": "2024-01-01",
                    "end": "2024-01-31",
                    "fee_bps": 10.0,
                    "slippage_bps": 5.0,
                    "risk_fraction": 0.01,
                    "tp2_to_full": 1.0,
                    "friday_cutoff_hour_utc": 23,
                    "total_return_pct": 1.23,
                    "return_after_costs_pct": 1.23,
                    "max_drawdown": -0.1,
                    "win_rate": 0.5,
                    "num_trades": 10,
                    "average_trade_pnl": 3.0,
                    "best_trade": 5.0,
                    "worst_trade": -2.0,
                    "total_fees_paid": 1.0,
                    "cost_drag_pct": 0.01,
                    "score": 0.0,
                }
            ]
        )
        config = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "risk_fractions": [0.01],
            "tp2_to_full_values": [1.0],
            "friday_cutoff_hours_utc": [23],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = write_sweep_outputs(
                output_root=tmp,
                symbol="BTCUSDT",
                interval="1h",
                results=results,
                sweep_config=config,
            )
            csv_path = out_dir / "sweep_results.csv"
            cfg_path = out_dir / "sweep_config.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(cfg_path.exists())

            loaded_results = pd.read_csv(csv_path)
            self.assertEqual(len(loaded_results), 1)
            self.assertEqual(loaded_results.iloc[0]["symbol"], "BTCUSDT")

            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded_cfg = json.load(f)
            self.assertEqual(loaded_cfg["interval"], "1h")

    def test_grid_parser(self) -> None:
        self.assertEqual(parse_float_grid("0.5, 1.0", name="x"), [0.5, 1.0])
        self.assertEqual(parse_int_grid("20,21,23", name="y"), [20, 21, 23])
        self.assertEqual(parse_direction_grid("both,long_only"), ["both", "long_only"])
        self.assertEqual(parse_optional_float_grid("none,0.1", name="a"), [None, 0.1])
        self.assertEqual(parse_optional_int_grid("none,23", name="b"), [None, 23])
        with self.assertRaises(ValueError):
            parse_float_grid("bad", name="x")
        with self.assertRaises(ValueError):
            parse_int_grid("", name="y")

    def test_grid_parser_space_separated_inputs(self) -> None:
        self.assertEqual(parse_float_grid(["0.5", "1.0"], name="x"), [0.5, 1.0])
        self.assertEqual(parse_float_grid(["0.5,1.0", "2.0"], name="x"), [0.5, 1.0, 2.0])
        self.assertEqual(parse_int_grid(["20", "21", "23"], name="y"), [20, 21, 23])
        self.assertEqual(parse_direction_grid(["both", "long_only", "short_only"]), ["both", "long_only", "short_only"])
        self.assertEqual(parse_optional_float_grid(["none", "0.2"], name="a"), [None, 0.2])
        self.assertEqual(parse_optional_int_grid(["none", "22"], name="b"), [None, 22])

    def test_run_sweep_passes_fees_and_slippage_to_each_backtest_call(self) -> None:
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-02 00:00:00+00:00",
            ]
        )
        ohlc = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
            },
            index=idx,
        )
        bt_df = pd.DataFrame(
            {
                "equity": [10_000.0, 10_010.0, 10_020.0],
                "fees_paid": [1.0, 1.0, 1.0],
            },
            index=idx,
        )
        trades = pd.DataFrame(
            [
                {
                    "pnl": 10.0,
                    "return_pct": 0.01,
                    "reason": "TP2",
                }
            ]
        )

        calls: list[tuple[float, float]] = []

        def fake_backtest(*args, **kwargs):
            calls.append((float(kwargs["fee_bps"]), float(kwargs["slippage_bps"])))
            return bt_df.copy(), trades.copy()

        with tempfile.TemporaryDirectory() as tmp, patch("run_parameter_sweep.load_cached_ohlc", return_value=ohlc), patch(
            "run_parameter_sweep.filter_ohlc_window", return_value=ohlc
        ), patch("run_parameter_sweep.add_monday_range", return_value=ohlc), patch(
            "run_parameter_sweep.backtest_sweep_fade", side_effect=fake_backtest
        ):
            results, _ = run_sweep(
                symbol="BTCUSDT",
                interval="1h",
                start="2024-01-01",
                end="2024-01-02",
                output_dir=tmp,
                fee_bps=12.0,
                slippage_bps=7.0,
                risk_fractions=[0.01, 0.02],
                tp2_to_full_values=[0.5],
                friday_cutoff_hours_utc=[22, 23],
            )

        self.assertEqual(len(calls), 4)
        self.assertTrue(all(fee == 12.0 and slip == 7.0 for fee, slip in calls))
        self.assertEqual(len(results), 4)
        self.assertTrue((results["fee_bps"] == 12.0).all())
        self.assertTrue((results["slippage_bps"] == 7.0).all())
        self.assertIn("total_fees_paid", results.columns)
        self.assertIn("cost_drag_pct", results.columns)


if __name__ == "__main__":
    unittest.main()
