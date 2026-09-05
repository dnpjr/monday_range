from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.backtest_diagnostics import (
    compute_buy_hold_benchmark,
    compute_diagnostics,
    load_backtest_artifacts,
    save_diagnostics_outputs,
)


class BacktestDiagnosticsTests(unittest.TestCase):
    def test_compute_diagnostics_core_and_groupings(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "entry_time": "2024-01-02T01:00:00+00:00",
                    "exit_time": "2024-01-02T03:00:00+00:00",
                    "side": "LONG",
                    "reason": "TP2",
                    "pnl": 95.0,
                    "net_pnl": 95.0,
                    "gross_pnl": 100.0,
                    "fees_paid": 5.0,
                    "holding_hours": 2.0,
                    "mon_range_pct": 0.015,
                },
                {
                    "entry_time": "2025-02-04T01:00:00+00:00",
                    "exit_time": "2025-02-04T02:00:00+00:00",
                    "side": "SHORT",
                    "reason": "STOP",
                    "pnl": -55.0,
                    "net_pnl": -55.0,
                    "gross_pnl": -50.0,
                    "fees_paid": 5.0,
                    "holding_hours": 1.0,
                    "mon_range_pct": 0.035,
                },
            ]
        )
        summary = {"initial_cash": 10_000.0, "total_fees_paid": 10.0}
        config = {}
        diagnostics, tables = compute_diagnostics(trades, summary, config)

        self.assertEqual(diagnostics["num_trades"], 2)
        self.assertAlmostEqual(float(diagnostics["gross_pnl_total"]), 50.0)
        self.assertAlmostEqual(float(diagnostics["net_pnl_total"]), 40.0)
        self.assertAlmostEqual(float(diagnostics["fees_total"]), 10.0)
        self.assertAlmostEqual(float(diagnostics["fee_drag_pct_of_initial_cash"]), 0.1)
        self.assertAlmostEqual(float(diagnostics["average_gross_trade_pnl"]), 25.0)
        self.assertAlmostEqual(float(diagnostics["average_net_trade_pnl"]), 20.0)
        self.assertAlmostEqual(float(diagnostics["average_holding_hours"]), 1.5)

        self.assertIn("by_year", tables)
        self.assertIn("by_direction", tables)
        self.assertIn("by_exit_reason", tables)
        self.assertIn("by_range_bucket", tables)
        self.assertEqual(set(tables["by_direction"]["side"].tolist()), {"LONG", "SHORT"})
        self.assertTrue(len(tables["by_range_bucket"]) >= 1)
        reasons = set(tables["by_exit_reason"]["reason"].tolist())
        self.assertEqual(reasons, {"TP1", "TP2", "STOP", "FRIDAY"})

    def test_buy_hold_benchmark_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = pathlib.Path(tmp)
            pd.DataFrame(
                {
                    "open_time": [
                        "2024-01-01T00:00:00+00:00",
                        "2024-01-01T01:00:00+00:00",
                        "2024-01-01T02:00:00+00:00",
                    ],
                    "open": [100, 110, 120],
                    "high": [101, 111, 121],
                    "low": [99, 109, 119],
                    "close": [100, 110, 120],
                }
            ).to_csv(cache_dir / "BTCUSDT_1h.csv", index=False)

            summary = {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start_used": "2024-01-01",
                "end_used": "2024-01-01",
                "initial_cash": 10_000.0,
            }
            config = {}
            bm = compute_buy_hold_benchmark(summary, config, cache_dir=cache_dir)
            self.assertIsNotNone(bm)
            assert bm is not None
            self.assertAlmostEqual(float(bm["buy_hold_return_pct"]), 20.0)
            self.assertAlmostEqual(float(bm["buy_hold_final_equity"]), 12_000.0)

    def test_load_and_save_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            trades = pd.DataFrame(
                [
                    {
                        "entry_time": "2024-01-02T01:00:00+00:00",
                        "exit_time": "2024-01-02T02:00:00+00:00",
                        "side": "LONG",
                        "reason": "FRIDAY",
                        "pnl": 1.0,
                    }
                ]
            )
            trades.to_csv(root / "trades.csv", index=False)
            with open(root / "summary.json", "w", encoding="utf-8") as f:
                json.dump({"symbol": "BTCUSDT", "interval": "1h", "initial_cash": 10_000.0}, f)
            with open(root / "config.json", "w", encoding="utf-8") as f:
                json.dump({"symbol": "BTCUSDT", "interval": "1h"}, f)

            loaded_trades, summary, config = load_backtest_artifacts(root)
            diagnostics, tables = compute_diagnostics(loaded_trades, summary, config)
            out_paths = save_diagnostics_outputs(backtest_dir=root, diagnostics=diagnostics, tables=tables)

            self.assertTrue((root / "diagnostics.json").exists())
            self.assertTrue((root / "diagnostics_by_year.csv").exists())
            self.assertTrue((root / "diagnostics_by_direction.csv").exists())
            self.assertTrue((root / "diagnostics_by_exit_reason.csv").exists())
            self.assertIn("diagnostics_json", out_paths)


if __name__ == "__main__":
    unittest.main()
