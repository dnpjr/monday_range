from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.dashboard_data import (
    load_paper_state,
    load_paper_trades,
    find_latest_binance_cache,
    load_binance_cache,
    find_backtest_runs,
    load_backtest_run,
    build_state_snapshot,
    compute_data_health,
    reconstruct_completed_trades,
)


class DashboardDataTests(unittest.TestCase):
    def test_missing_files_return_empty_safe_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.assertIsNone(load_paper_state(root / "paper_state.json"))
            self.assertTrue(load_paper_trades(root / "paper_trades.csv").empty)
            self.assertIsNone(find_latest_binance_cache(root / "binance"))

    def test_load_state_and_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            state_path = root / "paper_state.json"
            trades_path = root / "paper_trades.csv"

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"cash": 1234.5, "position_side": "LONG"}, f)
            pd.DataFrame(
                {
                    "timestamp": ["2024-01-01T00:00:00+00:00"],
                    "event": ["ENTRY"],
                    "side": ["LONG"],
                    "price": [100.0],
                    "qty": [1.0],
                    "fee": [0.1],
                    "realized_pnl": [-0.1],
                    "cash_after": [1234.4],
                    "position_qty_after": [1.0],
                    "trade_id": [1],
                }
            ).to_csv(trades_path, index=False)

            st = load_paper_state(state_path)
            tr = load_paper_trades(trades_path)
            self.assertIsNotNone(st)
            self.assertEqual(st["position_side"], "LONG")
            self.assertEqual(len(tr), 1)
            self.assertEqual(str(tr["timestamp"].dtype), "datetime64[ns, UTC]")

    def test_cache_discovery_loading_and_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = pathlib.Path(tmp) / "binance"
            cache_dir.mkdir(parents=True, exist_ok=True)
            f = cache_dir / "BTCUSDT_1h_1_2.csv"
            pd.DataFrame(
                {
                    "open_time": [
                        "2024-01-01T00:00:00+00:00",
                        "2024-01-01T01:00:00+00:00",
                        "2024-01-01T01:00:00+00:00",  # duplicate
                        "2024-01-01T02:00:00+00:00",
                        "2024-01-01T04:00:00+00:00",  # gap at 03:00
                    ],
                    "close_time": [
                        "2024-01-01T00:59:59.999000+00:00",
                        "2024-01-01T01:59:59.999000+00:00",
                        "2024-01-01T01:59:59.999000+00:00",
                        "2024-01-01T02:59:59.999000+00:00",
                        "2024-01-01T04:59:59.999000+00:00",
                    ],
                    "open": [1.0, 2.0, 2.1, 3.0, 5.0],
                    "high": [1.2, 2.2, 2.3, 3.2, 5.2],
                    "low": [0.8, 1.8, 1.9, 2.8, 4.8],
                    "close": [1.1, 2.1, 2.2, 3.1, 5.1],
                }
            ).to_csv(f, index=False)

            meta = find_latest_binance_cache(cache_dir)
            self.assertIsNotNone(meta)
            self.assertEqual(meta.symbol, "BTCUSDT")
            self.assertEqual(meta.interval, "1h")

            candles = load_binance_cache(meta)
            self.assertEqual(len(candles), 4)  # deduped

            health = compute_data_health(candles, interval="1h")
            self.assertTrue(health["has_candles"])
            self.assertEqual(health["duplicate_candles"], 0)
            self.assertTrue(health["timestamps_utc_ok"])
            self.assertGreaterEqual(int(health["missing_candles_estimate"]), 1)

    def test_snapshot_and_trade_reconstruction(self) -> None:
        candles = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00+00:00",
                        "2024-01-01 01:00:00+00:00",
                        "2024-01-02 00:00:00+00:00",
                    ]
                ),
                "close_time": pd.to_datetime(
                    [
                        "2024-01-01 00:59:59.999000+00:00",
                        "2024-01-01 01:59:59.999000+00:00",
                        "2024-01-02 00:59:59.999000+00:00",
                    ]
                ),
                "open": [100.0, 101.0, 102.0],
                "high": [110.0, 111.0, 112.0],
                "low": [90.0, 91.0, 92.0],
                "close": [101.0, 102.0, 103.0],
            }
        )
        events = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-02T01:00:00+00:00", "2024-01-02T02:00:00+00:00"], utc=True
                ),
                "event": ["ENTRY", "TP2"],
                "side": ["LONG", "LONG"],
                "price": [100.0, 105.0],
                "qty": [1.0, 1.0],
                "fee": [0.1, 0.1],
                "realized_pnl": [-0.1, 4.9],
                "cash_after": [9999.9, 10004.8],
                "position_qty_after": [1.0, 0.0],
                "trade_id": [7, 7],
            }
        )
        state = {
            "cash": 10004.8,
            "position_side": None,
            "position_qty": 0.0,
            "entry_price": None,
            "last_processed_open_time": "2024-01-02T00:00:00+00:00",
        }

        snap = build_state_snapshot(state, events, candles, friday_cutoff_hour_utc=23)
        self.assertAlmostEqual(float(snap["realized_pnl"]), 4.8, places=10)
        self.assertEqual(snap["event_count"], 2)
        self.assertEqual(snap["trade_count"], 1)
        self.assertIsNotNone(snap["mon_high"])

        completed = reconstruct_completed_trades(events)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed.iloc[0]["exit_reason"], "TP2")
        self.assertAlmostEqual(float(completed.iloc[0]["pnl"]), 4.8, places=10)

    def test_backtest_run_discovery_and_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "backtests"
            run1 = root / "BTCUSDT_1h_20260101T000000Z"
            run2 = root / "BTCUSDT_1h_20260102T000000Z"
            run1.mkdir(parents=True, exist_ok=True)
            run2.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"pnl": 1.0}, {"pnl": 2.0}]).to_csv(run1 / "trades.csv", index=False)
            pd.DataFrame([{"pnl": 3.0}]).to_csv(run2 / "trades.csv", index=False)
            with open(run1 / "summary.json", "w", encoding="utf-8") as f:
                json.dump({"total_return_pct": 1.1}, f)
            with open(run2 / "summary.json", "w", encoding="utf-8") as f:
                json.dump({"total_return_pct": 2.2}, f)

            runs = find_backtest_runs(root)
            self.assertEqual(len(runs), 2)
            loaded = load_backtest_run(runs[0])
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertIn("summary", loaded)
            self.assertIn("trades_count", loaded)


if __name__ == "__main__":
    unittest.main()
