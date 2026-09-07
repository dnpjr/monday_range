from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from run_backtest import (
    load_cached_ohlc,
    filter_ohlc_window,
    build_summary,
    write_outputs,
    run_backtest_cached,
)
from src.features import add_monday_range
from src.backtest import backtest_sweep_fade


class RunBacktestTests(unittest.TestCase):
    def test_load_cached_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = pathlib.Path(tmp)
            f = cache_dir / "BTCUSDT_1h.csv"
            pd.DataFrame(
                {
                    "open_time": [
                        "2024-01-01T00:00:00+00:00",
                        "2024-01-01T01:00:00+00:00",
                        "2024-01-01T01:00:00+00:00",
                    ],
                    "open": [100, 101, 111],
                    "high": [110, 111, 121],
                    "low": [90, 91, 101],
                    "close": [105, 106, 116],
                }
            ).to_csv(f, index=False)

            ohlc = load_cached_ohlc("BTCUSDT", "1h", cache_dir=cache_dir)
            self.assertEqual(len(ohlc), 2)
            self.assertAlmostEqual(float(ohlc.iloc[-1]["open"]), 111.0)

    def test_filter_window(self) -> None:
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-02 00:00:00+00:00",
            ]
        )
        df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3]}, index=idx)
        out = filter_ohlc_window(df, "2024-01-01", "2024-01-01")
        self.assertEqual(len(out), 2)  # end is inclusive day via exclusive next day

    def test_summary_metrics_and_output_files(self) -> None:
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-02 00:00:00+00:00",
                "2024-01-02 01:00:00+00:00",
                "2024-01-02 02:00:00+00:00",
            ]
        )
        ohlc = pd.DataFrame(
            {
                "open": [100.0, 100.0, 95.0, 100.0, 95.0],
                "high": [110.0, 108.0, 96.0, 99.0, 80.0],
                "low": [90.0, 92.0, 89.0, 95.0, 69.0],
                "close": [100.0, 101.0, 91.0, 96.0, 75.0],
            },
            index=idx,
        )
        df = add_monday_range(ohlc)
        df_out, trades = backtest_sweep_fade(
            df,
            initial_capital=10_000.0,
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_frac=0.5,
            fee_bps=10.0,
            slippage_bps=5.0,
        )
        summary = build_summary(
            strategy="current_monday_range",
            signal_count=1,
            exit_style="partial_tp2",
            single_target_level=None,
            symbol="BTCUSDT",
            interval="1h",
            initial_cash=10_000.0,
            df_out=df_out,
            trades=trades,
            start_used="2024-01-01",
            end_used="2024-01-02",
        )
        self.assertIn("initial_cash", summary)
        self.assertIn("final_equity", summary)
        self.assertIn("total_return_pct", summary)
        self.assertIn("total_fees_paid", summary)
        self.assertIn("max_drawdown", summary)
        self.assertIn("num_trades", summary)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = write_outputs(
                output_root=tmp,
                symbol="BTCUSDT",
                interval="1h",
                trades=trades,
                summary=summary,
                config={"symbol": "BTCUSDT"},
            )
            self.assertTrue((out_dir / "trades.csv").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "config.json").exists())

            with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
                loaded_summary = json.load(f)
            self.assertEqual(loaded_summary["symbol"], "BTCUSDT")

    def test_run_backtest_cached_legacy_aliases_supported(self) -> None:
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-02 00:00:00+00:00",
            ]
        )
        ohlc = pd.DataFrame(
            {"open": [100.0, 100.0, 100.0], "high": [101.0, 101.0, 101.0], "low": [99.0, 99.0, 99.0], "close": [100.0, 100.0, 100.0]},
            index=idx,
        )
        bt_df = pd.DataFrame({"equity": [10_000.0, 10_000.0, 10_000.0], "fees_paid": [0.0, 0.0, 0.0]}, index=idx)
        trades = pd.DataFrame([])

        with tempfile.TemporaryDirectory() as tmp, patch("run_backtest.load_cached_ohlc", return_value=ohlc), patch(
            "run_backtest.filter_ohlc_window", return_value=ohlc
        ), patch("run_backtest.add_monday_range", return_value=ohlc), patch(
            "run_backtest.backtest_sweep_fade", return_value=(bt_df, trades)
        ) as bt:
            run_backtest_cached(
                symbol="BTCUSDT",
                interval="1h",
                tp1_range_fraction=0.5,
                tp2_range_fraction=1.0,
                tp1_to_mid=1.0,
                tp2_to_full=1.0,
                output_dir=tmp,
            )
        kwargs = bt.call_args.kwargs
        self.assertEqual(kwargs["strategy"], "current_monday_range")
        self.assertEqual(kwargs["tp1_range_fraction"], 0.5)
        self.assertEqual(kwargs["tp2_range_fraction"], 1.0)
        self.assertEqual(kwargs["tp1_to_mid"], 1.0)
        self.assertEqual(kwargs["tp2_to_full"], 1.0)

    def test_run_backtest_cached_sweep_retest_strategy_pass_through(self) -> None:
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-02 00:00:00+00:00",
            ]
        )
        ohlc = pd.DataFrame(
            {"open": [100.0, 100.0, 100.0], "high": [101.0, 101.0, 101.0], "low": [99.0, 99.0, 99.0], "close": [100.0, 100.0, 100.0]},
            index=idx,
        )
        bt_df = pd.DataFrame({"equity": [10_000.0, 10_000.0, 10_000.0], "fees_paid": [0.0, 0.0, 0.0]}, index=idx)
        bt_df.attrs["signal_count"] = 3
        trades = pd.DataFrame([])

        with tempfile.TemporaryDirectory() as tmp, patch("run_backtest.load_cached_ohlc", return_value=ohlc), patch(
            "run_backtest.filter_ohlc_window", return_value=ohlc
        ), patch("run_backtest.add_monday_range", return_value=ohlc), patch(
            "run_backtest.analyze_weekly_sweep_signals", return_value=pd.DataFrame([{"x": 1}, {"x": 2}])
        ), patch("run_backtest.backtest_sweep_fade", return_value=(bt_df, trades)) as bt:
            out = run_backtest_cached(
                strategy="sweep_retest",
                symbol="BTCUSDT",
                interval="1h",
                output_dir=tmp,
            )
        kwargs = bt.call_args.kwargs
        self.assertEqual(kwargs["strategy"], "sweep_retest")
        self.assertEqual(kwargs["stop_mode"], "swept_boundary_offset")
        self.assertEqual(kwargs["risk_base"], "current_equity")
        self.assertEqual(kwargs["max_leverage"], 1.0)
        self.assertEqual(kwargs["intrabar_policy"], "conservative_stop_first")
        self.assertEqual(kwargs["stop_range_fraction"], 0.5)
        self.assertEqual(out["summary"]["strategy"], "sweep_retest")
        self.assertEqual(out["summary"]["signal_count"], 2)
        self.assertEqual(out["config"]["dataset_version"], "btcusdt_binance_spot_1h_v1")
        self.assertEqual(len(out["config"]["dataset_sha256"]), 64)
        self.assertEqual(out["config"]["accounting_version"], "marked_equity_v1")
        self.assertIn("code_commit", out["config"])
        self.assertIn("run_timestamp_utc", out["config"])


if __name__ == "__main__":
    unittest.main()
