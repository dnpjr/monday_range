from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
import json
import os
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.dashboard_actions import (
    assert_research_mode_only,
    get_cache_status,
    update_data_cache,
    run_research_backtest,
    load_backtest_outputs,
    discover_backtest_runs,
    run_research_diagnostics,
    load_diagnostics_outputs,
    build_dataset_label,
    get_date_preset_range,
    validate_backtest_date_range,
    build_backtest_run_label,
    format_probability_pct,
    format_hours,
    stop_mode_label,
    stop_mode_value,
    stop_mode_visible_inputs,
    run_research_lab_analysis,
    normalize_event_mode,
    normalize_research_lab_mode,
    default_stop_mode_for_context,
    estimate_sweep_retest_signals,
    generate_cost_sensitivity_grid,
    generate_neighbourhood_grid,
    build_train_validation_table,
    run_cost_sensitivity,
    run_neighbourhood_test,
    run_range_sweep_lab_analysis,
)
from src.features import add_monday_range
from src.research import analyze_weekly_sweep_signals, summarize_research


class DashboardActionsTests(unittest.TestCase):
    def test_research_safety_guard_exists_and_passes(self) -> None:
        self.assertTrue(assert_research_mode_only())

    def test_cache_status_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = get_cache_status(symbol="BTCUSDT", interval="1h", cache_dir=tmp)
            self.assertFalse(status["cache_exists"])
            self.assertEqual(status["rows"], 0)
            self.assertIsNone(status["first_candle"])

    def test_cache_status_when_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "BTCUSDT_1h.csv"
            pd.DataFrame(
                {
                    "open_time": [
                        "2024-01-01T00:00:00+00:00",
                        "2024-01-01T00:00:00+00:00",
                        "2024-01-01T01:00:00+00:00",
                    ],
                    "close_time": [
                        "2024-01-01T00:59:59.999+00:00",
                        "2024-01-01T00:59:59.999+00:00",
                        "2024-01-01T01:59:59.999+00:00",
                    ],
                    "open": [100, 101, 102],
                    "high": [101, 102, 103],
                    "low": [99, 100, 101],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [1, 1, 1],
                    "quote_asset_volume": [1, 1, 1],
                    "num_trades": [1, 1, 1],
                    "taker_buy_base_asset_volume": [1, 1, 1],
                    "taker_buy_quote_asset_volume": [1, 1, 1],
                }
            ).to_csv(p, index=False)

            status = get_cache_status(symbol="BTCUSDT", interval="1h", cache_dir=tmp)
            self.assertTrue(status["cache_exists"])
            self.assertEqual(status["rows"], 3)
            self.assertEqual(status["duplicate_count"], 1)
            self.assertEqual(status["cache_file"], str(p))
            self.assertIsNotNone(status["latest_candle_age_seconds"])

    def test_update_data_cache_passes_args_correctly(self) -> None:
        fake = {"symbol": "BTCUSDT", "interval": "1h", "rows_saved": 10}
        with patch("src.dashboard_actions.run_cache_update", return_value=fake) as run:
            out = update_data_cache(
                symbol="BTCUSDT",
                interval="1h",
                lookback_days=365,
                start=None,
                end=None,
                force=True,
                cache_dir="data/binance",
            )
            self.assertEqual(out, fake)
            run.assert_called_once_with(
                symbol="BTCUSDT",
                interval="1h",
                lookback_days=365,
                start=None,
                end=None,
                force=True,
                cache_dir="data/binance",
            )

    def test_run_range_sweep_lab_analysis_wrapper(self) -> None:
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-02 00:00:00+00:00",
                "2024-01-02 01:00:00+00:00",
            ]
        )
        ohlc = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5, 103.5],
            },
            index=idx,
        )
        out = run_range_sweep_lab_analysis(
            ohlcv=ohlc,
            base_interval="1h",
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="all_sweeps",
        )
        self.assertIn("events", out)
        self.assertIn("summary", out)
        self.assertIn("tables", out)
        self.assertIn("config", out)

    def test_discover_backtest_runs_missing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "missing_backtests"
            runs = discover_backtest_runs(p)
            self.assertEqual(runs, [])

    def test_discover_backtest_runs_populated_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run1 = root / "BTCUSDT_1h_old"
            run2 = root / "BTCUSDT_1h_new"
            run1.mkdir(parents=True, exist_ok=True)
            run2.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"pnl": 1.0}]).to_csv(run1 / "trades.csv", index=False)
            pd.DataFrame([{"pnl": 2.0}]).to_csv(run2 / "trades.csv", index=False)
            with open(run1 / "summary.json", "w", encoding="utf-8") as f:
                json.dump({"total_return_pct": 1.0}, f)
            with open(run2 / "summary.json", "w", encoding="utf-8") as f:
                json.dump({"total_return_pct": 2.0}, f)
            os.utime(run1 / "summary.json", (1_700_000_000, 1_700_000_000))
            os.utime(run2 / "summary.json", (1_800_000_000, 1_800_000_000))

            runs = discover_backtest_runs(root)
            self.assertEqual(len(runs), 2)
            self.assertTrue(runs[0].endswith("BTCUSDT_1h_new"))
            self.assertTrue(runs[1].endswith("BTCUSDT_1h_old"))

    def test_load_backtest_outputs_complete_and_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            complete = root / "complete"
            incomplete = root / "incomplete"
            complete.mkdir(parents=True, exist_ok=True)
            incomplete.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"pnl": 1.0}]).to_csv(complete / "trades.csv", index=False)
            with open(complete / "summary.json", "w", encoding="utf-8") as f:
                json.dump({"total_return_pct": 1.0}, f)
            with open(complete / "config.json", "w", encoding="utf-8") as f:
                json.dump({"symbol": "BTCUSDT"}, f)

            with open(incomplete / "summary.json", "w", encoding="utf-8") as f:
                json.dump({"total_return_pct": 0.0}, f)

            loaded_complete = load_backtest_outputs(complete)
            self.assertTrue(loaded_complete["summary_exists"])
            self.assertTrue(loaded_complete["config_exists"])
            self.assertTrue(loaded_complete["trades_exists"])
            self.assertEqual(loaded_complete["trades_count"], 1)

            loaded_incomplete = load_backtest_outputs(incomplete)
            self.assertTrue(loaded_incomplete["summary_exists"])
            self.assertFalse(loaded_incomplete["config_exists"])
            self.assertFalse(loaded_incomplete["trades_exists"])
            self.assertEqual(loaded_incomplete["trades_count"], 0)

    def test_run_research_backtest_passes_args(self) -> None:
        fake = {"summary": {"total_return_pct": 1.23}, "out_dir": "data/backtests/run1", "trades_count": 10}
        with patch("src.dashboard_actions.run_backtest_cached", return_value=fake) as run:
            out = run_research_backtest(
                strategy="sweep_retest",
                symbol="BTCUSDT",
                interval="1h",
                start="2024-01-01",
                end="2024-02-01",
                fee_bps=10.0,
                slippage_bps=5.0,
                risk_fraction=0.02,
                tp1_range_fraction=0.5,
                tp2_range_fraction=1.0,
                tp2_to_full=0.5,
                friday_cutoff_hour_utc=20,
                min_range_pct=0.01,
                max_range_pct=0.05,
                direction="both",
                max_entry_day_utc=2,
                max_entry_hour_utc=23,
                sma_period=50,
                tp1_to_mid=1.0,
                tp1_close_fraction=0.4,
                stop_mode="range_fraction",
                stop_range_fraction=0.8,
                stop_pct=0.02,
                move_stop_to_breakeven_after_tp1=False,
                breakeven_includes_fees=True,
                exit_style="single_target",
                single_target_mode=True,
                single_target_level="tp1",
                output_dir="data/backtests",
            )
            self.assertEqual(out, fake)
            run.assert_called_once_with(
                strategy="sweep_retest",
                symbol="BTCUSDT",
                interval="1h",
                start="2024-01-01",
                end="2024-02-01",
                fee_bps=10.0,
                slippage_bps=5.0,
                risk_fraction=0.02,
                tp1_range_fraction=0.5,
                tp2_range_fraction=1.0,
                tp2_to_full=0.5,
                friday_cutoff_hour_utc=20,
                min_range_pct=0.01,
                max_range_pct=0.05,
                direction="both",
                max_entry_day_utc=2,
                max_entry_hour_utc=23,
                sma_period=50,
                tp1_to_mid=1.0,
                tp1_close_fraction=0.4,
                stop_mode="range_fraction",
                stop_range_fraction=0.8,
                stop_pct=0.02,
                move_stop_to_breakeven_after_tp1=False,
                breakeven_includes_fees=True,
                exit_style="single_target",
                single_target_mode=True,
                single_target_level="tp1",
                output_dir="data/backtests",
            )

    def test_dataset_label_and_presets(self) -> None:
        status = {
            "cache_exists": True,
            "rows": 1000,
            "first_candle": "2021-05-21T00:00:00+00:00",
            "last_candle": "2026-05-20T00:00:00+00:00",
        }
        label = build_dataset_label("BTCUSDT", "1h", status)
        self.assertIn("BTCUSDT 1h", label)
        self.assertIn("2021-05-21", label)
        self.assertIn("2026-05-20", label)

        preset = get_date_preset_range("2022", status)
        self.assertEqual(preset["start"], "2022-01-01")
        self.assertEqual(preset["end"], "2022-12-31")

    def test_preset_clipping_and_validation(self) -> None:
        status = {
            "cache_exists": True,
            "rows": 100,
            "first_candle": "2024-01-10T00:00:00+00:00",
            "last_candle": "2024-04-15T00:00:00+00:00",
        }
        clipped = get_date_preset_range("2021-2024 training", status)
        self.assertTrue(clipped["clipped"])
        self.assertEqual(clipped["start"], "2024-01-10")
        self.assertEqual(clipped["end"], "2024-04-15")

        invalid = validate_backtest_date_range("2024-03-01", "2024-02-01", status)
        self.assertFalse(invalid["valid"])
        self.assertIn("End date", str(invalid["error"]))

        valid = validate_backtest_date_range("2024-01-01", "2024-04-20", status)
        self.assertTrue(valid["valid"])
        self.assertIsNotNone(valid["warning"])

    def test_build_backtest_run_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            with open(root / "summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "symbol": "BTCUSDT",
                        "interval": "1h",
                        "start_used": "2025-01-01 00:00:00+00:00",
                        "end_used": "2026-05-20 00:00:00+00:00",
                        "total_return_pct": -0.79,
                        "num_trades": 59,
                    },
                    f,
                )
            label = build_backtest_run_label(root)
            self.assertIn("BTCUSDT 1h", label)
            self.assertIn("return -0.79%", label)
            self.assertIn("trades 59", label)

    def test_formatting_helpers(self) -> None:
        self.assertEqual(format_probability_pct(0.1234), "12.3%")
        self.assertEqual(format_probability_pct(None), "N/A")
        self.assertEqual(format_hours(4.26), "4.3h")
        self.assertEqual(format_hours(None), "N/A")

    def test_stop_mode_label_mapping(self) -> None:
        self.assertEqual(stop_mode_value("Opposite boundary"), "opposite_boundary")
        self.assertEqual(stop_mode_value("Range fraction"), "range_fraction")
        self.assertEqual(stop_mode_value("Fixed percent"), "fixed_pct")
        self.assertEqual(stop_mode_label("opposite_boundary"), "Opposite boundary")

    def test_stop_mode_visible_inputs(self) -> None:
        self.assertEqual(
            stop_mode_visible_inputs("range_fraction"),
            {"show_stop_range_fraction": True, "show_stop_pct": False},
        )
        self.assertEqual(
            stop_mode_visible_inputs("fixed_pct"),
            {"show_stop_range_fraction": False, "show_stop_pct": True},
        )
        self.assertEqual(
            stop_mode_visible_inputs("opposite_boundary"),
            {"show_stop_range_fraction": False, "show_stop_pct": False},
        )
        self.assertEqual(default_stop_mode_for_context("research_lab"), "range_fraction")
        self.assertEqual(default_stop_mode_for_context("backtest_runner"), "opposite_boundary")

    def test_normalize_event_mode(self) -> None:
        self.assertEqual(normalize_event_mode("strategy_entries"), "strategy_signal")
        self.assertEqual(normalize_event_mode("sweep_events"), "hypothetical_both_sides")
        self.assertEqual(normalize_event_mode("strategy_signal"), "strategy_signal")
        with self.assertRaises(ValueError):
            normalize_event_mode("bad_mode")

    def test_normalize_research_lab_mode(self) -> None:
        self.assertEqual(normalize_research_lab_mode("original_sweep_retest"), "original_sweep_retest")
        self.assertEqual(normalize_research_lab_mode("legacy"), "original_sweep_retest")
        self.assertEqual(normalize_research_lab_mode("new_path_analysis"), "new_path_analysis")
        self.assertEqual(normalize_research_lab_mode("path"), "new_path_analysis")
        with self.assertRaises(ValueError):
            normalize_research_lab_mode("bad_mode")

    def test_run_research_lab_analysis_accepts_dashboard_parameters(self) -> None:
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-02 00:00:00+00:00",
                "2024-01-02 01:00:00+00:00",
                "2024-01-05 23:00:00+00:00",
            ]
        )
        ohlc = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100],
                "high": [110, 109, 106, 107, 105],
                "low": [90, 91, 99, 99, 99],
                "close": [100, 101, 100, 101, 100],
            },
            index=idx,
        )
        out = run_research_lab_analysis(
            ohlc=ohlc,
            tp1_range_fraction=0.5,
            tp2_range_fraction=1.0,
            stop_mode="range_fraction",
            stop_range_fraction=0.5,
            stop_pct=0.01,
            friday_cutoff_hour_utc=23,
            event_mode="sweep_events",
        )
        self.assertIn("results", out)
        self.assertIn("summary", out)
        self.assertIn("tables", out)
        self.assertEqual(out["analysis_mode"], "new_path_analysis")
        self.assertEqual(out["event_mode_normalized"], "hypothetical_both_sides")

    def test_run_research_lab_analysis_original_mode_matches_legacy_summary(self) -> None:
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-02 00:00:00+00:00",
                "2024-01-02 01:00:00+00:00",
                "2024-01-03 00:00:00+00:00",
                "2024-01-05 23:00:00+00:00",
            ]
        )
        ohlc = pd.DataFrame(
            {
                "open": [105, 105, 100, 100, 100, 100],
                "high": [110, 109, 104, 106, 111, 105],
                "low": [100, 101, 99, 99, 98, 99],
                "close": [105, 106, 100, 104, 110, 101],
            },
            index=idx,
        )

        out = run_research_lab_analysis(
            ohlc=ohlc,
            analysis_mode="original_sweep_retest",
        )
        self.assertEqual(out["analysis_mode"], "original_sweep_retest")
        self.assertIsNone(out["event_mode_normalized"])
        self.assertEqual(out["tables"], {})

        legacy_signals = analyze_weekly_sweep_signals(add_monday_range(ohlc))
        legacy_summary = summarize_research(legacy_signals)
        for key in (
            "total_signals",
            "p_hit_mid",
            "p_hit_full",
            "avg_mae_mid",
            "p90_mae_mid",
            "avg_mae_full",
            "p90_mae_full",
            "avg_reward_ratio",
            "p_sweep_retest",
            "p_sweep_retest_winners",
        ):
            self.assertIn(key, out["summary"])
            self.assertEqual(out["summary"][key], legacy_summary[key])

    def test_estimate_sweep_retest_signals(self) -> None:
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
                "open": [110, 112, 95, 100, 101],
                "high": [130, 128, 100, 101, 104],
                "low": [90, 92, 89, 99, 100],
                "close": [110, 111, 95, 100, 103],
            },
            index=idx,
        )
        self.assertEqual(estimate_sweep_retest_signals(ohlc), 1)

    def test_generate_cost_sensitivity_grid(self) -> None:
        grid = generate_cost_sensitivity_grid()
        self.assertEqual(len(grid), 12)
        self.assertIn({"fee_bps": 0.0, "slippage_bps": 0.0}, grid)
        self.assertIn({"fee_bps": 10.0, "slippage_bps": 5.0}, grid)

    def test_generate_neighbourhood_grid(self) -> None:
        grid = generate_neighbourhood_grid(tp1_range_fraction=0.5, stop_range_fraction=0.5)
        self.assertEqual(len(grid), 9)
        self.assertIn({"tp1_range_fraction": 0.5, "stop_range_fraction": 0.5}, grid)
        self.assertTrue(all(float(r["tp1_range_fraction"]) > 0 for r in grid))
        self.assertTrue(all(float(r["stop_range_fraction"]) > 0 for r in grid))

    def test_build_train_validation_table(self) -> None:
        train = {"total_return_pct": 10.0, "max_drawdown": 0.2, "total_fees_paid": 100.0, "num_trades": 20, "win_rate": 0.6}
        valid = {"total_return_pct": -2.0, "max_drawdown": 0.3, "total_fees_paid": 40.0, "num_trades": 7, "win_rate": 0.4}
        table = build_train_validation_table(train, valid)
        self.assertEqual(list(table["period"]), ["train", "validation"])
        self.assertAlmostEqual(float(table.iloc[0]["total_return_pct"]), 10.0)
        self.assertAlmostEqual(float(table.iloc[1]["total_return_pct"]), -2.0)

    def test_run_cost_sensitivity_result_table_construction(self) -> None:
        def fake_run(**kwargs):
            fee = float(kwargs["fee_bps"])
            slip = float(kwargs["slippage_bps"])
            return {
                "summary": {
                    "total_return_pct": 100.0 - fee - slip,
                    "max_drawdown": 0.1 + (fee + slip) / 1000.0,
                    "total_fees_paid": fee * 10.0,
                    "num_trades": 10,
                    "win_rate": 0.5,
                }
            }

        cfg = {
            "strategy": "sweep_retest",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "start": "2024-01-01",
            "end": "2024-12-31",
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "risk_fraction": 0.02,
            "tp1_range_fraction": 0.5,
            "tp2_range_fraction": 1.0,
            "tp2_to_full": 1.0,
            "friday_cutoff_hour_utc": 23,
            "min_range_pct": None,
            "max_range_pct": None,
            "direction": "both",
            "max_entry_day_utc": None,
            "max_entry_hour_utc": None,
            "sma_period": None,
            "tp1_to_mid": 1.0,
            "tp1_close_fraction": 0.5,
            "stop_mode": "range_fraction",
            "stop_range_fraction": 0.5,
            "stop_pct": 0.01,
            "move_stop_to_breakeven_after_tp1": True,
            "breakeven_includes_fees": False,
            "exit_style": "partial_tp2",
            "single_target_mode": False,
            "single_target_level": "tp2",
        }
        df = run_cost_sensitivity(config=cfg, run_fn=fake_run)
        self.assertEqual(len(df), 12)
        self.assertIn("total_return_pct", df.columns)
        self.assertIn("max_drawdown", df.columns)
        self.assertIn("total_fees_paid", df.columns)

    def test_run_neighbourhood_result_table_construction(self) -> None:
        def fake_run(**kwargs):
            tp1 = float(kwargs["tp1_range_fraction"])
            stopf = float(kwargs["stop_range_fraction"])
            return {
                "summary": {
                    "total_return_pct": tp1 * 10.0 - stopf * 5.0,
                    "max_drawdown": stopf / 10.0,
                    "total_fees_paid": 1.0,
                    "num_trades": 5,
                    "win_rate": 0.5,
                }
            }

        cfg = {
            "strategy": "sweep_retest",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "start": "2024-01-01",
            "end": "2024-12-31",
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "risk_fraction": 0.02,
            "tp1_range_fraction": 0.5,
            "tp2_range_fraction": 1.0,
            "tp2_to_full": 1.0,
            "friday_cutoff_hour_utc": 23,
            "min_range_pct": None,
            "max_range_pct": None,
            "direction": "both",
            "max_entry_day_utc": None,
            "max_entry_hour_utc": None,
            "sma_period": None,
            "tp1_to_mid": 1.0,
            "tp1_close_fraction": 0.5,
            "stop_mode": "range_fraction",
            "stop_range_fraction": 0.5,
            "stop_pct": 0.01,
            "move_stop_to_breakeven_after_tp1": True,
            "breakeven_includes_fees": False,
            "exit_style": "partial_tp2",
            "single_target_mode": False,
            "single_target_level": "tp2",
        }
        df = run_neighbourhood_test(config=cfg, run_fn=fake_run)
        self.assertEqual(len(df), 9)
        self.assertIn("tp1_range_fraction", df.columns)
        self.assertIn("stop_range_fraction", df.columns)
        self.assertIn("total_return_pct", df.columns)

    def test_run_research_lab_analysis_invalid_event_mode(self) -> None:
        idx = pd.to_datetime(["2024-01-01 00:00:00+00:00", "2024-01-01 01:00:00+00:00"])
        ohlc = pd.DataFrame({"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1]}, index=idx)
        with self.assertRaises(ValueError):
            run_research_lab_analysis(ohlc=ohlc, event_mode="nonsense")

    def test_run_research_diagnostics_passes_args(self) -> None:
        fake = {"diagnostics": {"num_trades": 3}, "out_paths": {"diagnostics_json": "x"}}
        with patch("src.dashboard_actions.run_backtest_diagnostics", return_value=fake) as run:
            out = run_research_diagnostics("data/backtests/run1", cache_dir="data/binance")
            self.assertEqual(out, fake)
            run.assert_called_once_with(
                backtest_dir="data/backtests/run1",
                cache_dir="data/binance",
            )

    def test_load_diagnostics_outputs_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_diagnostics_outputs(tmp)
            self.assertFalse(loaded["diagnostics_exists"])
            self.assertFalse(loaded["by_year_exists"])
            self.assertFalse(loaded["by_direction_exists"])
            self.assertFalse(loaded["by_exit_reason_exists"])
            self.assertFalse(loaded["by_range_bucket_exists"])

    def test_load_diagnostics_outputs_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            with open(root / "diagnostics.json", "w", encoding="utf-8") as f:
                json.dump({"net_pnl_total": 12.3}, f)
            pd.DataFrame([{"exit_year": 2024, "net_pnl": 1.0}]).to_csv(root / "diagnostics_by_year.csv", index=False)
            pd.DataFrame([{"side": "LONG", "net_pnl": 2.0}]).to_csv(root / "diagnostics_by_direction.csv", index=False)
            pd.DataFrame([{"reason": "TP2", "net_pnl": 3.0}]).to_csv(root / "diagnostics_by_exit_reason.csv", index=False)
            pd.DataFrame([{"range_bucket": "1-2%", "net_pnl": 4.0}]).to_csv(root / "diagnostics_by_range_bucket.csv", index=False)

            loaded = load_diagnostics_outputs(root)
            self.assertTrue(loaded["diagnostics_exists"])
            self.assertTrue(loaded["by_year_exists"])
            self.assertTrue(loaded["by_direction_exists"])
            self.assertTrue(loaded["by_exit_reason_exists"])
            self.assertTrue(loaded["by_range_bucket_exists"])
            self.assertIsInstance(loaded["diagnostics"], dict)
            self.assertEqual(float(loaded["diagnostics"]["net_pnl_total"]), 12.3)
            self.assertEqual(len(loaded["by_year"]), 1)
            self.assertEqual(len(loaded["by_direction"]), 1)
            self.assertEqual(len(loaded["by_exit_reason"]), 1)
            self.assertEqual(len(loaded["by_range_bucket"]), 1)


if __name__ == "__main__":
    unittest.main()
