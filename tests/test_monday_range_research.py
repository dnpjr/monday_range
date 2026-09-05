from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.monday_range_research import (
    _range_bucket,
    analyze_monday_range_research,
    probability_tables,
    summarize_monday_range_research,
)


def _base_week_ohlc() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-02 00:00:00+00:00",  # Tue entry
            "2024-01-02 01:00:00+00:00",  # Tue +1h
            "2024-01-03 00:00:00+00:00",
            "2024-01-05 23:00:00+00:00",
        ]
    )
    return pd.DataFrame(
        {
            "open": [105.0, 105.0, 100.0, 100.0, 100.0, 100.0],
            "high": [110.0, 109.0, 104.0, 106.0, 111.0, 105.0],
            "low": [100.0, 101.0, 99.0, 99.0, 98.0, 99.0],
            "close": [105.0, 106.0, 100.0, 104.0, 110.0, 101.0],
        },
        index=idx,
    )


def _three_week_ohlc_for_quantiles() -> pd.DataFrame:
    rows = []
    # Week 1
    rows.extend(
        [
            ("2024-01-01 00:00:00+00:00", 105, 110, 100, 105),
            ("2024-01-01 01:00:00+00:00", 105, 109, 101, 106),
            ("2024-01-02 00:00:00+00:00", 100, 104, 99, 100),
            ("2024-01-02 01:00:00+00:00", 100, 106, 99, 104),
            ("2024-01-03 00:00:00+00:00", 100, 111, 98, 110),
            ("2024-01-05 23:00:00+00:00", 100, 105, 99, 101),
        ]
    )
    # Week 2
    rows.extend(
        [
            ("2024-01-08 00:00:00+00:00", 210, 220, 200, 210),
            ("2024-01-08 01:00:00+00:00", 210, 219, 201, 212),
            ("2024-01-09 00:00:00+00:00", 200, 205, 198, 202),
            ("2024-01-09 01:00:00+00:00", 202, 215, 197, 214),
            ("2024-01-10 00:00:00+00:00", 210, 221, 196, 220),
            ("2024-01-12 23:00:00+00:00", 210, 212, 205, 206),
        ]
    )
    # Week 3
    rows.extend(
        [
            ("2024-01-15 00:00:00+00:00", 320, 330, 300, 320),
            ("2024-01-15 01:00:00+00:00", 320, 329, 302, 322),
            ("2024-01-16 00:00:00+00:00", 300, 308, 299, 304),
            ("2024-01-16 01:00:00+00:00", 304, 320, 298, 318),
            ("2024-01-17 00:00:00+00:00", 310, 331, 297, 330),
            ("2024-01-19 23:00:00+00:00", 310, 312, 305, 306),
        ]
    )

    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
        },
        index=idx,
    )


def _lookahead_guard_ohlc() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00:00+00:00", periods=60, freq="1H")
    df = pd.DataFrame(index=idx)
    df["open"] = 100.0
    df["high"] = 101.0
    df["low"] = 99.0
    df["close"] = 100.0
    # Add large volatility only after Tuesday 00:00 entry point
    df.loc[idx >= pd.Timestamp("2024-01-02 00:00:00+00:00"), "high"] = 150.0
    df.loc[idx >= pd.Timestamp("2024-01-02 00:00:00+00:00"), "low"] = 50.0
    return df


def _entry_bar_only_tp_hit_ohlc() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-02 00:00:00+00:00",  # entry bar
            "2024-01-02 01:00:00+00:00",  # no TP continuation
            "2024-01-05 23:00:00+00:00",
        ]
    )
    return pd.DataFrame(
        {
            "open": [105.0, 105.0, 100.0, 100.0, 100.0],
            "high": [110.0, 109.0, 106.0, 104.0, 103.0],  # TP1 (105) touched on entry bar only
            "low": [100.0, 101.0, 99.5, 99.0, 98.0],
            "close": [105.0, 106.0, 100.0, 100.0, 99.0],
        },
        index=idx,
    )


class MondayRangeResearchTests(unittest.TestCase):
    def test_tp_fraction_semantics_and_mfe_mae_in_range_units(self) -> None:
        res = analyze_monday_range_research(
            _base_week_ohlc(),
            tp1_range_fraction=0.5,
            tp2_range_fraction=1.0,
            stop_mode="opposite_boundary",
            event_mode="hypothetical_both_sides",
        )
        self.assertEqual(len(res), 2)
        long_row = res[res["side"] == "LONG"].iloc[0]
        short_row = res[res["side"] == "SHORT"].iloc[0]
        self.assertAlmostEqual(float(long_row["tp1_price"]), 105.0, places=10)
        self.assertAlmostEqual(float(long_row["tp2_price"]), 110.0, places=10)
        self.assertAlmostEqual(float(short_row["tp1_price"]), 105.0, places=10)
        self.assertAlmostEqual(float(short_row["tp2_price"]), 100.0, places=10)

        # Monday range is 10; long MAE before TP1 should be (100-99)/10 = 0.1
        self.assertAlmostEqual(float(long_row["mae_before_tp1_r"]), 0.1, places=10)
        # Long MFE before TP1 on hit bar should be (106-100)/10 = 0.6
        self.assertAlmostEqual(float(long_row["mfe_before_tp1_r"]), 0.6, places=10)

    def test_time_to_target_hours_and_bars(self) -> None:
        res = analyze_monday_range_research(_base_week_ohlc(), event_mode="hypothetical_both_sides")
        long_row = res[res["side"] == "LONG"].iloc[0]
        self.assertAlmostEqual(float(long_row["time_to_tp1_hours"]), 1.0, places=10)
        self.assertEqual(int(long_row["time_to_tp1_bars"]), 1)

    def test_quantiles_and_stop_guidance(self) -> None:
        res = analyze_monday_range_research(_three_week_ohlc_for_quantiles(), event_mode="hypothetical_both_sides")
        summary = summarize_monday_range_research(res)
        self.assertIn("stop_guidance", summary)
        sg = summary["stop_guidance"]
        self.assertIn("suggested_stop_tp1_conservative_r", sg)
        self.assertIn("suggested_stop_tp2_conservative_r", sg)
        self.assertIn("suggested_stop_very_conservative_r", sg)
        self.assertIsNotNone(summary["mae_before_tp1_winners_stats"]["p90"])
        self.assertIsNotNone(summary["mae_before_tp2_winners_stats"]["p95"])

    def test_volatility_and_trend_regime_bucketing(self) -> None:
        res = analyze_monday_range_research(_three_week_ohlc_for_quantiles(), event_mode="hypothetical_both_sides")
        self.assertIn("vol_regime", res.columns)
        self.assertIn("trend_regime", res.columns)
        self.assertTrue(set(res["vol_regime"].dropna().unique()).issubset({"low_vol", "mid_vol", "high_vol", "unknown"}))
        self.assertTrue(set(res["trend_regime"].dropna().unique()).issubset({"uptrend", "sideways", "downtrend", "unknown"}))

        tabs = probability_tables(res)
        self.assertIn("by_vol_regime", tabs)
        self.assertIn("by_trend_regime", tabs)
        self.assertIn("p_stop_before_tp1", tabs["by_vol_regime"].columns)

    def test_range_bucket_2p5_percent_increments(self) -> None:
        self.assertEqual(_range_bucket(0.001), "0-2.5%")
        self.assertEqual(_range_bucket(0.025), "0-2.5%")
        self.assertEqual(_range_bucket(0.026), "2.5-5.0%")
        self.assertEqual(_range_bucket(0.050), "2.5-5.0%")
        self.assertEqual(_range_bucket(0.051), "5.0-7.5%")
        self.assertEqual(_range_bucket(0.075), "5.0-7.5%")
        self.assertEqual(_range_bucket(0.076), "7.5%+")

    def test_no_lookahead_regime_features_use_pre_entry_data(self) -> None:
        res = analyze_monday_range_research(_lookahead_guard_ohlc(), event_mode="hypothetical_both_sides")
        self.assertTrue((res["pre_entry_volatility_24"].fillna(0.0) <= 1e-12).all())

    def test_event_mode_aliases_and_invalid_value(self) -> None:
        base = _base_week_ohlc()
        a = analyze_monday_range_research(base, event_mode="sweep_events")
        b = analyze_monday_range_research(base, event_mode="hypothetical_both_sides")
        self.assertEqual(len(a), len(b))
        c = analyze_monday_range_research(base, event_mode="strategy_entries")
        d = analyze_monday_range_research(base, event_mode="strategy_signal")
        self.assertEqual(len(c), len(d))
        with self.assertRaises(ValueError):
            analyze_monday_range_research(base, event_mode="bad_mode")

    def test_stop_range_fraction_passes_through(self) -> None:
        res_1 = analyze_monday_range_research(
            _base_week_ohlc(),
            stop_mode="range_fraction",
            stop_range_fraction=1.0,
            event_mode="hypothetical_both_sides",
        )
        res_2 = analyze_monday_range_research(
            _base_week_ohlc(),
            stop_mode="range_fraction",
            stop_range_fraction=0.5,
            event_mode="hypothetical_both_sides",
        )
        self.assertFalse(res_1.empty)
        self.assertFalse(res_2.empty)
        # LONG stop = mon_low - frac*range; with lower frac stop should be higher.
        long1 = float(res_1[res_1["side"] == "LONG"].iloc[0]["stop_price"])
        long2 = float(res_2[res_2["side"] == "LONG"].iloc[0]["stop_price"])
        self.assertGreater(long2, long1)

    def test_tp_events_do_not_trigger_on_entry_bar(self) -> None:
        res = analyze_monday_range_research(
            _entry_bar_only_tp_hit_ohlc(),
            event_mode="hypothetical_both_sides",
            tp1_range_fraction=0.5,
            tp2_range_fraction=1.0,
        )
        long_row = res[res["side"] == "LONG"].iloc[0]
        self.assertFalse(bool(long_row["hit_tp1"]))
        self.assertTrue(pd.isna(long_row["time_to_tp1_hours"]))

    def test_summary_and_grouped_median_ignore_nonpositive_time_values(self) -> None:
        synthetic = pd.DataFrame(
            [
                {
                    "iso_year": 2024,
                    "iso_week": 1,
                    "side": "LONG",
                    "entry_time": pd.Timestamp("2024-01-02T00:00:00Z"),
                    "week_has_long_sweep": True,
                    "week_has_short_sweep": False,
                    "time_to_tp1_hours": 0.0,
                    "time_to_tp2_hours": 0.0,
                    "time_to_stop_hours": None,
                    "hit_tp1": True,
                    "hit_tp2": False,
                    "hit_stop": False,
                    "tp1_before_stop": True,
                    "tp2_before_stop": False,
                    "tp2_after_tp1": False,
                    "stop_before_tp1": False,
                    "mae_before_tp1_r": 0.1,
                    "mae_before_tp2_r": None,
                    "mfe_before_eventual_stop_r": None,
                    "mfe_in_range": 0.6,
                    "mae_in_range": 0.1,
                    "outcome": "TP1_ONLY",
                    "range_bucket": "0-2.5%",
                    "vol_regime": "low_vol",
                    "trend_regime": "uptrend",
                },
                {
                    "iso_year": 2024,
                    "iso_week": 1,
                    "side": "LONG",
                    "entry_time": pd.Timestamp("2024-01-03T00:00:00Z"),
                    "week_has_long_sweep": True,
                    "week_has_short_sweep": False,
                    "time_to_tp1_hours": 2.0,
                    "time_to_tp2_hours": 4.0,
                    "time_to_stop_hours": None,
                    "hit_tp1": True,
                    "hit_tp2": True,
                    "hit_stop": False,
                    "tp1_before_stop": True,
                    "tp2_before_stop": True,
                    "tp2_after_tp1": True,
                    "stop_before_tp1": False,
                    "mae_before_tp1_r": 0.2,
                    "mae_before_tp2_r": 0.3,
                    "mfe_before_eventual_stop_r": None,
                    "mfe_in_range": 1.1,
                    "mae_in_range": 0.2,
                    "outcome": "TP2",
                    "range_bucket": "0-2.5%",
                    "vol_regime": "low_vol",
                    "trend_regime": "uptrend",
                },
            ]
        )
        summary = summarize_monday_range_research(synthetic)
        self.assertAlmostEqual(float(summary["tp1_time_hours"]["median"]), 2.0, places=10)
        self.assertAlmostEqual(float(summary["tp2_time_hours"]["median"]), 4.0, places=10)

        by_dir = probability_tables(synthetic)["by_direction"]
        row = by_dir.iloc[0]
        self.assertAlmostEqual(float(row["median_time_to_tp1_hours"]), 2.0, places=10)
        self.assertAlmostEqual(float(row["median_time_to_tp2_hours"]), 4.0, places=10)


if __name__ == "__main__":
    unittest.main()
