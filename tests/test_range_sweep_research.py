from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.range_sweep_research import (
    SweepAnalysisConfig,
    analyze_range_sweeps,
    summarize_range_sweeps,
    range_sweep_tables,
)


def _daily_sweep_dataset() -> pd.DataFrame:
    rows = [
        ("2024-01-01 00:00:00+00:00", 105, 110, 100, 105),  # day 1 range setup
        ("2024-01-01 01:00:00+00:00", 105, 108, 102, 106),
        ("2024-01-02 00:00:00+00:00", 109, 112, 107, 109),  # high sweep (close back inside)
        ("2024-01-02 01:00:00+00:00", 109, 109.5, 104, 105),  # midpoint hit
        ("2024-01-02 02:00:00+00:00", 105, 106, 99, 100),  # opposite boundary hit
        ("2024-01-02 03:00:00+00:00", 109, 113, 108, 109),  # second high sweep same day
        ("2024-01-02 04:00:00+00:00", 109, 111, 108, 109),  # allows path for second sweep
        ("2024-01-03 00:00:00+00:00", 101, 103, 98, 101),  # low sweep (vs day2 low=99)
        ("2024-01-03 01:00:00+00:00", 101, 106, 100, 105),  # midpoint hit for low sweep
    ]
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [1.0] * len(rows),
        },
        index=idx,
    )


def _continuation_dataset() -> pd.DataFrame:
    rows = [
        ("2024-01-01 00:00:00+00:00", 105, 110, 100, 105),
        ("2024-01-01 01:00:00+00:00", 105, 108, 102, 106),
        ("2024-01-02 00:00:00+00:00", 109, 111, 108, 109),  # high sweep
        ("2024-01-02 01:00:00+00:00", 109, 116, 108, 115),  # +0.5R (range is 10)
        ("2024-01-02 02:00:00+00:00", 115, 121, 114, 120),  # +1.0R
    ]
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [1.0] * len(rows),
        },
        index=idx,
    )


def _weekly_dataset() -> pd.DataFrame:
    rows = []
    # Week 1 (Mon/Tue)
    rows.extend(
        [
            ("2024-01-01 00:00:00+00:00", 100, 110, 90, 100),
            ("2024-01-02 00:00:00+00:00", 100, 108, 92, 101),
        ]
    )
    # Week 2 event: sweep above week1 high
    rows.extend(
        [
            ("2024-01-08 00:00:00+00:00", 109, 112, 108, 109),
            ("2024-01-08 01:00:00+00:00", 109, 110, 95, 96),
        ]
    )
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [1.0] * len(rows),
        },
        index=idx,
    )


def _monthly_dataset() -> pd.DataFrame:
    rows = [
        ("2024-01-30 00:00:00+00:00", 100, 110, 90, 100),
        ("2024-01-31 00:00:00+00:00", 100, 109, 91, 102),
        ("2024-02-01 00:00:00+00:00", 108, 112, 107, 109),  # sweep above Jan high
        ("2024-02-01 01:00:00+00:00", 109, 110, 95, 96),
    ]
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [1.0] * len(rows),
        },
        index=idx,
    )


def _pre_event_vol_lookahead_dataset() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00:00+00:00", periods=80, freq="1h")
    df = pd.DataFrame(index=idx)
    df["open"] = 100.0
    df["high"] = 100.5
    df["low"] = 99.5
    df["close"] = 100.0
    df["volume"] = 1.0
    # Big move on event and after, but pre-event should remain near 0.
    df.loc[pd.Timestamp("2024-01-02 00:00:00+00:00"), ["high", "low", "close"]] = [111.0, 98.0, 109.0]
    df.loc[pd.Timestamp("2024-01-02 01:00:00+00:00"), ["high", "low", "close"]] = [115.0, 95.0, 100.0]
    return df


class RangeSweepResearchTests(unittest.TestCase):
    def test_high_low_sweep_detection_and_confirmation(self) -> None:
        cfg = SweepAnalysisConfig(
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="all_sweeps",
        )
        events = analyze_range_sweeps(_daily_sweep_dataset(), base_interval="1h", config=cfg)
        self.assertGreaterEqual(len(events), 3)
        self.assertIn("high_sweep", set(events["side"]))
        self.assertIn("low_sweep", set(events["side"]))

    def test_first_sweep_only_vs_all_sweeps(self) -> None:
        cfg_all = SweepAnalysisConfig(
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="all_sweeps",
        )
        cfg_first = SweepAnalysisConfig(
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="first_sweep_per_reference_period",
        )
        events_all = analyze_range_sweeps(_daily_sweep_dataset(), base_interval="1h", config=cfg_all)
        events_first = analyze_range_sweeps(_daily_sweep_dataset(), base_interval="1h", config=cfg_first)
        self.assertGreater(len(events_all), len(events_first))

    def test_midpoint_and_opposite_hit_detection(self) -> None:
        cfg = SweepAnalysisConfig(
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="first_sweep_per_reference_period",
        )
        events = analyze_range_sweeps(_daily_sweep_dataset(), base_interval="1h", config=cfg)
        high = events[events["side"] == "high_sweep"].iloc[0]
        self.assertTrue(bool(high["hit_midpoint"]))
        self.assertTrue(bool(high["hit_opposite_boundary"]))
        self.assertIsNotNone(high["time_to_midpoint_hours"])
        self.assertIsNotNone(high["time_to_opposite_boundary_hours"])

    def test_continuation_detection(self) -> None:
        cfg = SweepAnalysisConfig(
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="first_sweep_per_reference_period",
        )
        events = analyze_range_sweeps(_continuation_dataset(), base_interval="1h", config=cfg)
        row = events.iloc[0]
        self.assertTrue(bool(row["continued_0_5r"]))
        self.assertTrue(bool(row["continued_1_0r"]))

    def test_mae_mfe_in_range_units(self) -> None:
        cfg = SweepAnalysisConfig(
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="first_sweep_per_reference_period",
            holding_window_mode="fixed_n_candles",
            fixed_n_candles=2,
        )
        events = analyze_range_sweeps(_daily_sweep_dataset(), base_interval="1h", config=cfg)
        row = events[events["side"] == "high_sweep"].iloc[0]
        self.assertAlmostEqual(float(row["mfe_r"]), 1.0, places=10)  # 109 -> 99 on ref range 10
        self.assertAlmostEqual(float(row["mae_r"]), 0.05, places=10)  # 109 -> 109.5 on ref range 10

    def test_no_lookahead_previous_day_week_month_reference(self) -> None:
        day_cfg = SweepAnalysisConfig(reference_timeframe="previous_day", execution_timeframe="1h", confirmation_mode="close_back_inside", event_mode="first_sweep_per_reference_period")
        day_events = analyze_range_sweeps(_daily_sweep_dataset(), base_interval="1h", config=day_cfg)
        self.assertAlmostEqual(float(day_events.iloc[0]["ref_high"]), 110.0, places=10)

        week_cfg = SweepAnalysisConfig(reference_timeframe="previous_week", execution_timeframe="1h", confirmation_mode="close_back_inside", event_mode="first_sweep_per_reference_period")
        week_events = analyze_range_sweeps(_weekly_dataset(), base_interval="1h", config=week_cfg)
        self.assertFalse(week_events.empty)
        self.assertAlmostEqual(float(week_events.iloc[0]["ref_high"]), 110.0, places=10)

        month_cfg = SweepAnalysisConfig(reference_timeframe="previous_month", execution_timeframe="1h", confirmation_mode="close_back_inside", event_mode="first_sweep_per_reference_period")
        month_events = analyze_range_sweeps(_monthly_dataset(), base_interval="1h", config=month_cfg)
        self.assertFalse(month_events.empty)
        self.assertAlmostEqual(float(month_events.iloc[0]["ref_high"]), 110.0, places=10)

    def test_volatility_and_trend_use_pre_event_data(self) -> None:
        cfg = SweepAnalysisConfig(
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="first_sweep_per_reference_period",
        )
        events = analyze_range_sweeps(_pre_event_vol_lookahead_dataset(), base_interval="1h", config=cfg)
        self.assertFalse(events.empty)
        first = events.iloc[0]
        self.assertLessEqual(float(first["pre_event_volatility"]) if pd.notna(first["pre_event_volatility"]) else 0.0, 1e-9)
        self.assertIn(str(first["trend_regime"]), {"uptrend", "downtrend", "neutral", "unknown"})

    def test_summary_and_tables(self) -> None:
        cfg = SweepAnalysisConfig(
            reference_timeframe="previous_day",
            execution_timeframe="1h",
            confirmation_mode="close_back_inside",
            event_mode="all_sweeps",
        )
        events = analyze_range_sweeps(_daily_sweep_dataset(), base_interval="1h", config=cfg)
        summary = summarize_range_sweeps(events)
        tables = range_sweep_tables(events)
        self.assertIn("num_sweeps", summary)
        self.assertIn("by_side", tables)
        self.assertIn("by_year", tables)
        self.assertTrue(isinstance(tables["by_side"], pd.DataFrame))

    def test_invalid_resolution_request(self) -> None:
        cfg = SweepAnalysisConfig(reference_timeframe="previous_day", execution_timeframe="1h")
        with self.assertRaises(ValueError):
            analyze_range_sweeps(_daily_sweep_dataset(), base_interval="4h", config=cfg)


if __name__ == "__main__":
    unittest.main()
