from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.backtest import backtest_sweep_fade
from src.metrics import equity_metrics, trade_metrics


def _featured(rows: list[dict], times: list[str] | None = None) -> pd.DataFrame:
    if times is None:
        times = [f"2024-01-02 {hour:02d}:00:00+00:00" for hour in range(len(rows))]
    frame = pd.DataFrame(rows, index=pd.to_datetime(times))
    defaults = {
        "mon_high": 110.0,
        "mon_low": 90.0,
        "mon_mid": 100.0,
        "mon_range": 20.0,
        "iso_year": 2024,
        "iso_week": 1,
        "weekday": 1,
        "is_tradeable": True,
    }
    for key, value in defaults.items():
        if key not in frame:
            frame[key] = value
        else:
            frame[key] = frame[key].fillna(value)
    return frame


def _long_open_position() -> pd.DataFrame:
    return _featured(
        [
            {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
            {"open": 100.0, "high": 101.0, "low": 94.0, "close": 95.0},
        ]
    )


class MarkedAccountingTests(unittest.TestCase):
    def test_mark_to_market_equity_during_open_position(self) -> None:
        out, trades = backtest_sweep_fade(
            _long_open_position(),
            risk_per_trade=100.0,
            max_leverage=10.0,
            tp1_at_mid=False,
            close_open_position_at_end=False,
        )
        self.assertTrue(trades.empty)
        qty = 100.0 / (100.0 - 70.0)
        last = out.iloc[-1]
        self.assertAlmostEqual(float(last["cash"]), 10_000.0)
        self.assertAlmostEqual(float(last["open_qty"]), qty)
        self.assertAlmostEqual(float(last["entry_price"]), 100.0)
        self.assertAlmostEqual(float(last["position_value"]), qty * 95.0)
        self.assertAlmostEqual(float(last["unrealized_pnl"]), (95.0 - 100.0) * qty)
        self.assertAlmostEqual(float(last["realized_pnl"]), 0.0)
        self.assertAlmostEqual(float(last["equity"]), 10_000.0 + (95.0 - 100.0) * qty)

    def test_drawdown_includes_open_losing_position(self) -> None:
        out, _ = backtest_sweep_fade(
            _long_open_position(),
            risk_per_trade=100.0,
            max_leverage=10.0,
            tp1_at_mid=False,
            close_open_position_at_end=False,
        )
        metrics = equity_metrics(out["equity"], bars_per_year=365.0 * 24.0)
        expected = float(out["equity"].iloc[-1] / out["equity"].iloc[0] - 1.0)
        self.assertAlmostEqual(float(metrics["max_drawdown"]), expected)
        self.assertLess(float(metrics["max_drawdown"]), 0.0)

    def test_current_equity_and_initial_capital_risk_bases_are_explicit(self) -> None:
        times = [
            "2024-01-02 00:00:00+00:00",
            "2024-01-02 01:00:00+00:00",
            "2024-01-02 02:00:00+00:00",
            "2024-01-09 00:00:00+00:00",
            "2024-01-09 01:00:00+00:00",
            "2024-01-09 02:00:00+00:00",
        ]
        rows = [
            {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0, "iso_week": 1},
            {"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.0, "iso_week": 1},
            {"open": 70.0, "high": 75.0, "low": 65.0, "close": 70.0, "iso_week": 1},
            {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0, "iso_week": 2, "is_tradeable": False},
            {"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.0, "iso_week": 2},
            {"open": 70.0, "high": 75.0, "low": 65.0, "close": 70.0, "iso_week": 2},
        ]
        data = _featured(rows, times)
        _, current = backtest_sweep_fade(
            data,
            risk_fraction=0.01,
            risk_base="current_equity",
            max_leverage=10.0,
            tp1_at_mid=False,
        )
        _, initial = backtest_sweep_fade(
            data,
            risk_fraction=0.01,
            risk_base="initial_capital",
            max_leverage=10.0,
            tp1_at_mid=False,
        )
        self.assertEqual(len(current), 2)
        self.assertAlmostEqual(float(current.iloc[0]["risk_capital"]), 100.0)
        self.assertAlmostEqual(float(current.iloc[1]["risk_capital"]), 99.0)
        self.assertAlmostEqual(float(initial.iloc[1]["risk_capital"]), 100.0)


    def test_equity_and_trade_metric_definitions(self) -> None:
        index = pd.to_datetime([
            "2024-01-01 00:00:00+00:00",
            "2024-07-01 00:00:00+00:00",
            "2024-12-31 00:00:00+00:00",
        ])
        equity = pd.Series([10_000.0, 11_000.0, 9_900.0], index=index)
        metrics = equity_metrics(equity, bars_per_year=365.0)
        self.assertAlmostEqual(float(metrics["total_return"]), -0.01)
        self.assertAlmostEqual(float(metrics["max_drawdown"]), -0.10)
        self.assertAlmostEqual(float(metrics["cagr"]), -0.01, places=3)
        self.assertIn("sharpe", metrics)
        self.assertIn("sortino", metrics)

        trades = pd.DataFrame(
            {
                "pnl": [100.0, -50.0],
                "return_pct": [0.01, -0.005],
                "fees_paid": [4.0, 6.0],
                "reason": ["TP2", "STOP"],
            }
        )
        tm = trade_metrics(trades)
        self.assertEqual(tm["num_trades"], 2)
        self.assertAlmostEqual(float(tm["win_rate"]), 0.5)
        self.assertAlmostEqual(float(tm["avg_pnl"]), 25.0)
        self.assertAlmostEqual(float(tm["profit_factor"]), 2.0)
        self.assertAlmostEqual(float(tm["total_fees"]), 10.0)

    def test_tight_stop_is_bounded_by_maximum_leverage(self) -> None:
        out, _ = backtest_sweep_fade(
            _featured(
                [
                    {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                    {"open": 100.0, "high": 101.0, "low": 100.0, "close": 100.0},
                ]
            ),
            risk_fraction=0.10,
            max_leverage=1.0,
            stop_mode="entry_fixed_pct",
            stop_pct=0.001,
            tp1_at_mid=False,
            close_open_position_at_end=False,
        )
        last = out.iloc[-1]
        self.assertAlmostEqual(float(last["open_qty"]), 100.0)
        self.assertAlmostEqual(float(last["gross_notional"]), 10_000.0)
        self.assertLessEqual(float(last["gross_notional"]), 10_000.0)


class FillConventionTests(unittest.TestCase):
    def test_gap_through_stop_long_uses_open_then_adverse_slippage(self) -> None:
        data = _featured(
            [
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                {"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.0},
                {"open": 60.0, "high": 75.0, "low": 55.0, "close": 65.0},
            ]
        )
        _, trades = backtest_sweep_fade(
            data,
            risk_per_trade=100.0,
            max_leverage=10.0,
            tp1_at_mid=False,
            slippage_bps=10.0,
            fee_bps=10.0,
        )
        self.assertEqual(trades.iloc[0]["reason"], "STOP")
        self.assertAlmostEqual(float(trades.iloc[0]["exit"]), 60.0 * 0.999)
        self.assertGreater(float(trades.iloc[0]["fees_paid"]), 0.0)

    def test_gap_through_stop_short_uses_open_then_adverse_slippage(self) -> None:
        data = _featured(
            [
                {"open": 105.0, "high": 111.0, "low": 104.0, "close": 109.0},
                {"open": 100.0, "high": 105.0, "low": 99.0, "close": 100.0},
                {"open": 140.0, "high": 145.0, "low": 135.0, "close": 140.0},
            ]
        )
        _, trades = backtest_sweep_fade(
            data,
            risk_per_trade=100.0,
            max_leverage=10.0,
            tp1_at_mid=False,
            slippage_bps=10.0,
            fee_bps=10.0,
        )
        self.assertEqual(trades.iloc[0]["side"], "SHORT")
        self.assertEqual(trades.iloc[0]["reason"], "STOP")
        self.assertAlmostEqual(float(trades.iloc[0]["exit"]), 140.0 * 1.001)

    def test_gap_through_target_uses_open_price(self) -> None:
        data = _featured(
            [
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0},
                {"open": 120.0, "high": 122.0, "low": 118.0, "close": 121.0},
            ]
        )
        _, trades = backtest_sweep_fade(
            data,
            risk_per_trade=100.0,
            max_leverage=10.0,
            single_target_mode=True,
            slippage_bps=5.0,
        )
        self.assertEqual(trades.iloc[0]["reason"], "TP2")
        self.assertAlmostEqual(float(trades.iloc[0]["exit"]), 120.0 * 0.9995)

    def test_same_bar_policy_is_explicit(self) -> None:
        data = _featured(
            [
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                {"open": 100.0, "high": 115.0, "low": 65.0, "close": 100.0},
            ]
        )
        _, conservative = backtest_sweep_fade(
            data,
            risk_per_trade=100.0,
            max_leverage=10.0,
            single_target_mode=True,
            intrabar_policy="conservative_stop_first",
        )
        _, target_first = backtest_sweep_fade(
            data,
            risk_per_trade=100.0,
            max_leverage=10.0,
            single_target_mode=True,
            intrabar_policy="target_first",
        )
        self.assertEqual(conservative.iloc[0]["reason"], "STOP")
        self.assertEqual(target_first.iloc[0]["reason"], "TP2")

    def test_terminal_position_closes_with_costs(self) -> None:
        data = _long_open_position()
        out, trades = backtest_sweep_fade(
            data,
            risk_per_trade=100.0,
            max_leverage=10.0,
            tp1_at_mid=False,
            fee_bps=10.0,
            slippage_bps=10.0,
        )
        trade = trades.iloc[0]
        self.assertEqual(trade["reason"], "END_OF_DATA")
        self.assertAlmostEqual(float(trade["exit"]), 95.0 * 0.999)
        self.assertGreater(float(trade["fees_paid"]), 0.0)
        self.assertAlmostEqual(float(out.iloc[-1]["unrealized_pnl"]), 0.0)
        self.assertAlmostEqual(float(out.iloc[-1]["equity"]), float(out.iloc[-1]["cash"]))


class PreservedSemanticsTests(unittest.TestCase):
    def test_next_bar_entry_and_one_trade_per_week(self) -> None:
        data = _featured(
            [
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                {"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.0},
                {"open": 70.0, "high": 75.0, "low": 65.0, "close": 70.0},
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0, "is_tradeable": False},
                {"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.0},
            ]
        )
        _, trades = backtest_sweep_fade(data, risk_per_trade=100.0, max_leverage=10.0, tp1_at_mid=False)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["entry_time"], data.index[1])
        self.assertAlmostEqual(float(trades.iloc[0]["entry"]), 100.0)

    def test_partial_exit_and_friday_exit_are_preserved(self) -> None:
        times = [
            "2024-01-02 00:00:00+00:00",
            "2024-01-02 01:00:00+00:00",
            "2024-01-03 00:00:00+00:00",
            "2024-01-05 23:00:00+00:00",
        ]
        data = _featured(
            [
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                {"open": 95.0, "high": 99.0, "low": 94.0, "close": 96.0},
                {"open": 98.0, "high": 101.0, "low": 96.0, "close": 99.0},
                {"open": 101.0, "high": 103.0, "low": 100.5, "close": 102.0, "weekday": 4},
            ],
            times,
        )
        _, trades = backtest_sweep_fade(data, risk_per_trade=100.0, max_leverage=10.0)
        trade = trades.iloc[0]
        self.assertEqual(trade["reason"], "FRIDAY")
        self.assertAlmostEqual(float(trade["qty"]), float(trade["initial_qty"]) * 0.5)


    def test_full_fraction_tp1_closes_and_records_trade(self) -> None:
        data = _featured(
            [
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                {"open": 95.0, "high": 99.0, "low": 94.0, "close": 96.0},
                {"open": 98.0, "high": 101.0, "low": 96.0, "close": 100.0},
            ]
        )
        out, trades = backtest_sweep_fade(
            data,
            risk_per_trade=100.0,
            max_leverage=10.0,
            tp1_close_fraction=1.0,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["reason"], "TP1")
        self.assertAlmostEqual(float(trades.iloc[0]["qty"]), float(trades.iloc[0]["initial_qty"]))
        self.assertAlmostEqual(float(out.iloc[-1]["open_qty"]), 0.0)

    def test_explicit_stop_semantics(self) -> None:
        data = _featured(
            [
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            ]
        )
        expected = {
            "swept_boundary": 90.0,
            "swept_boundary_offset": 80.0,
            "entry_fixed_pct": 98.0,
        }
        for mode, stop in expected.items():
            with self.subTest(mode=mode):
                _, trades = backtest_sweep_fade(
                    data,
                    risk_per_trade=100.0,
                    max_leverage=10.0,
                    stop_mode=mode,
                    stop_range_fraction=0.5,
                    stop_pct=0.02,
                    tp1_at_mid=False,
                )
                self.assertEqual(trades.iloc[0]["reason"], "END_OF_DATA")
                self.assertAlmostEqual(float(trades.iloc[0]["initial_stop"]), stop)


if __name__ == "__main__":
    unittest.main()
