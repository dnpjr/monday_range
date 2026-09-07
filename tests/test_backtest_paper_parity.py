from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.backtest import backtest_sweep_fade
from src.features import add_monday_range
from src.paper_trader import process_candles


def _parity_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-02 00:00:00+00:00",  # signal candle
            "2024-01-02 01:00:00+00:00",  # entry candle
            "2024-01-02 02:00:00+00:00",  # no exit
            "2024-01-02 03:00:00+00:00",  # hit TP2
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 100.0, 100.0, 105.0],
            "high": [110.0, 108.0, 96.0, 101.0, 105.0, 111.0],
            "low": [90.0, 92.0, 89.0, 99.0, 99.0, 103.0],
            "close": [100.0, 101.0, 91.0, 100.0, 104.0, 110.0],
        },
        index=idx,
    )


def _parity_df_partial_tp2() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",  # Monday
            "2024-01-01 01:00:00+00:00",  # Monday
            "2024-01-02 00:00:00+00:00",  # signal candle
            "2024-01-02 01:00:00+00:00",  # entry candle
            "2024-01-02 02:00:00+00:00",  # hit reduced tp2, not full tp2
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 100.0, 104.5],
            "high": [110.0, 108.0, 96.0, 101.0, 106.0],
            "low": [90.0, 92.0, 89.0, 99.0, 103.5],
            "close": [100.0, 101.0, 91.0, 100.0, 105.5],
        },
        index=idx,
    )


def _paper_trade_summaries(events: list[dict]) -> list[dict]:
    out: list[dict] = []
    by_trade: dict[int, dict] = {}
    for e in events:
        trade_id = e["trade_id"]
        if trade_id is None:
            continue
        bucket = by_trade.setdefault(
            trade_id,
            {
                "side": None,
                "entry_time": None,
                "entry_price": None,
                "entry_notional": 0.0,
                "pnl": 0.0,
                "exit_reason": None,
                "exit_time": None,
                "exit_price": None,
            },
        )
        if e["event"] == "ENTRY":
            bucket["side"] = e["side"]
            bucket["entry_time"] = e["timestamp"]
            bucket["entry_price"] = float(e["price"])
            qty = float(e["qty"])
            entry_notional = abs(float(e["price"]) * qty)
            bucket["entry_notional"] = entry_notional
            # entry event realized_pnl includes negative entry fee
            bucket["pnl"] += float(e["realized_pnl"])
        else:
            bucket["pnl"] += float(e["realized_pnl"])
            if float(e["position_qty_after"]) == 0.0:
                bucket["exit_reason"] = e["event"]
                bucket["exit_time"] = e["timestamp"]
                bucket["exit_price"] = float(e["price"])

    for trade_id in sorted(by_trade):
        t = by_trade[trade_id]
        if t["exit_reason"] is None:
            continue
        out.append(t)
    return out


class BacktestPaperParityTests(unittest.TestCase):
    def test_backfill_matches_backtest_trade_behavior(self) -> None:
        risk_per_trade = 100.0
        stop_mult = 1.0
        tp1_frac = 0.5
        fee_bps = 0.0
        slippage_bps = 0.0
        friday_cutoff_hour_utc = 23

        df = add_monday_range(_parity_df())
        _, bt_trades = backtest_sweep_fade(
            df,
            initial_capital=10_000.0,
            risk_per_trade=risk_per_trade,
            stop_mult=stop_mult,
            tp1_frac=tp1_frac,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            friday_cutoff_hour_utc=friday_cutoff_hour_utc,
        )

        state = {
            "cash": 10_000.0,
            "position_side": None,
            "position_qty": 0.0,
            "entry_price": None,
            "entry_time": None,
            "entry_notional": 0.0,
            "trade_pnl_accum": 0.0,
            "stop": None,
            "tp1": None,
            "tp2": None,
            "tp1_taken": False,
            "week_id_of_last_entry": None,
            "trade_id_seq": 0,
            "active_trade_id": None,
            "last_processed_open_time": None,
        }
        _, paper_events = process_candles(
            df,
            state,
            interval="1h",
            risk_per_trade=risk_per_trade,
            stop_mult=stop_mult,
            tp1_frac=tp1_frac,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            friday_cutoff_hour_utc=friday_cutoff_hour_utc,
        )
        paper_trades = _paper_trade_summaries(paper_events)

        self.assertEqual(len(bt_trades), len(paper_trades))
        self.assertEqual(len(bt_trades), 1)

        bt = bt_trades.iloc[0]
        pt = paper_trades[0]
        self.assertEqual(bt["side"], pt["side"])
        self.assertEqual(pd.Timestamp(bt["entry_time"]).isoformat(), pt["entry_time"])
        self.assertAlmostEqual(float(bt["entry"]), float(pt["entry_price"]), places=10)
        self.assertEqual(bt["reason"], pt["exit_reason"])
        self.assertEqual(pd.Timestamp(bt["exit_time"]).isoformat(), pt["exit_time"])
        self.assertAlmostEqual(float(bt["exit"]), float(pt["exit_price"]), places=10)
        self.assertAlmostEqual(float(bt["pnl"]), float(pt["pnl"]), places=8)

    def test_backfill_matches_backtest_with_nondefault_tp2_to_full(self) -> None:
        risk_per_trade = 100.0
        stop_mult = 1.0
        tp1_frac = 0.5
        tp2_to_full = 0.5
        fee_bps = 0.0
        slippage_bps = 0.0
        friday_cutoff_hour_utc = 23

        df = add_monday_range(_parity_df_partial_tp2())
        _, bt_trades = backtest_sweep_fade(
            df,
            initial_capital=10_000.0,
            risk_per_trade=risk_per_trade,
            stop_mult=stop_mult,
            tp1_frac=tp1_frac,
            tp2_to_full=tp2_to_full,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            friday_cutoff_hour_utc=friday_cutoff_hour_utc,
        )

        state = {
            "cash": 10_000.0,
            "position_side": None,
            "position_qty": 0.0,
            "entry_price": None,
            "entry_time": None,
            "entry_notional": 0.0,
            "trade_pnl_accum": 0.0,
            "stop": None,
            "tp1": None,
            "tp2": None,
            "tp1_taken": False,
            "week_id_of_last_entry": None,
            "trade_id_seq": 0,
            "active_trade_id": None,
            "last_processed_open_time": None,
        }
        _, paper_events = process_candles(
            df,
            state,
            interval="1h",
            risk_per_trade=risk_per_trade,
            stop_mult=stop_mult,
            tp1_frac=tp1_frac,
            tp2_to_full=tp2_to_full,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            friday_cutoff_hour_utc=friday_cutoff_hour_utc,
        )
        paper_trades = _paper_trade_summaries(paper_events)

        self.assertEqual(len(bt_trades), len(paper_trades))
        self.assertEqual(len(bt_trades), 1)
        bt = bt_trades.iloc[0]
        pt = paper_trades[0]

        self.assertEqual(bt["side"], pt["side"])
        self.assertEqual(pd.Timestamp(bt["entry_time"]).isoformat(), pt["entry_time"])
        self.assertAlmostEqual(float(bt["entry"]), float(pt["entry_price"]), places=10)
        self.assertEqual(bt["reason"], pt["exit_reason"])
        self.assertEqual(pd.Timestamp(bt["exit_time"]).isoformat(), pt["exit_time"])
        self.assertAlmostEqual(float(bt["exit"]), float(pt["exit_price"]), places=10)
        self.assertAlmostEqual(float(bt["pnl"]), float(pt["pnl"]), places=8)


if __name__ == "__main__":
    unittest.main()
