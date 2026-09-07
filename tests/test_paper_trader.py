from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.features import add_monday_range
from src.paper_trader import process_candles, run_paper_cycle


def _make_binance_df(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    open_time = pd.to_datetime([r[0] for r in rows], utc=True)
    close_time = open_time + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1.0] * len(rows),
            "close_time": close_time,
            "quote_asset_volume": [1.0] * len(rows),
            "num_trades": [1] * len(rows),
            "taker_buy_base_asset_volume": [1.0] * len(rows),
            "taker_buy_quote_asset_volume": [1.0] * len(rows),
        }
    )


def _signal_then_entry_df() -> pd.DataFrame:
    return _make_binance_df(
        [
            ("2024-01-01 00:00:00+00:00", 100.0, 110.0, 90.0, 100.0),  # Monday
            ("2024-01-01 01:00:00+00:00", 100.0, 108.0, 92.0, 101.0),  # Monday
            ("2024-01-02 00:00:00+00:00", 95.0, 125.0, 60.0, 91.0),    # signal candle (extreme high/low)
            ("2024-01-02 01:00:00+00:00", 100.0, 101.0, 99.0, 100.0),  # entry candle
            ("2024-01-02 02:00:00+00:00", 100.0, 101.0, 99.0, 100.0),  # no exit
        ]
    )


def _stop_sequence_df() -> pd.DataFrame:
    return _make_binance_df(
        [
            ("2024-01-01 00:00:00+00:00", 100.0, 110.0, 90.0, 100.0),  # Monday
            ("2024-01-01 01:00:00+00:00", 100.0, 108.0, 92.0, 101.0),  # Monday
            ("2024-01-02 00:00:00+00:00", 95.0, 120.0, 89.0, 91.0),    # signal candle
            ("2024-01-02 01:00:00+00:00", 100.0, 101.0, 99.0, 100.0),  # entry candle
            ("2024-01-02 02:00:00+00:00", 100.0, 100.0, 69.0, 75.0),   # stop candle
        ]
    )


def _friday_df(include_23: bool) -> pd.DataFrame:
    rows: list[tuple[str, float, float, float, float]] = [
        ("2024-01-01 00:00:00+00:00", 100.0, 110.0, 90.0, 100.0),  # Monday for week features
        ("2024-01-01 01:00:00+00:00", 100.0, 108.0, 92.0, 101.0),
        ("2024-01-05 20:00:00+00:00", 100.0, 101.0, 99.0, 100.0),
        ("2024-01-05 21:00:00+00:00", 100.0, 101.0, 99.0, 100.0),
        ("2024-01-05 22:00:00+00:00", 100.0, 101.0, 99.0, 100.0),
    ]
    if include_23:
        rows.append(("2024-01-05 23:00:00+00:00", 100.0, 101.0, 99.0, 100.0))
    return _make_binance_df(rows)


def _open_position_state() -> dict:
    return {
        "cash": 10_000.0,
        "position_side": "LONG",
        "position_qty": 1.0,
        "entry_price": 100.0,
        "entry_time": "2024-01-04T00:00:00+00:00",
        "entry_notional": 100.0,
        "trade_pnl_accum": 0.0,
        "stop": 50.0,
        "tp1": 150.0,
        "tp2": 200.0,
        "tp1_taken": False,
        "week_id_of_last_entry": "2024-1",
        "trade_id_seq": 1,
        "active_trade_id": 1,
        "last_processed_open_time": None,
    }


def _minimal_raw(now_utc: pd.Timestamp) -> pd.DataFrame:
    rows = [
        (now_utc - pd.Timedelta(hours=2), 100.0, 101.0, 99.0, 100.0),
        (now_utc - pd.Timedelta(hours=1), 100.0, 101.0, 99.0, 100.0),
    ]
    df = pd.DataFrame(
        {
            "open_time": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1.0, 1.0],
            "close_time": [r[0] + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1) for r in rows],
            "quote_asset_volume": [1.0, 1.0],
            "num_trades": [1, 1],
            "taker_buy_base_asset_volume": [1.0, 1.0],
            "taker_buy_quote_asset_volume": [1.0, 1.0],
        }
    )
    return df


class PaperTraderP0Tests(unittest.TestCase):
    def test_no_same_candle_exit_after_entry_signal(self) -> None:
        raw = _signal_then_entry_df()
        df_feat = add_monday_range(raw.set_index("open_time")[["open", "high", "low", "close"]].sort_index())
        state = _open_position_state()
        state["position_side"] = None
        state["position_qty"] = 0.0
        state["entry_price"] = None
        state["entry_time"] = None
        state["entry_notional"] = 0.0
        state["stop"] = None
        state["tp1"] = None
        state["tp2"] = None
        state["week_id_of_last_entry"] = None
        state["active_trade_id"] = None

        _, logs = process_candles(
            df_feat,
            state,
            interval="1h",
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_frac=0.5,
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        signal_ts = pd.Timestamp("2024-01-02 00:00:00+00:00").isoformat()
        self.assertFalse(any((e["event"] in {"STOP", "TP1", "TP2", "FRIDAY"}) and (e["timestamp"] == signal_ts) for e in logs))

    def test_entry_aligns_with_next_candle_open(self) -> None:
        raw = _signal_then_entry_df()
        df_feat = add_monday_range(raw.set_index("open_time")[["open", "high", "low", "close"]].sort_index())
        state = _open_position_state()
        state["position_side"] = None
        state["position_qty"] = 0.0
        state["entry_price"] = None
        state["entry_time"] = None
        state["entry_notional"] = 0.0
        state["stop"] = None
        state["tp1"] = None
        state["tp2"] = None
        state["week_id_of_last_entry"] = None
        state["active_trade_id"] = None

        slip_bps = 10.0
        _, logs = process_candles(
            df_feat,
            state,
            interval="1h",
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_frac=0.5,
            fee_bps=0.0,
            slippage_bps=slip_bps,
        )
        entry_events = [e for e in logs if e["event"] == "ENTRY"]
        self.assertEqual(len(entry_events), 1)

        expected_entry_ts = pd.Timestamp("2024-01-02 01:00:00+00:00").isoformat()
        self.assertEqual(entry_events[0]["timestamp"], expected_entry_ts)
        expected_entry_px = 100.0 * (1.0 + slip_bps / 10_000.0)
        self.assertAlmostEqual(float(entry_events[0]["price"]), expected_entry_px, places=10)

    def test_no_premature_friday_exit_before_cutoff(self) -> None:
        raw = _friday_df(include_23=False)
        df_feat = add_monday_range(raw.set_index("open_time")[["open", "high", "low", "close"]].sort_index())
        state = _open_position_state()

        state_after, logs = process_candles(
            df_feat,
            state,
            interval="1h",
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_frac=0.5,
            fee_bps=0.0,
            slippage_bps=0.0,
            friday_cutoff_hour_utc=23,
        )

        self.assertFalse(any(e["event"] == "FRIDAY" for e in logs))
        self.assertEqual(state_after["position_side"], "LONG")
        self.assertGreater(float(state_after["position_qty"]), 0.0)

    def test_friday_exit_at_cutoff_hour(self) -> None:
        raw = _friday_df(include_23=True)
        df_feat = add_monday_range(raw.set_index("open_time")[["open", "high", "low", "close"]].sort_index())
        state = _open_position_state()

        state_after, logs = process_candles(
            df_feat,
            state,
            interval="1h",
            risk_per_trade=100.0,
            stop_mult=1.0,
            tp1_frac=0.5,
            fee_bps=0.0,
            slippage_bps=0.0,
            friday_cutoff_hour_utc=23,
        )

        self.assertTrue(any(e["event"] == "FRIDAY" for e in logs))
        self.assertIsNone(state_after["position_side"])
        self.assertEqual(float(state_after["position_qty"]), 0.0)

    def test_restart_idempotency_second_run_writes_zero_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = pathlib.Path(tmp) / "paper_state.json"
            log_path = pathlib.Path(tmp) / "paper_trades.csv"
            raw = _stop_sequence_df()

            with patch("src.paper_trader.download_klines", return_value=raw):
                out1 = run_paper_cycle(
                    symbol="BTCUSDT",
                    interval="1h",
                    dry_run=False,
                    use_cache=False,
                    backfill=True,
                    state_path=state_path,
                    log_path=log_path,
                )

            self.assertGreater(out1["logs_written"], 0)
            first_log_rows = len(pd.read_csv(log_path))
            self.assertGreater(first_log_rows, 0)

            with patch("src.paper_trader.download_klines", return_value=raw):
                out2 = run_paper_cycle(
                    symbol="BTCUSDT",
                    interval="1h",
                    dry_run=False,
                    use_cache=False,
                    backfill=True,
                    state_path=state_path,
                    log_path=log_path,
                )

            self.assertEqual(out2["logs_written"], 0)
            second_log_rows = len(pd.read_csv(log_path))
            self.assertEqual(second_log_rows, first_log_rows)

    def test_default_lookback_start_is_30_days_when_no_state(self) -> None:
        fixed_now = pd.Timestamp("2026-01-31T12:00:00+00:00")
        raw = _minimal_raw(fixed_now)
        with tempfile.TemporaryDirectory() as tmp:
            state_path = pathlib.Path(tmp) / "paper_state.json"
            log_path = pathlib.Path(tmp) / "paper_trades.csv"
            with patch("src.paper_trader.pd.Timestamp.now", return_value=fixed_now):
                with patch("src.paper_trader.download_klines", return_value=raw) as mock_dl:
                    run_paper_cycle(
                        symbol="BTCUSDT",
                        interval="1h",
                        dry_run=True,
                        use_cache=False,
                        state_path=state_path,
                        log_path=log_path,
                    )
            called = mock_dl.call_args.kwargs
            self.assertEqual(called["end"], fixed_now)
            self.assertEqual(called["start"], fixed_now - pd.Timedelta(days=30))

    def test_custom_lookback_days_changes_fetch_start(self) -> None:
        fixed_now = pd.Timestamp("2026-01-31T12:00:00+00:00")
        raw = _minimal_raw(fixed_now)
        with tempfile.TemporaryDirectory() as tmp:
            state_path = pathlib.Path(tmp) / "paper_state.json"
            log_path = pathlib.Path(tmp) / "paper_trades.csv"
            with patch("src.paper_trader.pd.Timestamp.now", return_value=fixed_now):
                with patch("src.paper_trader.download_klines", return_value=raw) as mock_dl:
                    run_paper_cycle(
                        symbol="BTCUSDT",
                        interval="1h",
                        lookback_days=365,
                        dry_run=True,
                        use_cache=False,
                        state_path=state_path,
                        log_path=log_path,
                    )
            called = mock_dl.call_args.kwargs
            self.assertEqual(called["start"], fixed_now - pd.Timedelta(days=365))
            self.assertEqual(called["end"], fixed_now)

    def test_start_end_override_lookback(self) -> None:
        fixed_now = pd.Timestamp("2026-01-31T12:00:00+00:00")
        raw = _minimal_raw(fixed_now)
        with tempfile.TemporaryDirectory() as tmp:
            state_path = pathlib.Path(tmp) / "paper_state.json"
            log_path = pathlib.Path(tmp) / "paper_trades.csv"
            with patch("src.paper_trader.pd.Timestamp.now", return_value=fixed_now):
                with patch("src.paper_trader.download_klines", return_value=raw) as mock_dl:
                    run_paper_cycle(
                        symbol="BTCUSDT",
                        interval="1h",
                        lookback_days=365,
                        start="2024-01-01",
                        end="2024-02-01",
                        dry_run=True,
                        use_cache=False,
                        state_path=state_path,
                        log_path=log_path,
                    )
            called = mock_dl.call_args.kwargs
            self.assertEqual(called["start"], pd.Timestamp("2024-01-01", tz="UTC"))
            self.assertEqual(called["end"], pd.Timestamp("2024-02-01", tz="UTC"))

    def test_invalid_date_handling(self) -> None:
        with self.assertRaises(ValueError):
            run_paper_cycle(
                symbol="BTCUSDT",
                interval="1h",
                start="not-a-date",
                dry_run=True,
                use_cache=False,
            )


if __name__ == "__main__":
    unittest.main()
