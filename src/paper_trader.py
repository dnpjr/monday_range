from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

from .binance_data import download_klines, INTERVAL_MS
from .features import add_monday_range
from .signals import sweep_rejection_signal
from .execution_helpers import (
    apply_slippage,
    calculate_entry_fee,
    calculate_exit_fee,
    calculate_tp_levels,
    is_stop_hit,
    is_tp_hit,
    is_friday_cutoff_bar,
)


def _default_state(initial_cash: float) -> dict[str, Any]:
    return {
        "cash": float(initial_cash),
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


def _ensure_state_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_state(state_path: str | Path, initial_cash: float) -> dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        return _default_state(initial_cash)

    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)

    defaults = _default_state(initial_cash)
    for key, value in defaults.items():
        state.setdefault(key, value)
    return state


def save_state(state_path: str | Path, state: dict[str, Any]) -> None:
    path = Path(state_path)
    _ensure_state_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def append_trade_logs(log_path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def _safe_float(x: Any) -> float:
    return float(x) if x is not None else 0.0


def _to_utc_timestamp(value: Any, field_name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _trade_log_row(
    *,
    ts: pd.Timestamp,
    event: str,
    side: str,
    price: float,
    qty: float,
    fee: float,
    realized_pnl: float,
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": ts.isoformat(),
        "event": event,
        "side": side,
        "price": float(price),
        "qty": float(qty),
        "fee": float(fee),
        "realized_pnl": float(realized_pnl),
        "cash_after": float(state["cash"]),
        "position_qty_after": float(state["position_qty"]),
        "trade_id": state["active_trade_id"],
        "trade_pnl_total": float(state["trade_pnl_accum"]),
    }


def process_candles(
    df_feat: pd.DataFrame,
    state: dict[str, Any],
    *,
    interval: str,
    risk_per_trade: float,
    stop_mult: float,
    tp1_frac: float,
    fee_bps: float,
    slippage_bps: float,
    tp2_to_full: float = 1.0,
    friday_cutoff_hour_utc: int = 23,
    tp1_at_mid: bool = True,
    exit_friday_close: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Process completed candles for paper trading.

    Convention:
    - Signal is detected on candle N (completed, from `prev`).
    - Entry executes at candle N+1 open (current `row["open"]`), with slippage/fees.
    - Exits are evaluated on the entry candle and later candles (intrabar high/low checks).
    """
    logs: list[dict[str, Any]] = []
    if df_feat.empty:
        return state, logs

    interval_delta = pd.Timedelta(milliseconds=INTERVAL_MS[interval])

    last_processed = state.get("last_processed_open_time")
    last_processed_ts = pd.Timestamp(last_processed) if last_processed else None

    for i in range(1, len(df_feat)):
        prev = df_feat.iloc[i - 1]
        row = df_feat.iloc[i]
        ts = row.name

        if last_processed_ts is not None and ts <= last_processed_ts:
            continue

        ts_utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        bar_open = ts_utc
        bar_close = ts_utc + interval_delta
        friday_cutoff_hit = is_friday_cutoff_bar(bar_open, bar_close, friday_cutoff_hour_utc=friday_cutoff_hour_utc)
        week_id = f"{int(row['iso_year'])}-{int(row['iso_week'])}"

        mon_high = row["mon_high"]
        mon_low = row["mon_low"]
        mon_mid = row["mon_mid"]
        rng = row["mon_range"]

        has_position = state["position_side"] in {"LONG", "SHORT"} and state["position_qty"] > 0

        if (not has_position) and row["is_tradeable"] and (row["weekday"] < 4):
            if state.get("week_id_of_last_entry") != week_id:
                if pd.notna(mon_high) and pd.notna(mon_low) and rng > 0:
                    signal_side = sweep_rejection_signal(prev, mon_low=mon_low, mon_high=mon_high)
                    if signal_side is not None:
                        # Entry is next-candle open after signal candle.
                        entry_raw = float(row["open"])
                        entry = apply_slippage(entry_raw, signal_side, action="entry", slippage_bps=slippage_bps)

                        if signal_side == "LONG":
                            stop = float(mon_low - stop_mult * rng)
                            tp1, tp2 = calculate_tp_levels(signal_side, mon_low=float(mon_low), mon_mid=float(mon_mid), mon_high=float(mon_high), tp2_to_full=tp2_to_full)
                            dist = abs(entry - stop)
                        else:
                            stop = float(mon_high + stop_mult * rng)
                            tp1, tp2 = calculate_tp_levels(signal_side, mon_low=float(mon_low), mon_mid=float(mon_mid), mon_high=float(mon_high), tp2_to_full=tp2_to_full)
                            dist = abs(stop - entry)

                        qty = (risk_per_trade / dist) if dist > 0 else 0.0
                        if qty > 0:
                            entry_notional = abs(entry * qty)
                            entry_fee = calculate_entry_fee(entry, qty, fee_bps=fee_bps)
                            state["cash"] -= entry_fee

                            state["position_side"] = signal_side
                            state["position_qty"] = qty
                            state["entry_price"] = entry
                            state["entry_time"] = ts.isoformat()
                            state["entry_notional"] = entry_notional
                            state["trade_pnl_accum"] = -entry_fee
                            state["stop"] = stop
                            state["tp1"] = tp1
                            state["tp2"] = tp2
                            state["tp1_taken"] = False
                            state["week_id_of_last_entry"] = week_id
                            state["trade_id_seq"] = int(state.get("trade_id_seq", 0)) + 1
                            state["active_trade_id"] = state["trade_id_seq"]

                            logs.append(
                                _trade_log_row(
                                    ts=ts,
                                    event="ENTRY",
                                    side=signal_side,
                                    price=entry,
                                    qty=qty,
                                    fee=entry_fee,
                                    realized_pnl=-entry_fee,
                                    state=state,
                                )
                            )

        has_position = state["position_side"] in {"LONG", "SHORT"} and state["position_qty"] > 0
        if has_position:
            side = state["position_side"]
            qty = float(state["position_qty"])
            entry = float(state["entry_price"])

            def exit_fill(exit_px_raw: float, exit_qty: float, event: str) -> None:
                exit_px = apply_slippage(exit_px_raw, side, action="exit", slippage_bps=slippage_bps)
                gross = (exit_px - entry) * exit_qty if side == "LONG" else (entry - exit_px) * exit_qty
                fee = calculate_exit_fee(exit_px, exit_qty, fee_bps=fee_bps)
                realized = gross - fee
                state["cash"] += realized
                state["trade_pnl_accum"] += realized
                state["position_qty"] = max(0.0, float(state["position_qty"]) - exit_qty)
                logs.append(
                    _trade_log_row(
                        ts=ts,
                        event=event,
                        side=side,
                        price=exit_px,
                        qty=exit_qty,
                        fee=fee,
                        realized_pnl=realized,
                        state=state,
                    )
                )

            hit_stop = is_stop_hit(side, candle_high=float(row["high"]), candle_low=float(row["low"]), stop_price=float(state["stop"]))
            if hit_stop:
                exit_fill(float(state["stop"]), qty, "STOP")
                state["position_side"] = None
                state["entry_price"] = None
                state["entry_time"] = None
                state["entry_notional"] = 0.0
                state["stop"] = None
                state["tp1"] = None
                state["tp2"] = None
                state["tp1_taken"] = False
                state["active_trade_id"] = None

            has_position = state["position_side"] in {"LONG", "SHORT"} and state["position_qty"] > 0
            if has_position and tp1_at_mid and (not bool(state["tp1_taken"])):
                hit_tp1 = is_tp_hit(side, candle_high=float(row["high"]), candle_low=float(row["low"]), tp_price=float(state["tp1"]))
                if hit_tp1:
                    exit_qty = float(state["position_qty"]) * tp1_frac
                    exit_fill(float(state["tp1"]), exit_qty, "TP1")
                    state["tp1_taken"] = True
                    state["stop"] = state["entry_price"]

            has_position = state["position_side"] in {"LONG", "SHORT"} and state["position_qty"] > 0
            if has_position:
                hit_tp2 = is_tp_hit(side, candle_high=float(row["high"]), candle_low=float(row["low"]), tp_price=float(state["tp2"]))
                if hit_tp2:
                    exit_fill(float(state["tp2"]), float(state["position_qty"]), "TP2")
                    state["position_side"] = None
                    state["entry_price"] = None
                    state["entry_time"] = None
                    state["entry_notional"] = 0.0
                    state["stop"] = None
                    state["tp1"] = None
                    state["tp2"] = None
                    state["tp1_taken"] = False
                    state["active_trade_id"] = None

            has_position = state["position_side"] in {"LONG", "SHORT"} and state["position_qty"] > 0
            if has_position and exit_friday_close and friday_cutoff_hit:
                exit_fill(float(row["close"]), float(state["position_qty"]), "FRIDAY")
                state["position_side"] = None
                state["entry_price"] = None
                state["entry_time"] = None
                state["entry_notional"] = 0.0
                state["stop"] = None
                state["tp1"] = None
                state["tp2"] = None
                state["tp1_taken"] = False
                state["active_trade_id"] = None

        state["last_processed_open_time"] = ts.isoformat()

    return state, logs


def run_paper_cycle(
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    initial_cash: float = 10_000.0,
    risk_per_trade: float = 100.0,
    stop_mult: float = 1.0,
    tp1_frac: float = 0.5,
    tp2_to_full: float = 1.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    lookback_days: int = 30,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    state_path: str | Path = "data/paper_state.json",
    log_path: str | Path = "data/paper_trades.csv",
    use_cache: bool = False,
    dry_run: bool = True,
    backfill: bool = False,
    friday_cutoff_hour_utc: int = 23,
) -> dict[str, Any]:
    state_file_exists = Path(state_path).exists()
    state = load_state(state_path, initial_cash=initial_cash)

    now_utc = pd.Timestamp.now(tz="UTC")
    end_utc = _to_utc_timestamp(end, "end") if end is not None else now_utc

    # Explicit bounds override lookback-derived defaults.
    if (start is not None) or (end is not None):
        start_utc = _to_utc_timestamp(start, "start") if start is not None else (end_utc - pd.Timedelta(days=lookback_days))
    else:
        last_processed = state.get("last_processed_open_time")
        if last_processed:
            start_utc = pd.Timestamp(last_processed).tz_convert("UTC") - pd.Timedelta(days=8)
        else:
            start_utc = end_utc - pd.Timedelta(days=lookback_days)

    if end_utc <= start_utc:
        raise ValueError("end must be later than start")

    raw = download_klines(
        symbol=symbol,
        interval=interval,
        start=start_utc,
        end=end_utc,
        use_cache=use_cache,
        cache_format="parquet",
    )
    if raw.empty:
        return {"message": "No data returned", "state": state, "logs": []}

    completed = raw[raw["close_time"] <= now_utc].copy()
    if completed.empty:
        return {"message": "No completed candles available", "state": state, "logs": []}

    first_forward_run = (not backfill) and (not state_file_exists) and (state.get("last_processed_open_time") is None)
    if first_forward_run:
        latest_open = completed["open_time"].iloc[-1]
        state["last_processed_open_time"] = latest_open.isoformat()
        result = {
            "dry_run": dry_run,
            "mode": "forward",
            "symbol": symbol,
            "interval": interval,
            "candles_fetched": int(len(completed)),
            "latest_completed_open_time": completed["open_time"].iloc[-1].isoformat(),
            "latest_completed_close_time": completed["close_time"].iloc[-1].isoformat(),
            "cash_before": float(state["cash"]),
            "cash_after": float(state["cash"]),
            "equity_after": float(state["cash"]),
            "position_side": state.get("position_side"),
            "position_qty": _safe_float(state.get("position_qty")),
            "last_processed_open_time": state.get("last_processed_open_time"),
            "logs_written": 0,
            "events": [],
            "message": "Initialized forward paper trading from latest completed candle.",
        }
        if not dry_run:
            save_state(state_path, state)
        return result

    df = completed.set_index("open_time")[["open", "high", "low", "close"]].sort_index()
    df_feat = add_monday_range(df)

    state_before = dict(state)
    state_after, logs = process_candles(
        df_feat,
        state,
        interval=interval,
        risk_per_trade=risk_per_trade,
        stop_mult=stop_mult,
        tp1_frac=tp1_frac,
        tp2_to_full=tp2_to_full,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        friday_cutoff_hour_utc=friday_cutoff_hour_utc,
    )

    if not dry_run:
        save_state(state_path, state_after)
        append_trade_logs(log_path, logs)

    last_close = float(df_feat["close"].iloc[-1])
    side = state_after.get("position_side")
    qty = _safe_float(state_after.get("position_qty"))
    entry = state_after.get("entry_price")
    unrealized = 0.0
    if side == "LONG" and qty > 0 and entry is not None:
        unrealized = (last_close - float(entry)) * qty
    elif side == "SHORT" and qty > 0 and entry is not None:
        unrealized = (float(entry) - last_close) * qty

    equity = float(state_after["cash"]) + unrealized
    interval_ms = INTERVAL_MS[interval]
    freshness_sec = max(0.0, (now_utc - completed["close_time"].iloc[-1]).total_seconds())

    return {
        "dry_run": dry_run,
        "mode": "backfill" if backfill else "forward",
        "symbol": symbol,
        "interval": interval,
        "candles_fetched": int(len(completed)),
        "latest_completed_open_time": completed["open_time"].iloc[-1].isoformat(),
        "latest_completed_close_time": completed["close_time"].iloc[-1].isoformat(),
        "latest_candle_age_seconds": freshness_sec,
        "cash_before": float(state_before["cash"]),
        "cash_after": float(state_after["cash"]),
        "equity_after": equity,
        "position_side": state_after.get("position_side"),
        "position_qty": qty,
        "last_processed_open_time": state_after.get("last_processed_open_time"),
        "logs_written": 0 if dry_run else len(logs),
        "events": logs,
        "expected_candle_seconds": interval_ms / 1000.0,
        "message": "Backfill simulation mode: processed historical candles."
        if backfill
        else "Forward paper mode: processed candles newer than last processed candle.",
    }
