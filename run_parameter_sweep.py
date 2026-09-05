from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from run_backtest import build_summary, filter_ohlc_window, load_cached_ohlc
from src.backtest import backtest_sweep_fade
from src.features import add_monday_range


def _normalize_grid_tokens(raw: str | list[str], *, name: str) -> list[str]:
    parts = [raw] if isinstance(raw, str) else list(raw)
    tokens: list[str] = []
    for part in parts:
        for chunk in str(part).split(","):
            token = chunk.strip()
            if token:
                tokens.append(token)
    if not tokens:
        raise ValueError(f"{name} cannot be empty.")
    return tokens


def parse_float_grid(raw: str | list[str], *, name: str) -> list[float]:
    values: list[float] = []
    for token in _normalize_grid_tokens(raw, name=name):
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid float in {name}: {token}") from exc
    return values


def parse_int_grid(raw: str | list[str], *, name: str) -> list[int]:
    values: list[int] = []
    for token in _normalize_grid_tokens(raw, name=name):
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {name}: {token}") from exc
    return values


def parse_optional_float_grid(raw: str | list[str], *, name: str) -> list[float | None]:
    values: list[float | None] = []
    for token in _normalize_grid_tokens(raw, name=name):
        token = token.lower()
        if token in {"none", "null"}:
            values.append(None)
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid float/None in {name}: {token}") from exc
    return values


def parse_optional_int_grid(raw: str | list[str], *, name: str) -> list[int | None]:
    values: list[int | None] = []
    for token in _normalize_grid_tokens(raw, name=name):
        token = token.lower()
        if token in {"none", "null"}:
            values.append(None)
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer/None in {name}: {token}") from exc
    return values


def parse_direction_grid(raw: str | list[str]) -> list[str]:
    valid = {"both", "long_only", "short_only"}
    values = [v.lower() for v in _normalize_grid_tokens(raw, name="directions")]
    bad = [v for v in values if v not in valid]
    if bad:
        raise ValueError(f"Invalid direction values: {bad}")
    return values


def parse_bool_grid(raw: str | list[str], *, name: str) -> list[bool]:
    out: list[bool] = []
    for token in _normalize_grid_tokens(raw, name=name):
        t = token.strip().lower()
        if t in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif t in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean in {name}: {token}")
    return out


def generate_grid(
    *,
    risk_fractions: list[float],
    tp2_to_full_values: list[float],
    friday_cutoff_hours_utc: list[int],
    min_range_pcts: list[float | None] | None = None,
    max_range_pcts: list[float | None] | None = None,
    directions: list[str] | None = None,
    max_entry_days_utc: list[int | None] | None = None,
    max_entry_hours_utc: list[int | None] | None = None,
    tp1_range_fractions: list[float] | None = None,
    tp2_range_fraction_values: list[float] | None = None,
    tp1_close_fractions: list[float] | None = None,
    move_stop_to_breakeven_after_tp1_values: list[bool] | None = None,
    stop_modes: list[str] | None = None,
    stop_range_fractions: list[float] | None = None,
) -> list[dict[str, float | int | str | None]]:
    min_range_pcts = [None] if min_range_pcts is None else min_range_pcts
    max_range_pcts = [None] if max_range_pcts is None else max_range_pcts
    directions = ["both"] if directions is None else directions
    max_entry_days_utc = [None] if max_entry_days_utc is None else max_entry_days_utc
    max_entry_hours_utc = [None] if max_entry_hours_utc is None else max_entry_hours_utc
    tp1_range_fractions = [0.5] if tp1_range_fractions is None else tp1_range_fractions
    tp1_close_fractions = [0.5] if tp1_close_fractions is None else tp1_close_fractions
    move_stop_to_breakeven_after_tp1_values = [True] if move_stop_to_breakeven_after_tp1_values is None else move_stop_to_breakeven_after_tp1_values
    stop_modes = ["swept_boundary_offset"] if stop_modes is None else stop_modes
    stop_range_fractions = [1.0] if stop_range_fractions is None else stop_range_fractions

    tp2_pairs: list[tuple[float, float]] = []
    if tp2_range_fraction_values is not None:
        for rf in tp2_range_fraction_values:
            rf_f = float(rf)
            legacy = 1.0 if rf_f >= 1.0 else (2.0 * rf_f - 1.0)
            tp2_pairs.append((legacy, rf_f))
    else:
        for legacy in tp2_to_full_values:
            legacy_f = float(legacy)
            rf = 1.0 if legacy_f >= 1.0 else (0.5 + 0.5 * legacy_f)
            tp2_pairs.append((legacy_f, rf))

    combos = itertools.product(
        risk_fractions,
        tp2_pairs,
        friday_cutoff_hours_utc,
        min_range_pcts,
        max_range_pcts,
        directions,
        max_entry_days_utc,
        max_entry_hours_utc,
        tp1_range_fractions,
        tp1_close_fractions,
        move_stop_to_breakeven_after_tp1_values,
        stop_modes,
        stop_range_fractions,
    )
    return [
        {
            "risk_fraction": float(risk_fraction),
            "tp2_to_full": float(tp2_legacy),
            "tp2_range_fraction": float(tp2_range_fraction),
            "friday_cutoff_hour_utc": int(friday_cutoff_hour_utc),
            "min_range_pct": (None if min_range_pct is None else float(min_range_pct)),
            "max_range_pct": (None if max_range_pct is None else float(max_range_pct)),
            "direction": str(direction),
            "max_entry_day_utc": (None if max_entry_day_utc is None else int(max_entry_day_utc)),
            "max_entry_hour_utc": (None if max_entry_hour_utc is None else int(max_entry_hour_utc)),
            "tp1_range_fraction": float(tp1_range_fraction),
            "tp1_close_fraction": float(tp1_close_fraction),
            "move_stop_to_breakeven_after_tp1": bool(move_stop_to_breakeven_after_tp1),
            "stop_mode": str(stop_mode),
            "stop_range_fraction": float(stop_range_fraction),
        }
        for (
            risk_fraction,
            (tp2_legacy, tp2_range_fraction),
            friday_cutoff_hour_utc,
            min_range_pct,
            max_range_pct,
            direction,
            max_entry_day_utc,
            max_entry_hour_utc,
            tp1_range_fraction,
            tp1_close_fraction,
            move_stop_to_breakeven_after_tp1,
            stop_mode,
            stop_range_fraction,
        ) in combos
    ]


def sweep_score(
    *,
    total_return_pct: float,
    max_drawdown: float | None,
    num_trades: int,
    cost_drag_pct: float,
) -> float:
    drawdown = abs(float(max_drawdown)) if max_drawdown is not None else 0.0
    drawdown_penalty = drawdown * 120.0
    low_trade_penalty = max(0, 8 - int(num_trades)) * 1.5
    turnover_penalty = max(0, int(num_trades) - 80) * 0.15
    fee_drag_penalty = float(cost_drag_pct) * 1.0
    return float(total_return_pct) - drawdown_penalty - low_trade_penalty - turnover_penalty - fee_drag_penalty


def build_sweep_row(
    *,
    summary: dict[str, Any],
    symbol: str,
    interval: str,
    start: str | None,
    end: str | None,
    risk_fraction: float,
    tp1_range_fraction: float,
    tp2_range_fraction: float,
    tp2_to_full: float,
    friday_cutoff_hour_utc: int,
    fee_bps: float,
    slippage_bps: float,
    initial_cash: float,
    min_range_pct: float | None,
    max_range_pct: float | None,
    direction: str,
    max_entry_day_utc: int | None,
    max_entry_hour_utc: int | None,
    tp1_close_fraction: float = 0.5,
    move_stop_to_breakeven_after_tp1: bool = True,
    stop_mode: str = "swept_boundary_offset",
    stop_range_fraction: float = 1.0,
) -> dict[str, Any]:
    total_fees_paid = float(summary.get("total_fees_paid", 0.0))
    cost_drag_pct = (total_fees_paid / float(initial_cash) * 100.0) if initial_cash > 0 else 0.0
    return_after_costs_pct = float(summary["total_return_pct"])
    score = sweep_score(
        total_return_pct=float(summary["total_return_pct"]),
        max_drawdown=summary.get("max_drawdown"),
        num_trades=int(summary.get("num_trades", 0)),
        cost_drag_pct=cost_drag_pct,
    )
    return {
        "symbol": symbol,
        "interval": interval,
        "start": start if start is not None else summary.get("start_used"),
        "end": end if end is not None else summary.get("end_used"),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "risk_fraction": float(risk_fraction),
        "tp1_range_fraction": float(tp1_range_fraction),
        "tp2_range_fraction": float(tp2_range_fraction),
        "tp2_to_full": float(tp2_to_full),
        "friday_cutoff_hour_utc": int(friday_cutoff_hour_utc),
        "min_range_pct": min_range_pct,
        "max_range_pct": max_range_pct,
        "direction": direction,
        "max_entry_day_utc": max_entry_day_utc,
        "max_entry_hour_utc": max_entry_hour_utc,
        "tp1_close_fraction": float(tp1_close_fraction),
        "move_stop_to_breakeven_after_tp1": bool(move_stop_to_breakeven_after_tp1),
        "stop_mode": str(stop_mode),
        "stop_range_fraction": float(stop_range_fraction),
        "total_return_pct": float(summary["total_return_pct"]),
        "return_after_costs_pct": return_after_costs_pct,
        "max_drawdown": summary.get("max_drawdown"),
        "win_rate": summary.get("win_rate"),
        "num_trades": int(summary.get("num_trades", 0)),
        "average_trade_pnl": summary.get("average_trade_pnl"),
        "best_trade": summary.get("best_trade"),
        "worst_trade": summary.get("worst_trade"),
        "total_fees_paid": total_fees_paid,
        "cost_drag_pct": cost_drag_pct,
        "score": score,
    }


def write_sweep_outputs(
    *,
    output_root: str | Path,
    symbol: str,
    interval: str,
    results: pd.DataFrame,
    sweep_config: dict[str, Any],
) -> Path:
    root = Path(output_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / f"{symbol}_{interval}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "sweep_results.csv", index=False)
    with open(out_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(sweep_config, f, indent=2)
    return out_dir


def run_sweep(
    *,
    symbol: str,
    interval: str,
    start: str | None,
    end: str | None,
    output_dir: str | Path,
    fee_bps: float,
    slippage_bps: float,
    risk_fractions: list[float],
    tp2_to_full_values: list[float],
    friday_cutoff_hours_utc: list[int],
    min_range_pcts: list[float | None] | None = None,
    max_range_pcts: list[float | None] | None = None,
    directions: list[str] | None = None,
    max_entry_days_utc: list[int | None] | None = None,
    max_entry_hours_utc: list[int | None] | None = None,
    tp1_range_fractions: list[float] | None = None,
    tp2_range_fraction_values: list[float] | None = None,
    tp1_close_fractions: list[float] | None = None,
    move_stop_to_breakeven_after_tp1_values: list[bool] | None = None,
    stop_modes: list[str] | None = None,
    stop_range_fractions: list[float] | None = None,
    risk_base: str = "current_equity",
    max_leverage: float = 1.0,
    intrabar_policy: str = "conservative_stop_first",
) -> tuple[pd.DataFrame, Path]:
    initial_cash = 10_000.0

    ohlc = load_cached_ohlc(symbol, interval)
    ohlc = filter_ohlc_window(ohlc, start, end)
    if ohlc.empty:
        raise ValueError("No cached candles remain after applying start/end filters.")

    df_features = add_monday_range(ohlc)
    rows: list[dict[str, Any]] = []
    combos = generate_grid(
        risk_fractions=risk_fractions,
        tp2_to_full_values=tp2_to_full_values,
        friday_cutoff_hours_utc=friday_cutoff_hours_utc,
        min_range_pcts=min_range_pcts,
        max_range_pcts=max_range_pcts,
        directions=directions,
        max_entry_days_utc=max_entry_days_utc,
        max_entry_hours_utc=max_entry_hours_utc,
        tp1_range_fractions=tp1_range_fractions,
        tp2_range_fraction_values=tp2_range_fraction_values,
        tp1_close_fractions=tp1_close_fractions,
        move_stop_to_breakeven_after_tp1_values=move_stop_to_breakeven_after_tp1_values,
        stop_modes=stop_modes,
        stop_range_fractions=stop_range_fractions,
    )

    for combo in combos:
        risk_fraction = float(combo["risk_fraction"])
        tp2_to_full = float(combo["tp2_to_full"])
        friday_cutoff_hour_utc = int(combo["friday_cutoff_hour_utc"])
        min_range_pct = combo["min_range_pct"]
        max_range_pct = combo["max_range_pct"]
        direction = str(combo["direction"])
        max_entry_day_utc = combo["max_entry_day_utc"]
        max_entry_hour_utc = combo["max_entry_hour_utc"]
        tp1_range_fraction = float(combo["tp1_range_fraction"])
        tp2_range_fraction = float(combo["tp2_range_fraction"])
        tp1_close_fraction = float(combo["tp1_close_fraction"])
        move_stop_to_breakeven_after_tp1 = bool(combo["move_stop_to_breakeven_after_tp1"])
        stop_mode = str(combo["stop_mode"])
        stop_range_fraction = float(combo["stop_range_fraction"])
        if min_range_pct is not None and max_range_pct is not None and float(min_range_pct) > float(max_range_pct):
            continue

        risk_per_trade = initial_cash * risk_fraction
        df_out, trades = backtest_sweep_fade(
            df_features,
            initial_capital=initial_cash,
            risk_per_trade=risk_per_trade,
            risk_fraction=risk_fraction,
            risk_base=risk_base,
            max_leverage=max_leverage,
            intrabar_policy=intrabar_policy,
            stop_mult=1.0,
            tp1_frac=0.5,
            tp2_to_full=tp2_to_full,
            tp1_range_fraction=tp1_range_fraction,
            tp2_range_fraction=tp2_range_fraction,
            exit_friday_close=True,
            friday_cutoff_hour_utc=friday_cutoff_hour_utc,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_range_pct=min_range_pct,
            max_range_pct=max_range_pct,
            direction=direction,
            max_entry_day_utc=max_entry_day_utc,
            max_entry_hour_utc=max_entry_hour_utc,
            tp1_close_fraction=tp1_close_fraction,
            stop_mode=stop_mode,
            stop_range_fraction=stop_range_fraction,
            move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
        )
        summary = build_summary(
            strategy="current_monday_range",
            signal_count=int(df_out.attrs.get("signal_count", 0)),
            exit_style="partial_tp2",
            single_target_level=None,
            symbol=symbol,
            interval=interval,
            initial_cash=initial_cash,
            df_out=df_out,
            trades=trades,
            start_used=start,
            end_used=end,
        )
        rows.append(
            build_sweep_row(
                summary=summary,
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
                risk_fraction=risk_fraction,
                tp1_range_fraction=tp1_range_fraction,
                tp2_range_fraction=tp2_range_fraction,
                tp2_to_full=tp2_to_full,
                friday_cutoff_hour_utc=friday_cutoff_hour_utc,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                initial_cash=initial_cash,
                min_range_pct=min_range_pct,
                max_range_pct=max_range_pct,
                direction=direction,
                max_entry_day_utc=max_entry_day_utc,
                max_entry_hour_utc=max_entry_hour_utc,
                tp1_close_fraction=tp1_close_fraction,
                move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
                stop_mode=stop_mode,
                stop_range_fraction=stop_range_fraction,
            )
        )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            by=["score", "return_after_costs_pct", "cost_drag_pct", "num_trades"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)

    sweep_config = {
        "symbol": symbol,
        "interval": interval,
        "start": start,
        "end": end,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "risk_fractions": risk_fractions,
        "risk_base": risk_base,
        "max_leverage": max_leverage,
        "intrabar_policy": intrabar_policy,
        "accounting_version": "marked_equity_v1",
        "tp2_to_full_values": tp2_to_full_values,
        "friday_cutoff_hours_utc": friday_cutoff_hours_utc,
        "min_range_pcts": ([None] if min_range_pcts is None else min_range_pcts),
        "max_range_pcts": ([None] if max_range_pcts is None else max_range_pcts),
        "directions": (["both"] if directions is None else directions),
        "max_entry_days_utc": ([None] if max_entry_days_utc is None else max_entry_days_utc),
        "max_entry_hours_utc": ([None] if max_entry_hours_utc is None else max_entry_hours_utc),
        "tp1_range_fractions": ([0.5] if tp1_range_fractions is None else tp1_range_fractions),
        "tp2_range_fraction_values": (
            None
            if tp2_range_fraction_values is None
            else tp2_range_fraction_values
        ),
        "tp1_close_fractions": ([0.5] if tp1_close_fractions is None else tp1_close_fractions),
        "move_stop_to_breakeven_after_tp1_values": ([True] if move_stop_to_breakeven_after_tp1_values is None else move_stop_to_breakeven_after_tp1_values),
        "stop_modes": (["swept_boundary_offset"] if stop_modes is None else stop_modes),
        "stop_range_fractions": ([1.0] if stop_range_fractions is None else stop_range_fractions),
        "combinations": len(combos),
    }
    out_dir = write_sweep_outputs(
        output_root=output_dir,
        symbol=symbol,
        interval=interval,
        results=results,
        sweep_config=sweep_config,
    )
    return results, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Monday range parameter sweep on cached Binance candles.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--output_dir", default="data/sweeps")
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--risk_fractions", nargs="+", default=["0.005,0.01,0.02"])
    ap.add_argument("--risk_base", choices=["current_equity", "initial_capital"], default="current_equity")
    ap.add_argument("--max_leverage", type=float, default=1.0)
    ap.add_argument("--intrabar_policy", choices=["conservative_stop_first", "target_first"], default="conservative_stop_first")
    ap.add_argument("--tp2_to_full_values", nargs="+", default=["0.5,0.75,1.0"])
    ap.add_argument("--friday_cutoff_hours_utc", nargs="+", default=["20,21,22,23"])
    ap.add_argument("--min_range_pcts", nargs="+", default=["none"])
    ap.add_argument("--max_range_pcts", nargs="+", default=["none"])
    ap.add_argument("--directions", nargs="+", default=["both"])
    ap.add_argument("--max_entry_days_utc", nargs="+", default=["none"])
    ap.add_argument("--max_entry_hours_utc", nargs="+", default=["none"])
    ap.add_argument("--tp1_range_fractions", nargs="+", default=["0.5"])
    ap.add_argument("--tp2_range_fraction_values", nargs="+", default=None)
    ap.add_argument("--tp1_close_fractions", nargs="+", default=["0.5"])
    ap.add_argument("--move_stop_to_breakeven_after_tp1_values", nargs="+", default=["true"])
    ap.add_argument("--stop_modes", nargs="+", default=["swept_boundary_offset"])
    ap.add_argument("--stop_range_fractions", nargs="+", default=["1.0"])
    args = ap.parse_args()

    risk_fractions = parse_float_grid(args.risk_fractions, name="risk_fractions")
    tp2_to_full_values = parse_float_grid(args.tp2_to_full_values, name="tp2_to_full_values")
    friday_cutoff_hours_utc = parse_int_grid(args.friday_cutoff_hours_utc, name="friday_cutoff_hours_utc")
    min_range_pcts = parse_optional_float_grid(args.min_range_pcts, name="min_range_pcts")
    max_range_pcts = parse_optional_float_grid(args.max_range_pcts, name="max_range_pcts")
    directions = parse_direction_grid(args.directions)
    max_entry_days_utc = parse_optional_int_grid(args.max_entry_days_utc, name="max_entry_days_utc")
    max_entry_hours_utc = parse_optional_int_grid(args.max_entry_hours_utc, name="max_entry_hours_utc")
    tp1_range_fractions = parse_float_grid(args.tp1_range_fractions, name="tp1_range_fractions")
    tp2_range_fraction_values = (
        None if args.tp2_range_fraction_values is None else parse_float_grid(args.tp2_range_fraction_values, name="tp2_range_fraction_values")
    )
    tp1_close_fractions = parse_float_grid(args.tp1_close_fractions, name="tp1_close_fractions")
    move_stop_to_breakeven_after_tp1_values = parse_bool_grid(
        args.move_stop_to_breakeven_after_tp1_values,
        name="move_stop_to_breakeven_after_tp1_values",
    )
    stop_modes = [s.strip() for s in _normalize_grid_tokens(args.stop_modes, name="stop_modes")]
    valid_stop_modes = {"swept_boundary", "swept_boundary_offset", "entry_fixed_pct", "opposite_boundary", "range_fraction", "fixed_pct"}
    bad_stop_modes = [s for s in stop_modes if s not in valid_stop_modes]
    if bad_stop_modes:
        raise ValueError(f"Invalid stop_modes: {bad_stop_modes}")
    stop_range_fractions = parse_float_grid(args.stop_range_fractions, name="stop_range_fractions")

    if float(args.fee_bps) == 0.0 or float(args.slippage_bps) == 0.0:
        print("WARNING: fee_bps or slippage_bps is zero. Sweep may overestimate deployable performance.")

    results, out_dir = run_sweep(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        risk_fractions=risk_fractions,
        tp2_to_full_values=tp2_to_full_values,
        friday_cutoff_hours_utc=friday_cutoff_hours_utc,
        min_range_pcts=min_range_pcts,
        max_range_pcts=max_range_pcts,
        directions=directions,
        max_entry_days_utc=max_entry_days_utc,
        max_entry_hours_utc=max_entry_hours_utc,
        tp1_range_fractions=tp1_range_fractions,
        tp2_range_fraction_values=tp2_range_fraction_values,
        tp1_close_fractions=tp1_close_fractions,
        move_stop_to_breakeven_after_tp1_values=move_stop_to_breakeven_after_tp1_values,
        stop_modes=stop_modes,
        stop_range_fractions=stop_range_fractions,
        risk_base=args.risk_base,
        max_leverage=float(args.max_leverage),
        intrabar_policy=args.intrabar_policy,
    )

    print("=== Parameter sweep complete ===")
    print(f"Saved outputs under: {out_dir}")
    print(f"Total combinations: {len(results)}")
    if not results.empty:
        print("Top 5 parameter sets:")
        cols = [
            "risk_fraction",
            "tp2_to_full",
            "friday_cutoff_hour_utc",
            "min_range_pct",
            "max_range_pct",
            "direction",
            "max_entry_day_utc",
            "max_entry_hour_utc",
            "tp1_range_fraction",
            "tp2_range_fraction",
            "tp1_close_fraction",
            "move_stop_to_breakeven_after_tp1",
            "stop_mode",
            "stop_range_fraction",
            "total_return_pct",
            "return_after_costs_pct",
            "max_drawdown",
            "cost_drag_pct",
            "num_trades",
            "score",
        ]
        print(results[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
