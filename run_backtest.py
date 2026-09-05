from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.features import add_monday_range
from src.backtest import backtest_sweep_fade
from src.metrics import equity_metrics, trade_metrics
from src.research import analyze_weekly_sweep_signals


def bars_per_year_from_interval(interval: str) -> float:
    interval = interval.lower().strip()
    if interval.endswith("h"):
        return 365.0 * 24.0 / float(interval[:-1])
    if interval.endswith("d"):
        return 365.0 / float(interval[:-1])
    if interval.endswith("m"):
        return 365.0 * 24.0 * 60.0 / float(interval[:-1])
    return 365.0 * 24.0


def _parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _end_exclusive(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = _parse_utc(value)
    if len(value) == 10:  # YYYY-MM-DD
        ts = ts + pd.Timedelta(days=1)
    return ts


def load_cached_ohlc(symbol: str, interval: str, cache_dir: str | Path = "data/binance") -> pd.DataFrame:
    p = Path(cache_dir) / f"{symbol}_{interval}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Cached file not found: {p}")
    df = pd.read_csv(p)
    if "open_time" not in df.columns:
        raise ValueError(f"Cache file missing open_time column: {p}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"Cache file missing {c} column: {p}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df.dropna(subset=["open_time", "open", "high", "low", "close"]).copy()
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    return out.set_index("open_time")[["open", "high", "low", "close"]]


def filter_ohlc_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out[out.index >= _parse_utc(start)]
    end_excl = _end_exclusive(end)
    if end_excl is not None:
        out = out[out.index < end_excl]
    return out


def build_summary(
    *,
    strategy: str,
    signal_count: int | None,
    exit_style: str,
    single_target_level: str | None,
    symbol: str,
    interval: str,
    initial_cash: float,
    df_out: pd.DataFrame,
    trades: pd.DataFrame,
    start_used: str | None,
    end_used: str | None,
) -> dict[str, Any]:
    if df_out.empty:
        raise ValueError("No rows available after filtering.")

    bars_per_year = bars_per_year_from_interval(interval)
    em = equity_metrics(df_out["equity"], bars_per_year=bars_per_year)
    tm = trade_metrics(trades)
    final_equity = float(df_out["equity"].dropna().iloc[-1]) if len(df_out["equity"].dropna()) else float(initial_cash)
    total_fees = float(df_out["fees_paid"].sum()) if "fees_paid" in df_out.columns else 0.0

    best_trade = float(trades["pnl"].max()) if len(trades) else None
    worst_trade = float(trades["pnl"].min()) if len(trades) else None
    avg_trade_pnl = float(trades["pnl"].mean()) if len(trades) else None

    return {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "signal_count": (None if signal_count is None else int(signal_count)),
        "exit_style": exit_style,
        "single_target_level": single_target_level,
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "final_cash": final_equity,
        "total_return_pct": float((final_equity / float(initial_cash) - 1.0) * 100.0),
        "num_trades": int(tm.get("num_trades", 0)),
        "win_rate": tm.get("win_rate"),
        "average_trade_pnl": avg_trade_pnl,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "total_fees_paid": total_fees,
        "max_drawdown": em.get("max_drawdown"),
        "start_used": str(df_out.index.min()) if start_used is None else start_used,
        "end_used": str(df_out.index.max()) if end_used is None else end_used,
    }


def write_outputs(
    *,
    output_root: str | Path,
    symbol: str,
    interval: str,
    trades: pd.DataFrame,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    root = Path(output_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / f"{symbol}_{interval}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trades.to_csv(out_dir / "trades.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return out_dir


def run_backtest_cached(
    *,
    strategy: str = "current_monday_range",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str | None = None,
    end: str | None = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    risk_fraction: float = 0.01,
    tp1_range_fraction: float | None = 0.5,
    tp2_range_fraction: float | None = 1.0,
    tp2_to_full: float = 1.0,
    friday_cutoff_hour_utc: int = 23,
    min_range_pct: float | None = None,
    max_range_pct: float | None = None,
    direction: str = "both",
    max_entry_day_utc: int | None = None,
    max_entry_hour_utc: int | None = None,
    sma_period: int | None = None,
    tp1_to_mid: float = 1.0,
    tp1_close_fraction: float = 0.5,
    stop_mode: str | None = None,
    stop_range_fraction: float | None = None,
    stop_pct: float = 0.01,
    move_stop_to_breakeven_after_tp1: bool = True,
    breakeven_includes_fees: bool = False,
    exit_style: str = "partial_tp2",
    single_target_mode: bool | None = None,  # legacy alias
    single_target_level: str = "tp2",
    output_dir: str | Path = "data/backtests",
) -> dict[str, Any]:
    valid_strategies = {"current_monday_range", "sweep_retest"}
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {sorted(valid_strategies)}")
    valid_exit_styles = {"partial_tp2", "single_target"}
    if exit_style not in valid_exit_styles:
        raise ValueError(f"exit_style must be one of {sorted(valid_exit_styles)}")
    valid_single_target_levels = {"tp1", "tp2"}
    if single_target_level not in valid_single_target_levels:
        raise ValueError(f"single_target_level must be one of {sorted(valid_single_target_levels)}")

    if single_target_mode is None:
        single_target_mode_resolved = bool(exit_style == "single_target")
    else:
        single_target_mode_resolved = bool(single_target_mode)
        if single_target_mode_resolved and exit_style != "single_target":
            exit_style = "single_target"

    if stop_mode is None:
        stop_mode_resolved = "range_fraction" if strategy == "sweep_retest" else "opposite_boundary"
    else:
        stop_mode_resolved = stop_mode
    if stop_range_fraction is None:
        stop_range_fraction_resolved = 0.5 if strategy == "sweep_retest" else 1.0
    else:
        stop_range_fraction_resolved = float(stop_range_fraction)

    initial_cash = 10_000.0
    risk_per_trade = initial_cash * float(risk_fraction)

    ohlc = load_cached_ohlc(symbol, interval)
    ohlc = filter_ohlc_window(ohlc, start, end)
    if ohlc.empty:
        raise ValueError("No cached candles remain after applying start/end filters.")

    df = add_monday_range(ohlc)
    legacy_signal_count = None
    if strategy == "sweep_retest":
        legacy_signal_count = int(len(analyze_weekly_sweep_signals(df)))
    df_out, trades = backtest_sweep_fade(
        df,
        strategy=strategy,
        initial_capital=initial_cash,
        risk_per_trade=risk_per_trade,
        stop_mult=1.0,
        tp1_frac=0.5,
        tp2_to_full=float(tp2_to_full),
        tp1_range_fraction=tp1_range_fraction,
        tp2_range_fraction=tp2_range_fraction,
        exit_friday_close=True,
        friday_cutoff_hour_utc=int(friday_cutoff_hour_utc),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        min_range_pct=min_range_pct,
        max_range_pct=max_range_pct,
        direction=direction,
        max_entry_day_utc=max_entry_day_utc,
        max_entry_hour_utc=max_entry_hour_utc,
        sma_period=sma_period,
        tp1_to_mid=float(tp1_to_mid),
        tp1_close_fraction=float(tp1_close_fraction),
        stop_mode=stop_mode_resolved,
        stop_range_fraction=float(stop_range_fraction_resolved),
        stop_pct=float(stop_pct),
        move_stop_to_breakeven_after_tp1=bool(move_stop_to_breakeven_after_tp1),
        breakeven_includes_fees=bool(breakeven_includes_fees),
        single_target_mode=bool(single_target_mode_resolved),
        single_target_level=single_target_level,
    )

    if legacy_signal_count is None:
        legacy_signal_count = int(df_out.attrs.get("signal_count", 0))

    summary = build_summary(
        strategy=strategy,
        signal_count=legacy_signal_count,
        exit_style=exit_style,
        single_target_level=(single_target_level if single_target_mode_resolved else None),
        symbol=symbol,
        interval=interval,
        initial_cash=initial_cash,
        df_out=df_out,
        trades=trades,
        start_used=start,
        end_used=end,
    )
    config = {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "start": start,
        "end": end,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "risk_fraction": risk_fraction,
        "risk_per_trade": risk_per_trade,
        "tp1_range_fraction": tp1_range_fraction,
        "tp2_range_fraction": tp2_range_fraction,
        "tp2_to_full": tp2_to_full,
        "friday_cutoff_hour_utc": friday_cutoff_hour_utc,
        "min_range_pct": min_range_pct,
        "max_range_pct": max_range_pct,
        "direction": direction,
        "max_entry_day_utc": max_entry_day_utc,
        "max_entry_hour_utc": max_entry_hour_utc,
        "sma_period": sma_period,
        "tp1_to_mid": tp1_to_mid,
        "tp1_close_fraction": tp1_close_fraction,
        "stop_mode": stop_mode_resolved,
        "stop_range_fraction": stop_range_fraction_resolved,
        "stop_pct": stop_pct,
        "move_stop_to_breakeven_after_tp1": move_stop_to_breakeven_after_tp1,
        "breakeven_includes_fees": breakeven_includes_fees,
        "exit_style": exit_style,
        "single_target_mode": single_target_mode_resolved,
        "single_target_level": single_target_level,
        "signal_count": legacy_signal_count,
        "initial_cash": initial_cash,
    }
    out_dir = write_outputs(
        output_root=output_dir,
        symbol=symbol,
        interval=interval,
        trades=trades,
        summary=summary,
        config=config,
    )
    return {
        "summary": summary,
        "config": config,
        "out_dir": str(out_dir),
        "trades_count": int(len(trades)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Monday range backtest from cached Binance candles.")
    ap.add_argument("--strategy", choices=["current_monday_range", "sweep_retest"], default="current_monday_range")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--risk_fraction", type=float, default=0.01)
    ap.add_argument("--tp1_range_fraction", type=float, default=0.5)
    ap.add_argument("--tp2_range_fraction", type=float, default=1.0)
    ap.add_argument("--tp2_to_full", type=float, default=1.0, help="Legacy alias for TP2; prefer --tp2_range_fraction.")
    ap.add_argument("--friday_cutoff_hour_utc", type=int, default=23)
    ap.add_argument("--min_range_pct", type=float, default=None)
    ap.add_argument("--max_range_pct", type=float, default=None)
    ap.add_argument("--direction", choices=["both", "long_only", "short_only"], default="both")
    ap.add_argument("--max_entry_day_utc", type=int, default=None)
    ap.add_argument("--max_entry_hour_utc", type=int, default=None)
    ap.add_argument("--sma_period", type=int, default=None)
    ap.add_argument("--tp1_to_mid", type=float, default=1.0, help="Legacy alias for TP1; prefer --tp1_range_fraction.")
    ap.add_argument("--tp1_close_fraction", type=float, default=0.5)
    ap.add_argument("--stop_mode", choices=["opposite_boundary", "range_fraction", "fixed_pct"], default=None)
    ap.add_argument("--stop_range_fraction", type=float, default=None)
    ap.add_argument("--stop_pct", type=float, default=0.01)
    ap.add_argument("--move_stop_to_breakeven_after_tp1", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--breakeven_includes_fees", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exit_style", choices=["partial_tp2", "single_target"], default="partial_tp2")
    ap.add_argument("--single_target_mode", action=argparse.BooleanOptionalAction, default=None, help="Legacy alias; prefer --exit_style.")
    ap.add_argument("--single_target_level", choices=["tp1", "tp2"], default="tp2")
    ap.add_argument("--output_dir", default="data/backtests")
    args = ap.parse_args()

    result = run_backtest_cached(
        strategy=args.strategy,
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        risk_fraction=float(args.risk_fraction),
        tp1_range_fraction=args.tp1_range_fraction,
        tp2_range_fraction=args.tp2_range_fraction,
        tp2_to_full=float(args.tp2_to_full),
        friday_cutoff_hour_utc=int(args.friday_cutoff_hour_utc),
        min_range_pct=args.min_range_pct,
        max_range_pct=args.max_range_pct,
        direction=args.direction,
        max_entry_day_utc=args.max_entry_day_utc,
        max_entry_hour_utc=args.max_entry_hour_utc,
        sma_period=args.sma_period,
        tp1_to_mid=float(args.tp1_to_mid),
        tp1_close_fraction=float(args.tp1_close_fraction),
        stop_mode=args.stop_mode,
        stop_range_fraction=(None if args.stop_range_fraction is None else float(args.stop_range_fraction)),
        stop_pct=float(args.stop_pct),
        move_stop_to_breakeven_after_tp1=bool(args.move_stop_to_breakeven_after_tp1),
        breakeven_includes_fees=bool(args.breakeven_includes_fees),
        exit_style=args.exit_style,
        single_target_mode=(None if args.single_target_mode is None else bool(args.single_target_mode)),
        single_target_level=args.single_target_level,
        output_dir=args.output_dir,
    )

    print("=== Backtest run complete ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved outputs under: {result['out_dir']}")


if __name__ == "__main__":
    main()
