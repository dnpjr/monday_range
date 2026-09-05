from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json

import pandas as pd

from download_binance_data import run_cache_update
from src.binance_cache import load_cache_csv, detect_missing_candles
from run_backtest import run_backtest_cached
from analyze_backtest import run_backtest_diagnostics
from src.monday_range_research import (
    analyze_monday_range_research,
    summarize_monday_range_research,
    probability_tables,
    analyze_original_sweep_retest_research,
    summarize_original_sweep_retest_research,
)
from src.range_sweep_research import (
    SweepAnalysisConfig,
    analyze_range_sweeps,
    summarize_range_sweeps,
    range_sweep_tables,
)
from src.features import add_monday_range
from src.research import analyze_weekly_sweep_signals


RESEARCH_MODE_ONLY = True
STOP_MODE_LABEL_TO_VALUE = {
    "Opposite boundary": "opposite_boundary",
    "Range fraction": "range_fraction",
    "Fixed percent": "fixed_pct",
}
STOP_MODE_VALUE_TO_LABEL = {v: k for k, v in STOP_MODE_LABEL_TO_VALUE.items()}


def assert_research_mode_only() -> bool:
    if not RESEARCH_MODE_ONLY:
        raise RuntimeError("Research mode safety guard disabled.")
    return True


def format_probability_pct(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100.0:.{int(decimals)}f}%"


def format_hours(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{int(decimals)}f}h"


def format_float(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{int(decimals)}f}"


def stop_mode_label(mode_value: str) -> str:
    return STOP_MODE_VALUE_TO_LABEL.get(mode_value, mode_value)


def stop_mode_value(mode_label: str) -> str:
    return STOP_MODE_LABEL_TO_VALUE.get(mode_label, mode_label)


def stop_mode_visible_inputs(stop_mode: str) -> dict[str, bool]:
    mode = stop_mode_value(stop_mode)
    return {
        "show_stop_range_fraction": mode == "range_fraction",
        "show_stop_pct": mode == "fixed_pct",
    }


def default_stop_mode_for_context(context: str) -> str:
    c = str(context).strip().lower()
    if c == "research_lab":
        return "range_fraction"
    return "opposite_boundary"


def normalize_event_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    alias_map = {
        "strategy_entries": "strategy_signal",
        "strategy_signal": "strategy_signal",
        "sweep_events": "hypothetical_both_sides",
        "hypothetical_both_sides": "hypothetical_both_sides",
    }
    if m not in alias_map:
        raise ValueError(
            "Invalid event_mode. Use one of: strategy_entries, sweep_events."
        )
    return alias_map[m]


def normalize_research_lab_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    alias_map = {
        "original_sweep_retest": "original_sweep_retest",
        "original": "original_sweep_retest",
        "legacy": "original_sweep_retest",
        "new_path_analysis": "new_path_analysis",
        "path": "new_path_analysis",
        "path_analysis": "new_path_analysis",
        "new": "new_path_analysis",
    }
    if m not in alias_map:
        raise ValueError(
            "Invalid analysis_mode. Use one of: original_sweep_retest, new_path_analysis."
        )
    return alias_map[m]


def run_research_lab_analysis(
    *,
    ohlc: pd.DataFrame,
    analysis_mode: str = "new_path_analysis",
    tp1_range_fraction: float = 0.5,
    tp2_range_fraction: float = 1.0,
    stop_mode: str = "range_fraction",
    stop_range_fraction: float = 1.0,
    stop_pct: float = 0.01,
    friday_cutoff_hour_utc: int = 23,
    event_mode: str = "sweep_events",
) -> dict[str, Any]:
    assert_research_mode_only()
    normalized_analysis_mode = normalize_research_lab_mode(analysis_mode)

    if normalized_analysis_mode == "original_sweep_retest":
        results = analyze_original_sweep_retest_research(ohlc)
        return {
            "analysis_mode": normalized_analysis_mode,
            "results": results,
            "summary": summarize_original_sweep_retest_research(results),
            "tables": {},
            "event_mode_normalized": None,
        }

    normalized_event_mode = normalize_event_mode(event_mode)
    results = analyze_monday_range_research(
        ohlc,
        tp1_range_fraction=float(tp1_range_fraction),
        tp2_range_fraction=float(tp2_range_fraction),
        stop_mode=stop_mode_value(stop_mode),
        stop_range_fraction=float(stop_range_fraction),
        stop_pct=float(stop_pct),
        friday_cutoff_hour_utc=int(friday_cutoff_hour_utc),
        event_mode=normalized_event_mode,
    )
    return {
        "analysis_mode": normalized_analysis_mode,
        "results": results,
        "summary": summarize_monday_range_research(results),
        "tables": probability_tables(results),
        "event_mode_normalized": normalized_event_mode,
    }


def run_range_sweep_lab_analysis(
    *,
    ohlcv: pd.DataFrame,
    base_interval: str,
    reference_timeframe: str = "previous_day",
    execution_timeframe: str = "1h",
    confirmation_mode: str = "close_back_inside",
    event_mode: str = "first_sweep_per_reference_period",
    holding_window_mode: str = "until_end_of_reference_period",
    fixed_n_candles: int = 48,
    fixed_hours: int = 48,
) -> dict[str, Any]:
    assert_research_mode_only()
    cfg = SweepAnalysisConfig(
        reference_timeframe=str(reference_timeframe),
        execution_timeframe=str(execution_timeframe),
        confirmation_mode=str(confirmation_mode),
        event_mode=str(event_mode),
        holding_window_mode=str(holding_window_mode),
        fixed_n_candles=int(fixed_n_candles),
        fixed_hours=int(fixed_hours),
    )
    events = analyze_range_sweeps(
        ohlcv,
        base_interval=str(base_interval),
        config=cfg,
    )
    return {
        "events": events,
        "summary": summarize_range_sweeps(events),
        "tables": range_sweep_tables(events),
        "config": cfg.__dict__,
    }


def update_data_cache(
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    lookback_days: int | None = 1825,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    cache_dir: str = "data/binance",
) -> dict[str, Any]:
    assert_research_mode_only()
    return run_cache_update(
        symbol=symbol,
        interval=interval,
        lookback_days=lookback_days,
        start=start,
        end=end,
        force=force,
        cache_dir=cache_dir,
    )


def get_cache_status(
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    cache_dir: str = "data/binance",
) -> dict[str, Any]:
    cache_path = Path(cache_dir) / f"{symbol}_{interval}.csv"
    if not cache_path.exists():
        return {
            "cache_exists": False,
            "cache_file": str(cache_path),
            "rows": 0,
            "first_candle": None,
            "last_candle": None,
            "duplicate_count": 0,
            "missing_candle_count": None,
            "latest_candle_age_seconds": None,
        }

    raw = load_cache_csv(cache_path)
    rows = int(len(raw))
    duplicate_count = int(raw["open_time"].duplicated().sum()) if "open_time" in raw.columns else 0
    first_candle = raw["open_time"].min().isoformat() if rows and "open_time" in raw.columns else None
    last_candle = raw["open_time"].max().isoformat() if rows and "open_time" in raw.columns else None
    missing = detect_missing_candles(raw, interval)

    latest_age = None
    if rows:
        latest_ts = raw["close_time"].max() if "close_time" in raw.columns else raw["open_time"].max()
        if latest_ts is not None and pd.notna(latest_ts):
            now_utc = pd.Timestamp.now(tz="UTC")
            latest_age = max(0.0, float((now_utc - latest_ts).total_seconds()))

    return {
        "cache_exists": True,
        "cache_file": str(cache_path),
        "rows": rows,
        "first_candle": first_candle,
        "last_candle": last_candle,
        "duplicate_count": duplicate_count,
        "missing_candle_count": missing,
        "latest_candle_age_seconds": latest_age,
    }


def run_research_backtest(
    *,
    strategy: str = "current_monday_range",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str | None = None,
    end: str | None = None,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    risk_fraction: float = 0.02,
    tp1_range_fraction: float | None = 0.5,
    tp2_range_fraction: float | None = 1.0,
    tp2_to_full: float = 1.0,
    friday_cutoff_hour_utc: int = 20,
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
    single_target_mode: bool | None = None,
    single_target_level: str = "tp2",
    output_dir: str = "data/backtests",
) -> dict[str, Any]:
    assert_research_mode_only()
    return run_backtest_cached(
        strategy=strategy,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        risk_fraction=risk_fraction,
        tp1_range_fraction=tp1_range_fraction,
        tp2_range_fraction=tp2_range_fraction,
        tp2_to_full=tp2_to_full,
        friday_cutoff_hour_utc=friday_cutoff_hour_utc,
        min_range_pct=min_range_pct,
        max_range_pct=max_range_pct,
        direction=direction,
        max_entry_day_utc=max_entry_day_utc,
        max_entry_hour_utc=max_entry_hour_utc,
        sma_period=sma_period,
        tp1_to_mid=tp1_to_mid,
        tp1_close_fraction=tp1_close_fraction,
        stop_mode=stop_mode,
        stop_range_fraction=stop_range_fraction,
        stop_pct=stop_pct,
        move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
        breakeven_includes_fees=breakeven_includes_fees,
        exit_style=exit_style,
        single_target_mode=single_target_mode,
        single_target_level=single_target_level,
        output_dir=output_dir,
    )


def estimate_sweep_retest_signals(ohlc: pd.DataFrame) -> int:
    if ohlc is None or ohlc.empty:
        return 0
    df = add_monday_range(ohlc)
    return int(len(analyze_weekly_sweep_signals(df)))


def _to_iso_date(value: str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts_utc.date().isoformat()


def _extract_metrics_row(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "period": label,
        "total_return_pct": summary.get("total_return_pct"),
        "max_drawdown": summary.get("max_drawdown"),
        "total_fees_paid": summary.get("total_fees_paid"),
        "num_trades": summary.get("num_trades"),
        "win_rate": summary.get("win_rate"),
    }


def build_train_validation_table(train_summary: dict[str, Any], validation_summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _extract_metrics_row("train", train_summary),
            _extract_metrics_row("validation", validation_summary),
        ]
    )


def generate_cost_sensitivity_grid(
    fee_bps_values: list[float] | None = None,
    slippage_bps_values: list[float] | None = None,
) -> list[dict[str, float]]:
    fees = [0.0, 7.5, 10.0] if fee_bps_values is None else [float(v) for v in fee_bps_values]
    slips = [0.0, 1.0, 2.5, 5.0] if slippage_bps_values is None else [float(v) for v in slippage_bps_values]
    out: list[dict[str, float]] = []
    for f in fees:
        for s in slips:
            out.append({"fee_bps": float(f), "slippage_bps": float(s)})
    return out


def generate_neighbourhood_grid(
    *,
    tp1_range_fraction: float,
    stop_range_fraction: float,
    tp1_steps: list[float] | None = None,
    stop_steps: list[float] | None = None,
) -> list[dict[str, float]]:
    tp1_step_vals = [-0.1, 0.0, 0.1] if tp1_steps is None else [float(v) for v in tp1_steps]
    stop_step_vals = [-0.1, 0.0, 0.1] if stop_steps is None else [float(v) for v in stop_steps]
    out: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()
    for dt in tp1_step_vals:
        for ds in stop_step_vals:
            tp1_val = max(0.05, float(tp1_range_fraction) + dt)
            stop_val = max(0.05, float(stop_range_fraction) + ds)
            key = (round(tp1_val, 6), round(stop_val, 6))
            if key in seen:
                continue
            seen.add(key)
            out.append({"tp1_range_fraction": tp1_val, "stop_range_fraction": stop_val})
    return out


def _base_backtest_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "strategy",
        "symbol",
        "interval",
        "start",
        "end",
        "fee_bps",
        "slippage_bps",
        "risk_fraction",
        "tp1_range_fraction",
        "tp2_range_fraction",
        "tp2_to_full",
        "friday_cutoff_hour_utc",
        "min_range_pct",
        "max_range_pct",
        "direction",
        "max_entry_day_utc",
        "max_entry_hour_utc",
        "sma_period",
        "tp1_to_mid",
        "tp1_close_fraction",
        "stop_mode",
        "stop_range_fraction",
        "stop_pct",
        "move_stop_to_breakeven_after_tp1",
        "breakeven_includes_fees",
        "exit_style",
        "single_target_mode",
        "single_target_level",
    ]
    out = {k: config.get(k) for k in keys}
    out["output_dir"] = config.get("output_dir", "data/backtests")
    return out


def run_train_validation_split(
    *,
    config: dict[str, Any],
    latest_cached_date: str,
    run_fn: Callable[..., dict[str, Any]] = run_research_backtest,
) -> dict[str, Any]:
    assert_research_mode_only()
    base = _base_backtest_kwargs(config)
    train_cfg = dict(base)
    train_cfg["start"] = "2021-05-21"
    train_cfg["end"] = "2024-12-31"
    val_cfg = dict(base)
    val_cfg["start"] = "2025-01-01"
    val_cfg["end"] = _to_iso_date(latest_cached_date)

    train_res = run_fn(**train_cfg)
    val_res = run_fn(**val_cfg)
    train_summary = train_res.get("summary", {}) if isinstance(train_res, dict) else {}
    val_summary = val_res.get("summary", {}) if isinstance(val_res, dict) else {}
    table = build_train_validation_table(train_summary, val_summary)
    return {
        "train": train_res,
        "validation": val_res,
        "table": table,
    }


def run_cost_sensitivity(
    *,
    config: dict[str, Any],
    fee_bps_values: list[float] | None = None,
    slippage_bps_values: list[float] | None = None,
    run_fn: Callable[..., dict[str, Any]] = run_research_backtest,
) -> pd.DataFrame:
    assert_research_mode_only()
    base = _base_backtest_kwargs(config)
    rows: list[dict[str, Any]] = []
    for combo in generate_cost_sensitivity_grid(fee_bps_values=fee_bps_values, slippage_bps_values=slippage_bps_values):
        cfg = dict(base)
        cfg["fee_bps"] = float(combo["fee_bps"])
        cfg["slippage_bps"] = float(combo["slippage_bps"])
        res = run_fn(**cfg)
        summary = res.get("summary", {}) if isinstance(res, dict) else {}
        rows.append(
            {
                "fee_bps": cfg["fee_bps"],
                "slippage_bps": cfg["slippage_bps"],
                "total_return_pct": summary.get("total_return_pct"),
                "max_drawdown": summary.get("max_drawdown"),
                "total_fees_paid": summary.get("total_fees_paid"),
                "num_trades": summary.get("num_trades"),
                "win_rate": summary.get("win_rate"),
            }
        )
    return pd.DataFrame(rows).sort_values(["fee_bps", "slippage_bps"]).reset_index(drop=True)


def run_neighbourhood_test(
    *,
    config: dict[str, Any],
    tp1_steps: list[float] | None = None,
    stop_steps: list[float] | None = None,
    run_fn: Callable[..., dict[str, Any]] = run_research_backtest,
) -> pd.DataFrame:
    assert_research_mode_only()
    base = _base_backtest_kwargs(config)
    grid = generate_neighbourhood_grid(
        tp1_range_fraction=float(base.get("tp1_range_fraction", 0.5)),
        stop_range_fraction=float(base.get("stop_range_fraction", 0.5 if base.get("strategy") == "sweep_retest" else 1.0)),
        tp1_steps=tp1_steps,
        stop_steps=stop_steps,
    )
    rows: list[dict[str, Any]] = []
    for combo in grid:
        cfg = dict(base)
        cfg["tp1_range_fraction"] = float(combo["tp1_range_fraction"])
        cfg["stop_range_fraction"] = float(combo["stop_range_fraction"])
        res = run_fn(**cfg)
        summary = res.get("summary", {}) if isinstance(res, dict) else {}
        rows.append(
            {
                "tp1_range_fraction": cfg["tp1_range_fraction"],
                "stop_range_fraction": cfg["stop_range_fraction"],
                "total_return_pct": summary.get("total_return_pct"),
                "max_drawdown": summary.get("max_drawdown"),
                "total_fees_paid": summary.get("total_fees_paid"),
                "num_trades": summary.get("num_trades"),
                "win_rate": summary.get("win_rate"),
            }
        )
    return pd.DataFrame(rows).sort_values(["tp1_range_fraction", "stop_range_fraction"]).reset_index(drop=True)


def _parse_cache_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def build_dataset_label(symbol: str, interval: str, cache_status: dict[str, Any]) -> str:
    if not cache_status.get("cache_exists"):
        return f"{symbol} {interval} | cache not found"
    first_ts = _parse_cache_date(cache_status.get("first_candle"))
    last_ts = _parse_cache_date(cache_status.get("last_candle"))
    rows = int(cache_status.get("rows", 0))
    first_txt = first_ts.date().isoformat() if first_ts is not None else "N/A"
    last_txt = last_ts.date().isoformat() if last_ts is not None else "N/A"
    return f"{symbol} {interval} | {first_txt} -> {last_txt} | {rows:,} candles"


def get_date_preset_range(preset_name: str, cache_status: dict[str, Any]) -> dict[str, Any]:
    preset = str(preset_name).strip().lower()
    if preset == "custom":
        return {"start": None, "end": None, "clipped": False, "warning": None}

    first_ts = _parse_cache_date(cache_status.get("first_candle"))
    last_ts = _parse_cache_date(cache_status.get("last_candle"))
    if first_ts is None or last_ts is None:
        return {"start": None, "end": None, "clipped": False, "warning": "No cache range available."}

    if preset == "full cached range":
        start_ts, end_ts = first_ts, last_ts
    elif preset == "last 12 months":
        end_ts = last_ts
        start_ts = last_ts - pd.Timedelta(days=365)
    elif preset == "2021":
        start_ts, end_ts = pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-12-31", tz="UTC")
    elif preset == "2022":
        start_ts, end_ts = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31", tz="UTC")
    elif preset == "2023":
        start_ts, end_ts = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
    elif preset == "2024":
        start_ts, end_ts = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")
    elif preset == "2025":
        start_ts, end_ts = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
    elif preset == "2022-2023":
        start_ts, end_ts = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
    elif preset == "2021-2024 training":
        start_ts, end_ts = pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")
    elif preset == "2025-2026 validation":
        start_ts, end_ts = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-12-31", tz="UTC")
    else:
        return {"start": None, "end": None, "clipped": False, "warning": f"Unknown preset: {preset_name}"}

    clipped = False
    if start_ts < first_ts:
        start_ts = first_ts
        clipped = True
    if end_ts > last_ts:
        end_ts = last_ts
        clipped = True

    warning = None
    if end_ts < start_ts:
        warning = "Preset range is outside cached range."
    elif clipped:
        warning = "Preset range was clipped to available cache range."

    return {
        "start": start_ts.date().isoformat(),
        "end": end_ts.date().isoformat(),
        "clipped": clipped,
        "warning": warning,
    }


def validate_backtest_date_range(
    start: str | None,
    end: str | None,
    cache_status: dict[str, Any],
) -> dict[str, Any]:
    if start is None or end is None:
        return {"valid": False, "error": "Start and end are required.", "start": start, "end": end, "warning": None}
    start_ts = _parse_cache_date(start)
    end_ts = _parse_cache_date(end)
    if start_ts is None or end_ts is None:
        return {"valid": False, "error": "Invalid start/end date.", "start": start, "end": end, "warning": None}
    if end_ts < start_ts:
        return {"valid": False, "error": "End date must be on or after start date.", "start": start, "end": end, "warning": None}

    warning = None
    first_ts = _parse_cache_date(cache_status.get("first_candle"))
    last_ts = _parse_cache_date(cache_status.get("last_candle"))
    if first_ts is not None and last_ts is not None:
        if start_ts < first_ts or end_ts > last_ts:
            warning = "Selected date range extends outside cached range; results will be truncated."
    return {"valid": True, "error": None, "start": start_ts.date().isoformat(), "end": end_ts.date().isoformat(), "warning": warning}


def build_backtest_run_label(run_dir: str | Path) -> str:
    loaded = load_backtest_outputs(run_dir)
    summary = loaded.get("summary")
    config = loaded.get("config")
    if not isinstance(summary, dict):
        return Path(run_dir).name
    symbol = summary.get("symbol") or (config.get("symbol") if isinstance(config, dict) else None) or "N/A"
    interval = summary.get("interval") or (config.get("interval") if isinstance(config, dict) else None) or "N/A"
    start_used = str(summary.get("start_used", "N/A"))[:10]
    end_used = str(summary.get("end_used", "N/A"))[:10]
    ret = summary.get("total_return_pct")
    trades = summary.get("num_trades")
    ret_txt = f"{float(ret):.2f}%" if ret is not None else "N/A"
    trades_txt = str(int(trades)) if trades is not None else "N/A"
    return f"{symbol} {interval} | {start_used} -> {end_used} | return {ret_txt} | trades {trades_txt}"


def load_backtest_outputs(backtest_dir: str | Path) -> dict[str, Any]:
    root = Path(backtest_dir)
    out: dict[str, Any] = {
        "backtest_dir": str(root),
        "summary": None,
        "config": None,
        "trades": None,
        "trades_count": 0,
        "summary_exists": False,
        "config_exists": False,
        "trades_exists": False,
    }

    summary_path = root / "summary.json"
    config_path = root / "config.json"
    trades_path = root / "trades.csv"

    if summary_path.exists():
        out["summary_exists"] = True
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                out["summary"] = json.load(f)
        except Exception:
            out["summary"] = None

    if config_path.exists():
        out["config_exists"] = True
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                out["config"] = json.load(f)
        except Exception:
            out["config"] = None

    if trades_path.exists():
        out["trades_exists"] = True
        try:
            trades = pd.read_csv(trades_path)
            out["trades"] = trades
            out["trades_count"] = int(len(trades))
        except Exception:
            out["trades"] = None
            out["trades_count"] = 0

    return out


def discover_backtest_runs(backtests_dir: str | Path = "data/backtests") -> list[str]:
    root = Path(backtests_dir)
    if not root.exists():
        return []

    runs: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "trades.csv").exists() and (p / "summary.json").exists():
            runs.append(p)

    runs = sorted(runs, key=lambda x: (x / "summary.json").stat().st_mtime, reverse=True)
    return [str(p) for p in runs]


def run_research_diagnostics(backtest_dir: str, cache_dir: str = "data/binance") -> dict[str, Any]:
    assert_research_mode_only()
    return run_backtest_diagnostics(
        backtest_dir=backtest_dir,
        cache_dir=cache_dir,
    )


def load_diagnostics_outputs(backtest_dir: str | Path) -> dict[str, Any]:
    root = Path(backtest_dir)
    out: dict[str, Any] = {
        "backtest_dir": str(root),
        "diagnostics": None,
        "by_year": None,
        "by_direction": None,
        "by_exit_reason": None,
        "by_range_bucket": None,
        "diagnostics_exists": False,
        "by_year_exists": False,
        "by_direction_exists": False,
        "by_exit_reason_exists": False,
        "by_range_bucket_exists": False,
    }

    diagnostics_path = root / "diagnostics.json"
    by_year_path = root / "diagnostics_by_year.csv"
    by_direction_path = root / "diagnostics_by_direction.csv"
    by_exit_reason_path = root / "diagnostics_by_exit_reason.csv"
    by_range_bucket_path = root / "diagnostics_by_range_bucket.csv"

    if diagnostics_path.exists():
        out["diagnostics_exists"] = True
        try:
            with open(diagnostics_path, "r", encoding="utf-8") as f:
                out["diagnostics"] = json.load(f)
        except Exception:
            out["diagnostics"] = None

    if by_year_path.exists():
        out["by_year_exists"] = True
        try:
            out["by_year"] = pd.read_csv(by_year_path)
        except Exception:
            out["by_year"] = None

    if by_direction_path.exists():
        out["by_direction_exists"] = True
        try:
            out["by_direction"] = pd.read_csv(by_direction_path)
        except Exception:
            out["by_direction"] = None

    if by_exit_reason_path.exists():
        out["by_exit_reason_exists"] = True
        try:
            out["by_exit_reason"] = pd.read_csv(by_exit_reason_path)
        except Exception:
            out["by_exit_reason"] = None

    if by_range_bucket_path.exists():
        out["by_range_bucket_exists"] = True
        try:
            out["by_range_bucket"] = pd.read_csv(by_range_bucket_path)
        except Exception:
            out["by_range_bucket"] = None

    return out
