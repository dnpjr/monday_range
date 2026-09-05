from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from run_backtest import build_summary, filter_ohlc_window, load_cached_ohlc
from src.canonical_data import DATASET_VERSION as DEFAULT_DATASET_VERSION, experiment_metadata
from run_parameter_sweep import (
    build_sweep_row,
    parse_float_grid,
    parse_int_grid,
    sweep_score,
)
from src.backtest import backtest_sweep_fade
from src.features import add_monday_range


DEFAULT_SWEEP_RETEST_FEE_BPS = 10.0
DEFAULT_SWEEP_RETEST_SLIPPAGE_BPS = 5.0


def _parse_token_grid(raw: str | list[str], *, name: str) -> list[str]:
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


def _period_bounds(start_year: int, end_year: int) -> tuple[str, str]:
    return f"{start_year:04d}-01-01", f"{end_year:04d}-12-31"


def _to_iso_date(value: str | pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts_utc.date().isoformat()


def build_yearly_mode_folds(latest_cached_date: str | pd.Timestamp | None = None) -> list[dict[str, Any]]:
    latest = _to_iso_date(latest_cached_date or "2026-12-31")
    folds: list[dict[str, Any]] = [
        {
            "fold_id": 1,
            "train_start": "2021-05-21",
            "train_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2023-12-31",
            "train_years": [2021, 2022],
            "test_years": [2023, 2023],
        },
        {
            "fold_id": 2,
            "train_start": "2022-01-01",
            "train_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2024-12-31",
            "train_years": [2022, 2023],
            "test_years": [2024, 2024],
        },
        {
            "fold_id": 3,
            "train_start": "2023-01-01",
            "train_end": "2024-12-31",
            "test_start": "2025-01-01",
            "test_end": "2025-12-31",
            "train_years": [2023, 2024],
            "test_years": [2025, 2025],
        },
    ]
    if pd.Timestamp(latest, tz="UTC") >= pd.Timestamp("2026-01-01", tz="UTC"):
        folds.append(
            {
                "fold_id": 4,
                "train_start": "2024-01-01",
                "train_end": "2025-12-31",
                "test_start": "2026-01-01",
                "test_end": latest,
                "train_years": [2024, 2025],
                "test_years": [2026, pd.Timestamp(latest).year],
            }
        )
    return folds


def generate_sweep_retest_grid(
    *,
    risk_fraction: float = 0.02,
    tp1_range_fractions: list[float] | None = None,
    stop_range_fractions: list[float] | None = None,
    tp1_close_fractions: list[float] | None = None,
    tp2_range_fractions: list[float] | None = None,
    friday_cutoff_hours_utc: list[int] | None = None,
    exit_styles: list[str] | None = None,
    stop_mode: str = "swept_boundary_offset",
    direction: str = "both",
) -> list[dict[str, Any]]:
    tp1_range_fractions = [0.35, 0.40, 0.45, 0.50, 0.55] if tp1_range_fractions is None else [float(v) for v in tp1_range_fractions]
    stop_range_fractions = [0.75, 1.0, 1.1, 1.25] if stop_range_fractions is None else [float(v) for v in stop_range_fractions]
    tp1_close_fractions = [0.75, 0.90, 0.99, 1.0] if tp1_close_fractions is None else [float(v) for v in tp1_close_fractions]
    tp2_range_fractions = [1.0] if tp2_range_fractions is None else [float(v) for v in tp2_range_fractions]
    friday_cutoff_hours_utc = [20, 23] if friday_cutoff_hours_utc is None else [int(v) for v in friday_cutoff_hours_utc]
    exit_styles = ["partial_tp2", "single_target"] if exit_styles is None else [str(v) for v in exit_styles]
    valid_exit_styles = {"partial_tp2", "single_target"}
    bad_exit = [x for x in exit_styles if x not in valid_exit_styles]
    if bad_exit:
        raise ValueError(f"Invalid exit styles: {bad_exit}")

    rows: list[dict[str, Any]] = []
    for tp1_rf in tp1_range_fractions:
        for stop_rf in stop_range_fractions:
            for tp1_cf in tp1_close_fractions:
                for tp2_rf in tp2_range_fractions:
                    for fri_hour in friday_cutoff_hours_utc:
                        for exit_style in exit_styles:
                            rows.append(
                                {
                                    "risk_fraction": float(risk_fraction),
                                    "tp1_range_fraction": float(tp1_rf),
                                    "tp2_range_fraction": float(tp2_rf),
                                    "tp2_to_full": (1.0 if float(tp2_rf) >= 1.0 else (2.0 * float(tp2_rf) - 1.0)),
                                    "tp1_close_fraction": float(tp1_cf),
                                    "stop_mode": str(stop_mode),
                                    "stop_range_fraction": float(stop_rf),
                                    "friday_cutoff_hour_utc": int(fri_hour),
                                    "direction": str(direction),
                                    "exit_style": str(exit_style),
                                    "single_target_level": ("tp1" if exit_style == "single_target" else "tp2"),
                                }
                            )
    return rows


def build_rolling_year_folds(
    *,
    available_years: list[int],
    train_window: int,
    test_window: int,
) -> list[dict[str, Any]]:
    if train_window < 1 or test_window < 1:
        raise ValueError("train_window and test_window must be >= 1.")
    years = sorted(set(int(y) for y in available_years))
    folds: list[dict[str, Any]] = []
    fold_id = 1
    i = 0
    while True:
        train_start_idx = i
        train_end_idx = i + train_window - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_window - 1
        if test_end_idx >= len(years):
            break
        train_start_year = years[train_start_idx]
        train_end_year = years[train_end_idx]
        test_start_year = years[test_start_idx]
        test_end_year = years[test_end_idx]
        train_start, train_end = _period_bounds(train_start_year, train_end_year)
        test_start, test_end = _period_bounds(test_start_year, test_end_year)
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_years": [train_start_year, train_end_year],
                "test_years": [test_start_year, test_end_year],
            }
        )
        fold_id += 1
        i += test_window
    return folds


def _evaluate_grid_on_features(
    *,
    df_features: pd.DataFrame,
    combos: list[dict[str, Any]],
    strategy: str,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    fee_bps: float,
    slippage_bps: float,
    initial_cash: float = 10_000.0,
    risk_base: str = "current_equity",
    max_leverage: float = 1.0,
    intrabar_policy: str = "conservative_stop_first",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for combo in combos:
        risk_fraction = float(combo["risk_fraction"])
        tp1_range_fraction = float(combo.get("tp1_range_fraction", 0.5))
        tp2_range_fraction = float(combo.get("tp2_range_fraction", 1.0))
        friday_cutoff_hour_utc = int(combo["friday_cutoff_hour_utc"])
        min_range_pct = combo.get("min_range_pct")
        max_range_pct = combo.get("max_range_pct")
        direction = str(combo.get("direction", "both"))
        max_entry_day_utc = combo.get("max_entry_day_utc")
        max_entry_hour_utc = combo.get("max_entry_hour_utc")
        tp1_close_fraction = float(combo["tp1_close_fraction"])
        move_stop_to_breakeven_after_tp1 = bool(combo.get("move_stop_to_breakeven_after_tp1", True))
        stop_mode = str(combo.get("stop_mode", "swept_boundary_offset"))
        stop_range_fraction = float(combo["stop_range_fraction"])
        tp2_to_full = float(combo.get("tp2_to_full", 1.0))
        exit_style = str(combo.get("exit_style", "partial_tp2"))
        single_target_mode = bool(exit_style == "single_target")
        single_target_level = str(combo.get("single_target_level", "tp1"))

        if min_range_pct is not None and max_range_pct is not None and float(min_range_pct) > float(max_range_pct):
            continue

        risk_per_trade = initial_cash * risk_fraction
        df_out, trades = backtest_sweep_fade(
            df_features,
            strategy=strategy,
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
            single_target_mode=single_target_mode,
            single_target_level=single_target_level,
        )
        summary = build_summary(
            strategy=strategy,
            signal_count=int(df_out.attrs.get("signal_count", 0)),
            exit_style=exit_style,
            single_target_level=(single_target_level if single_target_mode else None),
            symbol=symbol,
            interval=interval,
            initial_cash=initial_cash,
            df_out=df_out,
            trades=trades,
            start_used=start,
            end_used=end,
        )
        row = build_sweep_row(
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
        row["exit_style"] = exit_style
        row["single_target_level"] = single_target_level
        row["strategy"] = strategy
        row["signal_count"] = summary.get("signal_count")
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            by=["score", "return_after_costs_pct", "cost_drag_pct", "num_trades"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)
        out["train_rank"] = range(1, len(out) + 1)
    return out


def _window_features(ohlc: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    period = filter_ohlc_window(ohlc, start, end)
    if period.empty:
        return period
    return add_monday_range(period)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def run_walk_forward(
    *,
    strategy: str = "sweep_retest",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    yearly_mode: bool = True,
    train_window: int = 1,
    test_window: int = 1,
    fee_bps: float = DEFAULT_SWEEP_RETEST_FEE_BPS,
    slippage_bps: float = DEFAULT_SWEEP_RETEST_SLIPPAGE_BPS,
    top_n: int = 3,
    output_dir: str | Path = "data/walk_forward",
    risk_fraction: float = 0.02,
    tp1_range_fractions: list[float] | None = None,
    stop_range_fractions: list[float] | None = None,
    tp1_close_fractions: list[float] | None = None,
    tp2_range_fractions: list[float] | None = None,
    friday_cutoff_hours_utc: list[int] | None = None,
    exit_styles: list[str] | None = None,
    risk_base: str = "current_equity",
    max_leverage: float = 1.0,
    intrabar_policy: str = "conservative_stop_first",
    dataset_version: str | None = DEFAULT_DATASET_VERSION,
    # Legacy compatibility args (optional)
    risk_fractions: list[float] | None = None,
    tp2_to_full_values: list[float] | None = None,
    min_range_pcts: list[float | None] | None = None,
    max_range_pcts: list[float | None] | None = None,
    directions: list[str] | None = None,
    max_entry_days_utc: list[int | None] | None = None,
    max_entry_hours_utc: list[int | None] | None = None,
    move_stop_to_breakeven_after_tp1_values: list[bool] | None = None,
    stop_modes: list[str] | None = None,
) -> dict[str, Any]:
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")
    if strategy != "sweep_retest":
        raise ValueError("run_walk_forward currently supports strategy='sweep_retest' only.")

    if risk_fractions:
        risk_fraction = float(risk_fractions[0])
    tp1_range_fractions = [0.35, 0.40, 0.45, 0.50, 0.55] if tp1_range_fractions is None else tp1_range_fractions
    stop_range_fractions = [0.75, 1.0, 1.1, 1.25] if stop_range_fractions is None else stop_range_fractions
    tp1_close_fractions = [0.75, 0.90, 0.99, 1.0] if tp1_close_fractions is None else tp1_close_fractions
    tp2_range_fractions = [1.0] if tp2_range_fractions is None else tp2_range_fractions
    if tp2_to_full_values is not None and tp2_range_fractions == [1.0]:
        tp2_range_fractions = [1.0 if float(v) >= 1.0 else (0.5 + 0.5 * float(v)) for v in tp2_to_full_values]
    friday_cutoff_hours_utc = [20, 23] if friday_cutoff_hours_utc is None else friday_cutoff_hours_utc
    exit_styles = ["partial_tp2", "single_target"] if exit_styles is None else exit_styles
    if stop_modes:
        if "range_fraction" in stop_modes or "swept_boundary_offset" in stop_modes:
            pass
        else:
            # Sweep/retest walk-forward is defined with range-fraction stops.
            stop_range_fractions = [float(stop_range_fractions[0])] if stop_range_fractions else [0.75]

    ohlc = load_cached_ohlc(symbol, interval, dataset_version=dataset_version)
    if ohlc.empty:
        raise ValueError("Evaluation OHLC is empty.")

    latest_cached_date = _to_iso_date(ohlc.index.max())
    if yearly_mode:
        folds = build_yearly_mode_folds(latest_cached_date=latest_cached_date)
    else:
        years = sorted(set(int(y) for y in ohlc.index.year))
        folds = build_rolling_year_folds(
            available_years=years,
            train_window=int(train_window),
            test_window=int(test_window),
        )
    if not folds:
        raise ValueError("No valid folds were generated for walk-forward.")

    combos = generate_sweep_retest_grid(
        risk_fraction=float(risk_fraction),
        tp1_range_fractions=tp1_range_fractions,
        stop_range_fractions=stop_range_fractions,
        tp1_close_fractions=tp1_close_fractions,
        tp2_range_fractions=tp2_range_fractions,
        friday_cutoff_hours_utc=friday_cutoff_hours_utc,
        exit_styles=exit_styles,
        stop_mode="swept_boundary_offset",
        direction="both",
    )

    train_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []
    fold_summary_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_start = str(fold["train_start"])
        train_end = str(fold["train_end"])
        test_start = str(fold["test_start"])
        test_end = str(fold["test_end"])

        train_features = _window_features(ohlc, train_start, train_end)
        test_features = _window_features(ohlc, test_start, test_end)
        if train_features.empty or test_features.empty:
            fold_summary_rows.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "status": "skipped_empty_window",
                }
            )
            continue

        train_df = _evaluate_grid_on_features(
            df_features=train_features,
            combos=combos,
            strategy=strategy,
            symbol=symbol,
            interval=interval,
            start=train_start,
            end=train_end,
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            risk_base=risk_base,
            max_leverage=max_leverage,
            intrabar_policy=intrabar_policy,
        )
        if train_df.empty:
            fold_summary_rows.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "status": "skipped_no_train_results",
                }
            )
            continue

        train_df = train_df.copy()
        train_df.insert(0, "fold_id", fold_id)
        train_df["train_start"] = train_start
        train_df["train_end"] = train_end
        train_df["test_start"] = test_start
        train_df["test_end"] = test_end
        train_rows.append(train_df)

        selected = train_df.sort_values("train_rank", ascending=True).head(int(top_n)).copy()

        selected_test_rows: list[dict[str, Any]] = []
        for _, sel in selected.iterrows():
            combo_like = {
                "risk_fraction": float(sel["risk_fraction"]),
                "tp2_to_full": float(sel["tp2_to_full"]),
                "tp1_range_fraction": float(sel.get("tp1_range_fraction", 0.5)),
                "tp2_range_fraction": float(sel.get("tp2_range_fraction", 1.0)),
                "friday_cutoff_hour_utc": int(sel["friday_cutoff_hour_utc"]),
                "min_range_pct": (None if ("min_range_pct" not in sel or pd.isna(sel["min_range_pct"])) else float(sel["min_range_pct"])),
                "max_range_pct": (None if ("max_range_pct" not in sel or pd.isna(sel["max_range_pct"])) else float(sel["max_range_pct"])),
                "direction": str(sel.get("direction", "both")),
                "max_entry_day_utc": (None if ("max_entry_day_utc" not in sel or pd.isna(sel["max_entry_day_utc"])) else int(sel["max_entry_day_utc"])),
                "max_entry_hour_utc": (None if ("max_entry_hour_utc" not in sel or pd.isna(sel["max_entry_hour_utc"])) else int(sel["max_entry_hour_utc"])),
                "tp1_close_fraction": float(sel["tp1_close_fraction"]),
                "move_stop_to_breakeven_after_tp1": bool(sel.get("move_stop_to_breakeven_after_tp1", True)),
                "stop_mode": str(sel.get("stop_mode", "swept_boundary_offset")),
                "stop_range_fraction": float(sel["stop_range_fraction"]),
                "exit_style": str(sel.get("exit_style", "partial_tp2")),
                "single_target_level": str(sel.get("single_target_level", "tp1")),
            }
            test_eval_df = _evaluate_grid_on_features(
                df_features=test_features,
                combos=[combo_like],
                strategy=strategy,
                symbol=symbol,
                interval=interval,
                start=test_start,
                end=test_end,
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                risk_base=risk_base,
                max_leverage=max_leverage,
                intrabar_policy=intrabar_policy,
            )
            if test_eval_df.empty:
                continue
            test_row = test_eval_df.iloc[0].to_dict()
            train_score = float(sel["score"])
            train_return = float(sel["total_return_pct"])
            train_drawdown = sel.get("max_drawdown")
            train_num_trades = int(sel.get("num_trades", 0))
            test_row.update(
                {
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_rank": int(sel["train_rank"]),
                    "is_primary": bool(int(sel["train_rank"]) == 1),
                    "train_total_return_pct": train_return,
                    "train_max_drawdown": train_drawdown,
                    "train_num_trades": train_num_trades,
                    "robustness_score": train_score,
                    "test_total_return_pct": float(test_row["total_return_pct"]),
                    "test_max_drawdown": test_row.get("max_drawdown"),
                    "test_num_trades": int(test_row.get("num_trades", 0)),
                    "test_score": sweep_score(
                        total_return_pct=float(test_row["total_return_pct"]),
                        max_drawdown=test_row.get("max_drawdown"),
                        num_trades=int(test_row.get("num_trades", 0)),
                        cost_drag_pct=float(test_row.get("cost_drag_pct", 0.0)),
                    ),
                }
            )
            selected_test_rows.append(test_row)

        if selected_test_rows:
            fold_test_df = pd.DataFrame(selected_test_rows)
            test_rows.append(fold_test_df)
            primary = fold_test_df[fold_test_df["is_primary"] == True]
            if primary.empty:
                primary = fold_test_df.head(1)
            p = primary.iloc[0]
            fold_summary_rows.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "status": "ok",
                    "primary_train_rank": int(p["train_rank"]),
                    "train_return_pct": float(p["train_total_return_pct"]),
                    "test_return_pct": float(p["test_total_return_pct"]),
                    "train_drawdown": p.get("train_max_drawdown"),
                    "test_drawdown": p.get("test_max_drawdown"),
                    "train_num_trades": int(p.get("train_num_trades", 0)),
                    "test_num_trades": int(p.get("test_num_trades", 0)),
                    "robustness_score": float(p.get("robustness_score", 0.0)),
                    "test_positive": bool(float(p["test_total_return_pct"]) > 0.0),
                    "tp1_range_fraction": p.get("tp1_range_fraction"),
                    "stop_range_fraction": p.get("stop_range_fraction"),
                    "tp1_close_fraction": p.get("tp1_close_fraction"),
                    "friday_cutoff_hour_utc": p.get("friday_cutoff_hour_utc"),
                    "exit_style": p.get("exit_style"),
                }
            )
        else:
            fold_summary_rows.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "status": "skipped_no_test_results",
                }
            )

    fold_train_results = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    fold_test_results = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    fold_summary = pd.DataFrame(fold_summary_rows)

    ok_folds = fold_summary[fold_summary["status"] == "ok"] if not fold_summary.empty else pd.DataFrame()
    pct_positive = (
        float(ok_folds["test_positive"].mean() * 100.0)
        if not ok_folds.empty and "test_positive" in ok_folds.columns
        else 0.0
    )
    avg_oos_return = (
        float(ok_folds["test_return_pct"].mean())
        if not ok_folds.empty and "test_return_pct" in ok_folds.columns
        else 0.0
    )
    worst_oos_return = (
        float(ok_folds["test_return_pct"].min())
        if not ok_folds.empty and "test_return_pct" in ok_folds.columns
        else 0.0
    )
    worst_oos_dd = (
        float(ok_folds["test_drawdown"].min())
        if not ok_folds.empty and "test_drawdown" in ok_folds.columns
        else None
    )
    stability: dict[str, float] = {}
    mode_params: dict[str, Any] = {}
    for col in ["tp1_range_fraction", "stop_range_fraction", "tp1_close_fraction", "friday_cutoff_hour_utc", "exit_style"]:
        if not ok_folds.empty and col in ok_folds.columns:
            vc = ok_folds[col].value_counts(dropna=True)
            if len(vc):
                mode_val = vc.index[0]
                mode_params[col] = mode_val
                stability[col] = float(vc.iloc[0] / len(ok_folds))

    walk_forward_results = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "symbol": symbol,
                "interval": interval,
                "yearly_mode": bool(yearly_mode),
                "train_window": int(train_window),
                "test_window": int(test_window),
                "top_n": int(top_n),
                "fee_bps": float(fee_bps),
                "slippage_bps": float(slippage_bps),
                "num_folds": int(len(folds)),
                "num_ok_folds": int(len(ok_folds)),
                "percentage_positive_out_of_sample": pct_positive,
                "average_out_of_sample_return_pct": avg_oos_return,
                "worst_out_of_sample_return_pct": worst_oos_return,
                "worst_out_of_sample_drawdown": worst_oos_dd,
                "stability_tp1_range_fraction": stability.get("tp1_range_fraction"),
                "stability_stop_range_fraction": stability.get("stop_range_fraction"),
                "stability_tp1_close_fraction": stability.get("tp1_close_fraction"),
                "stability_friday_cutoff_hour_utc": stability.get("friday_cutoff_hour_utc"),
                "stability_exit_style": stability.get("exit_style"),
            }
        ]
    )

    root = Path(output_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / f"{symbol}_{interval}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    walk_forward_results.to_csv(out_dir / "walk_forward_results.csv", index=False)
    fold_train_results.to_csv(out_dir / "fold_train_results.csv", index=False)
    fold_test_results.to_csv(out_dir / "fold_test_results.csv", index=False)

    config = {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "yearly_mode": bool(yearly_mode),
        "train_window": int(train_window),
        "test_window": int(test_window),
        "top_n": int(top_n),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "risk_fraction": float(risk_fraction),
        "risk_base": risk_base,
        "max_leverage": float(max_leverage),
        "intrabar_policy": intrabar_policy,
        "accounting_version": "marked_equity_v1",
        "tp1_range_fractions": list(tp1_range_fractions),
        "stop_range_fractions": list(stop_range_fractions),
        "tp1_close_fractions": list(tp1_close_fractions),
        "tp2_range_fractions": list(tp2_range_fractions),
        "friday_cutoff_hours_utc": list(friday_cutoff_hours_utc),
        "exit_styles": list(exit_styles),
        "stop_mode": "swept_boundary_offset",
        "direction": "both",
        "folds": folds,
        "mode_parameters_by_fold_metric": {k: _to_jsonable(v) for k, v in mode_params.items()},
        "parameter_stability": {k: float(v) for k, v in stability.items()},
        "latest_cached_date": latest_cached_date,
    }
    if dataset_version is not None:
        config.update(
            experiment_metadata(
                dataset_version=dataset_version, interval=interval, start=None, end=None
            )
        )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {
        "out_dir": str(out_dir),
        "walk_forward_results": walk_forward_results,
        "fold_train_results": fold_train_results,
        "fold_test_results": fold_test_results,
        "fold_summary": fold_summary,
        "config": config,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run walk-forward optimisation on versioned canonical Binance candles.")
    ap.add_argument("--strategy", choices=["sweep_retest"], default="sweep_retest")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--train_window", type=int, default=1, help="Years in train window for rolling mode.")
    ap.add_argument("--test_window", type=int, default=1, help="Years in test window for rolling mode.")
    ap.add_argument("--yearly_mode", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fee_bps", type=float, default=DEFAULT_SWEEP_RETEST_FEE_BPS)
    ap.add_argument("--slippage_bps", type=float, default=DEFAULT_SWEEP_RETEST_SLIPPAGE_BPS)
    ap.add_argument("--top_n", type=int, default=3)
    ap.add_argument("--output_dir", default="data/walk_forward")
    ap.add_argument("--dataset_version", default=DEFAULT_DATASET_VERSION)

    ap.add_argument("--risk_fraction", type=float, default=0.02)
    ap.add_argument("--risk_base", choices=["current_equity", "initial_capital"], default="current_equity")
    ap.add_argument("--max_leverage", type=float, default=1.0)
    ap.add_argument("--intrabar_policy", choices=["conservative_stop_first", "target_first"], default="conservative_stop_first")
    ap.add_argument("--tp1_range_fractions", nargs="+", default=["0.35,0.40,0.45,0.50,0.55"])
    ap.add_argument("--stop_range_fractions", nargs="+", default=["0.75,1.0,1.1,1.25"])
    ap.add_argument("--tp1_close_fractions", nargs="+", default=["0.75,0.90,0.99,1.0"])
    ap.add_argument("--tp2_range_fractions", nargs="+", default=["1.0"])
    ap.add_argument("--friday_cutoff_hours_utc", nargs="+", default=["20,23"])
    ap.add_argument("--exit_styles", nargs="+", default=["partial_tp2,single_target"])

    args = ap.parse_args()

    if float(args.fee_bps) == 0.0 or float(args.slippage_bps) == 0.0:
        print("WARNING: fee_bps or slippage_bps is zero. Walk-forward may overestimate deployable performance.")

    exit_styles = [s.strip() for s in _parse_token_grid(args.exit_styles, name="exit_styles")]
    valid_exit_styles = {"partial_tp2", "single_target"}
    bad_exit_styles = [s for s in exit_styles if s not in valid_exit_styles]
    if bad_exit_styles:
        raise ValueError(f"Invalid exit_styles: {bad_exit_styles}")

    result = run_walk_forward(
        strategy=args.strategy,
        symbol=args.symbol,
        interval=args.interval,
        yearly_mode=bool(args.yearly_mode),
        train_window=int(args.train_window),
        test_window=int(args.test_window),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        top_n=int(args.top_n),
        output_dir=args.output_dir,
        risk_fraction=float(args.risk_fraction),
        risk_base=args.risk_base,
        max_leverage=float(args.max_leverage),
        intrabar_policy=args.intrabar_policy,
        dataset_version=args.dataset_version,
        tp1_range_fractions=parse_float_grid(args.tp1_range_fractions, name="tp1_range_fractions"),
        stop_range_fractions=parse_float_grid(args.stop_range_fractions, name="stop_range_fractions"),
        tp1_close_fractions=parse_float_grid(args.tp1_close_fractions, name="tp1_close_fractions"),
        tp2_range_fractions=parse_float_grid(args.tp2_range_fractions, name="tp2_range_fractions"),
        friday_cutoff_hours_utc=parse_int_grid(args.friday_cutoff_hours_utc, name="friday_cutoff_hours_utc"),
        exit_styles=exit_styles,
    )

    wf = result["walk_forward_results"].iloc[0].to_dict()
    print("=== Walk-forward complete ===")
    print(f"Saved outputs under: {result['out_dir']}")
    print(json.dumps(wf, indent=2))


if __name__ == "__main__":
    main()
