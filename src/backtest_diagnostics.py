from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_backtest import filter_ohlc_window, load_cached_ohlc


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_backtest_artifacts(backtest_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    root = Path(backtest_dir)
    trades_path = root / "trades.csv"
    summary_path = root / "summary.json"
    config_path = root / "config.json"
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades file: {trades_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    trades = pd.read_csv(trades_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    return trades, summary, config


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    if out.empty:
        out["net_pnl"] = []
        out["gross_pnl"] = []
        out["fees_paid"] = []
        return out

    for c in ("entry_time", "exit_time"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")

    out["net_pnl"] = pd.to_numeric(out["net_pnl"], errors="coerce") if "net_pnl" in out.columns else pd.to_numeric(out.get("pnl"), errors="coerce")
    out["gross_pnl"] = pd.to_numeric(out["gross_pnl"], errors="coerce") if "gross_pnl" in out.columns else pd.Series([pd.NA] * len(out), dtype="Float64")
    out["fees_paid"] = pd.to_numeric(out["fees_paid"], errors="coerce") if "fees_paid" in out.columns else pd.Series([pd.NA] * len(out), dtype="Float64")

    if "holding_hours" in out.columns:
        out["holding_hours"] = pd.to_numeric(out["holding_hours"], errors="coerce")
    elif "entry_time" in out.columns and "exit_time" in out.columns:
        out["holding_hours"] = (out["exit_time"] - out["entry_time"]).dt.total_seconds() / 3600.0
    else:
        out["holding_hours"] = pd.Series([pd.NA] * len(out), dtype="Float64")

    if out["fees_paid"].isna().all() and out["gross_pnl"].notna().any() and out["net_pnl"].notna().any():
        out["fees_paid"] = out["gross_pnl"] - out["net_pnl"]

    if "entry_time" in out.columns:
        out["entry_year"] = out["entry_time"].dt.year
        out["entry_day_name"] = out["entry_time"].dt.day_name()
        out["entry_hour_utc"] = out["entry_time"].dt.hour
    else:
        out["entry_year"] = pd.Series([pd.NA] * len(out), dtype="Float64")
        out["entry_day_name"] = pd.Series([None] * len(out), dtype="object")
        out["entry_hour_utc"] = pd.Series([pd.NA] * len(out), dtype="Float64")

    if "exit_time" in out.columns:
        out["exit_year"] = out["exit_time"].dt.year
    else:
        out["exit_year"] = pd.Series([pd.NA] * len(out), dtype="Float64")

    if "mon_range_pct" in out.columns:
        out["mon_range_pct"] = pd.to_numeric(out["mon_range_pct"], errors="coerce")
    elif "mon_range" in out.columns and "mon_mid" in out.columns:
        mon_range = pd.to_numeric(out["mon_range"], errors="coerce")
        mon_mid = pd.to_numeric(out["mon_mid"], errors="coerce").abs()
        out["mon_range_pct"] = mon_range / mon_mid.replace(0.0, pd.NA)
    else:
        out["mon_range_pct"] = pd.Series([pd.NA] * len(out), dtype="Float64")

    return out


def _group_stats(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if df.empty or by not in df.columns:
        return pd.DataFrame(columns=[by, "num_trades", "gross_pnl", "net_pnl", "fees_paid", "avg_holding_hours", "win_rate"])
    g = df.groupby(by, dropna=False)
    out = pd.DataFrame(
        {
            by: list(g.groups.keys()),
            "num_trades": g.size().astype(int).values,
            "gross_pnl": g["gross_pnl"].sum(min_count=1).values,
            "net_pnl": g["net_pnl"].sum(min_count=1).values,
            "fees_paid": g["fees_paid"].sum(min_count=1).values,
            "avg_holding_hours": g["holding_hours"].mean().values,
            "win_rate": g["net_pnl"].apply(lambda s: float((s > 0).mean()) if len(s) else 0.0).values,
        }
    )
    return out.sort_values(by=by).reset_index(drop=True)


def _range_bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "mon_range_pct" not in df.columns or df["mon_range_pct"].notna().sum() == 0:
        return pd.DataFrame(columns=["range_bucket", "num_trades", "gross_pnl", "net_pnl", "fees_paid", "avg_holding_hours", "win_rate"])
    bins = [-float("inf"), 0.01, 0.02, 0.03, 0.05, float("inf")]
    labels = ["<=1%", "1-2%", "2-3%", "3-5%", ">5%"]
    bucketed = df.copy()
    bucketed["range_bucket"] = pd.cut(bucketed["mon_range_pct"], bins=bins, labels=labels)
    out = _group_stats(bucketed, "range_bucket")
    if len(out):
        out["range_bucket"] = out["range_bucket"].astype(str)
    return out


def compute_buy_hold_benchmark(summary: dict[str, Any], config: dict[str, Any], cache_dir: str | Path = "data/binance") -> dict[str, Any] | None:
    symbol = summary.get("symbol") or config.get("symbol")
    interval = summary.get("interval") or config.get("interval")
    if not symbol or not interval:
        return None
    start = summary.get("start_used") or config.get("start")
    end = summary.get("end_used") or config.get("end")
    try:
        ohlc = load_cached_ohlc(
            symbol,
            interval,
            cache_dir=cache_dir,
            dataset_version=config.get("dataset_version"),
        )
        ohlc = filter_ohlc_window(ohlc, start, end)
    except (FileNotFoundError, ValueError):
        return None
    if ohlc.empty:
        return None

    start_close = float(ohlc["close"].iloc[0])
    end_close = float(ohlc["close"].iloc[-1])
    if start_close == 0:
        return None
    total_return_pct = (end_close / start_close - 1.0) * 100.0
    initial_cash = _safe_float(summary.get("initial_cash")) or _safe_float(config.get("initial_cash")) or 10_000.0
    final_equity = initial_cash * (1.0 + total_return_pct / 100.0)
    return {
        "symbol": symbol,
        "interval": interval,
        "start": str(ohlc.index.min()),
        "end": str(ohlc.index.max()),
        "start_close": start_close,
        "end_close": end_close,
        "buy_hold_return_pct": total_return_pct,
        "buy_hold_final_equity": float(final_equity),
    }


def compute_diagnostics(
    trades: pd.DataFrame,
    summary: dict[str, Any],
    config: dict[str, Any],
    *,
    cache_dir: str | Path = "data/binance",
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    t = _prepare_trades(trades)

    net_total = float(t["net_pnl"].sum()) if len(t) else 0.0
    gross_total = float(t["gross_pnl"].sum()) if len(t) and t["gross_pnl"].notna().any() else None
    fees_total = float(t["fees_paid"].sum()) if len(t) and t["fees_paid"].notna().any() else _safe_float(summary.get("total_fees_paid"))

    initial_cash = _safe_float(summary.get("initial_cash")) or _safe_float(config.get("initial_cash")) or 10_000.0
    fee_drag_pct = (fees_total / initial_cash * 100.0) if fees_total is not None and initial_cash > 0 else None
    avg_gross_trade = float(t["gross_pnl"].mean()) if len(t) and t["gross_pnl"].notna().any() else None
    avg_net_trade = float(t["net_pnl"].mean()) if len(t) else None
    avg_holding_hours = float(t["holding_hours"].mean()) if len(t) and t["holding_hours"].notna().any() else None

    by_year = _group_stats(t, "exit_year")
    by_direction = _group_stats(t, "side")
    by_exit_reason = _group_stats(t, "reason")
    expected_reasons = ["TP1", "TP2", "STOP", "FRIDAY"]
    if len(by_exit_reason):
        by_exit_reason = by_exit_reason.set_index("reason")
    else:
        by_exit_reason = pd.DataFrame(columns=["num_trades", "gross_pnl", "net_pnl", "fees_paid", "avg_holding_hours", "win_rate"]).set_index(pd.Index([], name="reason"))
    by_exit_reason = by_exit_reason.reindex(expected_reasons).fillna(
        {
            "num_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
            "avg_holding_hours": 0.0,
            "win_rate": 0.0,
        }
    )
    by_exit_reason["num_trades"] = by_exit_reason["num_trades"].astype(int)
    by_exit_reason = by_exit_reason.reset_index()
    by_range_bucket = _range_bucket_table(t)
    by_entry_day = _group_stats(t, "entry_day_name")
    by_entry_hour = _group_stats(t, "entry_hour_utc")

    benchmark = compute_buy_hold_benchmark(summary, config, cache_dir=cache_dir)

    diagnostics = {
        "num_trades": int(len(t)),
        "gross_pnl_total": gross_total,
        "net_pnl_total": net_total,
        "fees_total": fees_total,
        "fee_drag_pct_of_initial_cash": fee_drag_pct,
        "average_gross_trade_pnl": avg_gross_trade,
        "average_net_trade_pnl": avg_net_trade,
        "average_holding_hours": avg_holding_hours,
        "pnl_by_entry_day": {str(k): _safe_float(v) for k, v in zip(by_entry_day["entry_day_name"], by_entry_day["net_pnl"])} if len(by_entry_day) else {},
        "pnl_by_entry_hour_utc": {str(k): _safe_float(v) for k, v in zip(by_entry_hour["entry_hour_utc"], by_entry_hour["net_pnl"])} if len(by_entry_hour) else {},
        "buy_hold_benchmark": benchmark,
    }

    tables = {
        "by_year": by_year,
        "by_direction": by_direction,
        "by_exit_reason": by_exit_reason,
        "by_range_bucket": by_range_bucket,
    }
    return diagnostics, tables


def save_diagnostics_outputs(
    *,
    backtest_dir: str | Path,
    diagnostics: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    root = Path(backtest_dir)
    out_paths = {
        "diagnostics_json": root / "diagnostics.json",
        "by_year_csv": root / "diagnostics_by_year.csv",
        "by_direction_csv": root / "diagnostics_by_direction.csv",
        "by_exit_reason_csv": root / "diagnostics_by_exit_reason.csv",
    }
    with open(out_paths["diagnostics_json"], "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    tables["by_year"].to_csv(out_paths["by_year_csv"], index=False)
    tables["by_direction"].to_csv(out_paths["by_direction_csv"], index=False)
    tables["by_exit_reason"].to_csv(out_paths["by_exit_reason_csv"], index=False)

    if "by_range_bucket" in tables and len(tables["by_range_bucket"]):
        p = root / "diagnostics_by_range_bucket.csv"
        tables["by_range_bucket"].to_csv(p, index=False)
        out_paths["by_range_bucket_csv"] = p

    return out_paths
