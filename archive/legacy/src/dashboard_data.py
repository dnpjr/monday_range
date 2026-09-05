from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

import pandas as pd

from .features import add_monday_range


@dataclass(frozen=True)
class CacheMeta:
    path: Path
    symbol: str | None
    interval: str | None


@dataclass(frozen=True)
class BacktestRunMeta:
    path: Path
    name: str
    modified_ts: float


def load_paper_state(path: str | Path = "data/paper_state.json") -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def load_paper_trades(path: str | Path = "data/paper_trades.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _parse_cache_name(path: Path) -> tuple[str | None, str | None]:
    # expected: SYMBOL_INTERVAL_start_end.{csv|parquet}, e.g. BTCUSDT_1h_...parquet
    m = re.match(r"^([A-Z0-9]+)_([0-9]+[mhd])_.+\.(csv|parquet)$", path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def find_latest_binance_cache(cache_dir: str | Path = "data/binance") -> CacheMeta | None:
    p = Path(cache_dir)
    if not p.exists():
        return None
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".csv", ".parquet"}]
    if not files:
        return None
    latest = max(files, key=lambda x: x.stat().st_mtime)
    symbol, interval = _parse_cache_name(latest)
    return CacheMeta(path=latest, symbol=symbol, interval=interval)


def load_binance_cache(meta: CacheMeta | None) -> pd.DataFrame:
    if meta is None:
        return pd.DataFrame()
    try:
        if meta.path.suffix.lower() == ".parquet":
            df = pd.read_parquet(meta.path)
        else:
            df = pd.read_csv(meta.path)
    except Exception:
        return pd.DataFrame()

    for col in ("open_time", "close_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True) if "open_time" in df.columns else df


def build_state_snapshot(
    state: dict[str, Any] | None,
    trades: pd.DataFrame,
    candles: pd.DataFrame,
    friday_cutoff_hour_utc: int = 23,
) -> dict[str, Any]:
    state = state or {}
    latest_price = float(candles["close"].iloc[-1]) if ("close" in candles.columns and len(candles)) else None
    last_event = trades.iloc[-1].to_dict() if len(trades) else None

    cash = float(state["cash"]) if state.get("cash") is not None else None
    side = state.get("position_side")
    qty = float(state.get("position_qty") or 0.0)
    entry = state.get("entry_price")

    unrealized = None
    if latest_price is not None and side in {"LONG", "SHORT"} and qty > 0 and entry is not None:
        e = float(entry)
        unrealized = (latest_price - e) * qty if side == "LONG" else (e - latest_price) * qty

    equity = None
    if cash is not None:
        equity = cash + (unrealized or 0.0)

    realized_pnl = float(trades["realized_pnl"].sum()) if "realized_pnl" in trades.columns and len(trades) else 0.0

    mon_high = mon_low = mon_mid = None
    if len(candles) and {"open_time", "open", "high", "low", "close"}.issubset(candles.columns):
        ohlc = candles.set_index("open_time")[["open", "high", "low", "close"]]
        feat = add_monday_range(ohlc)
        if len(feat):
            mon_high = feat["mon_high"].iloc[-1] if "mon_high" in feat.columns else None
            mon_low = feat["mon_low"].iloc[-1] if "mon_low" in feat.columns else None
            mon_mid = feat["mon_mid"].iloc[-1] if "mon_mid" in feat.columns else None

    return {
        "cash": cash,
        "equity": equity,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized,
        "position_side": side,
        "position_qty": qty,
        "entry_price": entry,
        "stop": state.get("stop"),
        "tp1": state.get("tp1"),
        "tp2": state.get("tp2"),
        "pending_entry": state.get("pending_entry"),
        "last_processed_open_time": state.get("last_processed_open_time"),
        "last_event": last_event,
        "event_count": int(len(trades)),
        "trade_count": int(trades["trade_id"].nunique()) if "trade_id" in trades.columns and len(trades) else 0,
        "current_price": latest_price,
        "mon_high": mon_high,
        "mon_low": mon_low,
        "mon_mid": mon_mid,
        "friday_cutoff_hour_utc": friday_cutoff_hour_utc,
    }


def compute_data_health(candles: pd.DataFrame, interval: str | None) -> dict[str, Any]:
    out = {
        "has_candles": bool(len(candles)),
        "latest_candle_age_seconds": None,
        "duplicate_candles": 0,
        "missing_candles_estimate": None,
        "timestamps_utc_ok": None,
    }
    if not len(candles) or "open_time" not in candles.columns:
        return out

    now = pd.Timestamp.now(tz="UTC")
    latest_time = candles["close_time"].max() if "close_time" in candles.columns else candles["open_time"].max()
    out["latest_candle_age_seconds"] = max(0.0, float((now - latest_time).total_seconds()))
    out["duplicate_candles"] = int(candles["open_time"].duplicated().sum())
    out["timestamps_utc_ok"] = str(candles["open_time"].dtype).endswith("UTC]")

    if len(candles) >= 2:
        diffs = candles["open_time"].sort_values().diff().dropna()
        if len(diffs):
            median = diffs.median()
            total_span = candles["open_time"].max() - candles["open_time"].min()
            if median > pd.Timedelta(0):
                expected = int(total_span / median) + 1
                out["missing_candles_estimate"] = max(0, expected - len(candles))
    return out


def reconstruct_completed_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "trade_id" not in trades.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for trade_id, grp in trades.groupby("trade_id", sort=True):
        grp = grp.sort_values("timestamp")
        entry = grp[grp["event"] == "ENTRY"]
        exits = grp[grp["position_qty_after"] == 0]
        if entry.empty or exits.empty:
            continue
        e = entry.iloc[0]
        x = exits.iloc[-1]
        pnl = float(grp["realized_pnl"].sum()) if "realized_pnl" in grp.columns else None
        rows.append(
            {
                "trade_id": trade_id,
                "side": e.get("side"),
                "entry_time": e.get("timestamp"),
                "entry_price": e.get("price"),
                "exit_time": x.get("timestamp"),
                "exit_price": x.get("price"),
                "exit_reason": x.get("event"),
                "pnl": pnl,
            }
        )
    return pd.DataFrame(rows)


def find_backtest_runs(backtest_root: str | Path = "data/backtests") -> list[BacktestRunMeta]:
    root = Path(backtest_root)
    if not root.exists():
        return []
    out: list[BacktestRunMeta] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not (p / "trades.csv").exists() or not (p / "summary.json").exists():
            continue
        try:
            mtime = (p / "summary.json").stat().st_mtime
        except OSError:
            continue
        out.append(BacktestRunMeta(path=p, name=p.name, modified_ts=mtime))
    return sorted(out, key=lambda x: x.modified_ts, reverse=True)


def load_backtest_run(meta: BacktestRunMeta | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    summary_path = meta.path / "summary.json"
    trades_path = meta.path / "trades.csv"
    if not summary_path.exists() or not trades_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return None
    try:
        trades = pd.read_csv(trades_path)
    except Exception:
        return None
    return {
        "path": str(meta.path),
        "name": meta.name,
        "summary": summary,
        "trades_count": int(len(trades)),
    }
