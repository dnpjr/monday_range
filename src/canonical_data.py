from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .timeframe_utils import resample_ohlcv, timeframe_to_timedelta


DATASET_VERSION = "btcusdt_binance_spot_1h_v1"
SYMBOL = "BTCUSDT"
VENUE = "Binance spot"
SOURCE_ENDPOINT = "https://api.binance.com/api/v3/klines"
BASE_INTERVAL = "1h"
TIMEZONE = "UTC"
CANONICAL_START = pd.Timestamp("2021-05-21T15:00:00Z")
CANONICAL_END_EXCLUSIVE = pd.Timestamp("2026-05-20T15:00:00Z")

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = REPO_ROOT / "data" / "canonical"
DATA_FILE = CANONICAL_DIR / f"{DATASET_VERSION}.csv"
MANIFEST_FILE = CANONICAL_DIR / f"{DATASET_VERSION}.manifest.json"

CANONICAL_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]
NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_asset_volume",
    "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
]


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_commit(repo_root: str | Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def missing_timestamps(
    frame: pd.DataFrame, *, interval: str,
    start: Any | None = None, end_exclusive: Any | None = None,
) -> list[pd.Timestamp]:
    if frame.empty and (start is None or end_exclusive is None):
        return []
    times = pd.DatetimeIndex(
        pd.to_datetime(frame["open_time"], utc=True, errors="coerce").dropna()
    ).unique()
    if start is None:
        start = times.min()
    if end_exclusive is None:
        end_exclusive = times.max() + timeframe_to_timedelta(interval)
    expected = pd.date_range(
        _utc(start), _utc(end_exclusive) - timeframe_to_timedelta(interval),
        freq=timeframe_to_timedelta(interval),
    )
    return list(expected.difference(times))


def remove_incomplete_candles(
    frame: pd.DataFrame, *, interval: str, now_utc: Any | None = None,
) -> tuple[pd.DataFrame, int]:
    """Remove candles whose nominal interval end is later than the observation time."""
    out = frame.copy()
    if out.empty:
        return out, 0
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
    now = _utc(datetime.now(timezone.utc) if now_utc is None else now_utc)
    complete = out["open_time"] + timeframe_to_timedelta(interval) <= now
    return out.loc[complete].copy(), int((~complete).sum())


def validate_ohlcv(frame: pd.DataFrame) -> dict[str, Any]:
    required = ("open_time", "open", "high", "low", "close", "volume")
    missing_columns = [c for c in required if c not in frame]
    if missing_columns:
        return {"passed": False, "missing_columns": missing_columns, "invalid_row_count": len(frame)}
    times = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    values = {c: pd.to_numeric(frame[c], errors="coerce") for c in required[1:]}
    invalid_numeric = times.isna()
    for series in values.values():
        invalid_numeric = invalid_numeric | series.isna()
    invalid_ohlcv = (
        (values["high"] < values["open"])
        | (values["high"] < values["close"])
        | (values["high"] < values["low"])
        | (values["low"] > values["open"])
        | (values["low"] > values["close"])
        | (values["volume"] < 0)
    )
    invalid = invalid_numeric | invalid_ohlcv
    return {
        "passed": not bool(invalid.any()), "missing_columns": [],
        "invalid_numeric_count": int(invalid_numeric.sum()),
        "invalid_ohlcv_count": int(invalid_ohlcv.sum()),
        "invalid_row_count": int(invalid.sum()),
    }


def prepare_canonical_frame(
    frame: pd.DataFrame, *, start: Any = CANONICAL_START,
    end_exclusive: Any = CANONICAL_END_EXCLUSIVE,
    interval: str = BASE_INTERVAL, now_utc: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize, bound, validate, and inventory candles without filling gaps."""
    start_ts, end_ts = _utc(start), _utc(end_exclusive)
    if end_ts <= start_ts:
        raise ValueError("end_exclusive must be later than start")
    absent = [c for c in CANONICAL_COLUMNS if c not in frame]
    if absent:
        raise ValueError(f"Source data is missing canonical columns: {absent}")
    out = frame[CANONICAL_COLUMNS].copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[(out["open_time"] >= start_ts) & (out["open_time"] < end_ts)]
    duplicate_count = int(out.duplicated("open_time", keep="last").sum())
    out = out.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)
    out, incomplete_removed = remove_incomplete_candles(out, interval=interval, now_utc=now_utc)
    validation = validate_ohlcv(out)
    if not validation["passed"]:
        raise ValueError(f"Canonical OHLCV validation failed: {validation}")
    if out.empty:
        raise ValueError("Canonical data is empty after normalization")
    if not out["open_time"].is_monotonic_increasing or out["open_time"].duplicated().any():
        raise ValueError("Canonical timestamps are not sorted and unique")
    gaps = missing_timestamps(out, interval=interval, start=start_ts, end_exclusive=end_ts)
    return out, {
        "duplicate_count": duplicate_count,
        "incomplete_candles_removed": incomplete_removed,
        "missing_timestamp_count": len(gaps),
        "missing_timestamps": [ts.isoformat() for ts in gaps],
        "ohlcv_validation": validation,
        "chronologically_sorted": True,
        "unique_timestamps": True,
        "forward_filled": False,
        "interpolated": False,
    }


def write_canonical_csv(frame: pd.DataFrame, path: str | Path = DATA_FILE) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = frame[CANONICAL_COLUMNS].to_csv(
        index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ",
        float_format="%.12g", lineterminator="\n",
    ).encode("utf-8")
    if target.exists() and target.read_bytes() != rendered:
        raise FileExistsError(f"Refusing to overwrite immutable canonical data: {target}")
    target.write_bytes(rendered)
    return hashlib.sha256(rendered).hexdigest()


def write_manifest(manifest: dict[str, Any], path: str | Path = MANIFEST_FILE) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.exists() and target.read_bytes() != rendered:
        raise FileExistsError(f"Refusing to overwrite immutable canonical manifest: {target}")
    target.write_bytes(rendered)


def build_manifest(
    frame: pd.DataFrame, *, data_hash: str, quality: dict[str, Any],
    retrieval_timestamp: Any, generator_commit: str | None,
    gap_investigation: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION, "symbol": SYMBOL, "venue": VENUE,
        "market_type": "spot", "source_endpoint": SOURCE_ENDPOINT,
        "interval": BASE_INTERVAL, "timezone": TIMEZONE,
        "first_timestamp": frame["open_time"].iloc[0].isoformat(),
        "last_timestamp": frame["open_time"].iloc[-1].isoformat(),
        "window_end_exclusive": CANONICAL_END_EXCLUSIVE.isoformat(),
        "row_count": int(len(frame)),
        "retrieval_timestamp": _utc(retrieval_timestamp).isoformat(),
        "final_candle_completeness_checked": True,
        "completion_rule": "open_time + interval <= retrieval_timestamp",
        "missing_timestamp_count": quality["missing_timestamp_count"],
        "missing_timestamps": quality["missing_timestamps"],
        "duplicate_count": quality["duplicate_count"],
        "incomplete_candles_removed": quality["incomplete_candles_removed"],
        "ohlcv_validation": quality["ohlcv_validation"],
        "chronologically_sorted": quality["chronologically_sorted"],
        "unique_timestamps": quality["unique_timestamps"],
        "forward_filled": False, "interpolated": False,
        "data_file": DATA_FILE.name, "sha256": data_hash,
        "generator_commit": generator_commit,
        "resampling_policy": {
            "source_interval": "1h", "utc_anchored": True,
            "label": "left", "closed": "left",
            "ohlcv": {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"},
            "incomplete_periods": "drop if any expected 1h timestamp is absent",
        },
        "gap_investigation": gap_investigation,
    }


def load_manifest(path: str | Path = MANIFEST_FILE) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_canonical_dataset(
    *, dataset_version: str = DATASET_VERSION,
    data_dir: str | Path = CANONICAL_DIR, verify_hash: bool = True,
) -> pd.DataFrame:
    root = Path(data_dir)
    manifest = load_manifest(root / f"{dataset_version}.manifest.json")
    data_path = root / manifest["data_file"]
    if verify_hash and sha256_file(data_path) != manifest["sha256"]:
        raise ValueError(f"Canonical dataset hash mismatch: {data_path}")
    frame = pd.read_csv(data_path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    validation = validate_ohlcv(frame)
    if not validation["passed"]:
        raise ValueError(f"Canonical dataset failed validation: {validation}")
    if not frame["open_time"].is_monotonic_increasing or frame["open_time"].duplicated().any():
        raise ValueError("Canonical dataset timestamps are not sorted and unique")
    if len(frame) != int(manifest["row_count"]):
        raise ValueError("Canonical dataset row count does not match manifest")
    return frame


def load_canonical_ohlcv(
    interval: str = BASE_INTERVAL, *, dataset_version: str = DATASET_VERSION,
    data_dir: str | Path = CANONICAL_DIR,
) -> pd.DataFrame:
    indexed = load_canonical_dataset(
        dataset_version=dataset_version, data_dir=data_dir
    ).set_index("open_time")
    if interval == BASE_INTERVAL:
        return indexed
    return resample_ohlcv(
        indexed, base_interval=BASE_INTERVAL, target_interval=interval,
        drop_incomplete_final=True,
    )


def dataset_identity(
    dataset_version: str = DATASET_VERSION, *, data_dir: str | Path = CANONICAL_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(Path(data_dir) / f"{dataset_version}.manifest.json")
    return {
        "dataset_version": manifest["dataset_version"],
        "dataset_sha256": manifest["sha256"],
        "dataset_symbol": manifest["symbol"],
        "dataset_venue": manifest["venue"],
        "dataset_base_interval": manifest["interval"],
        "dataset_first_timestamp": manifest["first_timestamp"],
        "dataset_last_timestamp": manifest["last_timestamp"],
    }


def experiment_metadata(
    *, dataset_version: str = DATASET_VERSION, interval: str,
    start: str | None, end: str | None,
    data_dir: str | Path = CANONICAL_DIR,
) -> dict[str, Any]:
    return {
        **dataset_identity(dataset_version, data_dir=data_dir),
        "code_commit": code_commit(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timeframe": interval,
        "requested_start": start,
        "requested_end": end,
    }
