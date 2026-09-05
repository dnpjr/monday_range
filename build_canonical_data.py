from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.binance_data import download_klines
from src.canonical_data import (
    BASE_INTERVAL,
    CANONICAL_END_EXCLUSIVE,
    CANONICAL_START,
    DATASET_VERSION,
    MANIFEST_FILE,
    SOURCE_ENDPOINT,
    build_manifest,
    code_commit,
    prepare_canonical_frame,
    write_canonical_csv,
    write_manifest,
)
from src.timeframe_utils import resample_ohlcv


GAP_INVESTIGATION = [
    {
        "timestamps": [
            "2021-08-13T02:00:00+00:00", "2021-08-13T03:00:00+00:00",
            "2021-08-13T04:00:00+00:00", "2021-08-13T05:00:00+00:00",
        ],
        "authoritative_result": "Binance spot API omits these hours while returning 01:00 and 06:00.",
        "event": "Scheduled Binance spot trading system upgrade beginning 02:00 UTC.",
        "decision": "Preserve the four-hour gap; do not synthesize candles.",
    },
    {
        "timestamps": ["2021-09-29T07:00:00+00:00", "2021-09-29T08:00:00+00:00"],
        "authoritative_result": "Binance spot API omits these hours while returning 06:00 and 09:00.",
        "event": "Scheduled Binance spot trading system upgrade beginning 07:00 UTC.",
        "decision": "Preserve the two-hour gap; do not synthesize candles.",
    },
    {
        "timestamps": ["2023-03-24T13:00:00+00:00"],
        "authoritative_result": "Binance spot API omits 13:00 while returning a shortened zero-volume 12:00 candle and 14:00.",
        "event": "Binance suspended spot trading after a matching-engine issue.",
        "decision": "Preserve the one-hour gap and the authoritative 12:00 candle; do not synthesize data.",
    },
]


def _frame_hash(frame: pd.DataFrame) -> str:
    rendered = frame.reset_index().to_csv(
        index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ",
        float_format="%.12g", lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _comparison(canonical: pd.DataFrame, path: Path) -> dict[str, Any]:
    old = pd.read_csv(path)
    old["open_time"] = pd.to_datetime(old["open_time"], utc=True, errors="coerce")
    shared_columns = [c for c in ("open", "high", "low", "close", "volume") if c in old]
    left = canonical.set_index("open_time")[shared_columns]
    right = old.set_index("open_time")[shared_columns].apply(pd.to_numeric, errors="coerce")
    common = left.index.intersection(right.index)
    differing = int((~left.loc[common].eq(right.loc[common])).any(axis=1).sum())
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": int(len(old)),
        "first": old["open_time"].min().isoformat(),
        "last": old["open_time"].max().isoformat(),
        "shared_rows": int(len(common)),
        "canonical_only_rows": int(len(left.index.difference(right.index))),
        "legacy_only_rows": int(len(right.index.difference(left.index))),
        "differing_shared_ohlcv_rows": differing,
    }


def _report(
    *, manifest: dict[str, Any], monday: dict[str, Any], lab: dict[str, Any],
    derived: dict[str, Any], report_path: Path,
) -> None:
    gaps = "\n".join(
        f"- `{', '.join(item['timestamps'])}`: {item['authoritative_result']} "
        f"{item['event']} {item['decision']}" for item in GAP_INVESTIGATION
    )
    text = f"""# BTCUSDT Binance spot 1h canonical dataset v1

## Policy and coverage

- Dataset version: `{manifest['dataset_version']}`
- Authoritative file: `data/canonical/{manifest['data_file']}`
- Source: Binance spot public kline endpoint, `{SOURCE_ENDPOINT}`
- Symbol / interval / timezone: `BTCUSDT` / `1h` / `UTC`
- Frozen window: `{manifest['first_timestamp']}` through `{manifest['last_timestamp']}` (end exclusive `{manifest['window_end_exclusive']}`)
- Boundary rationale: preserve the established Monday cache timestamp window. The final 14:00 candle was re-fetched after completion; no newer candle was added.
- Rows: `{manifest['row_count']}`
- SHA-256: `{manifest['sha256']}`
- Retrieval timestamp: `{manifest['retrieval_timestamp']}`

Only completed candles are admitted. Timestamps are sorted and unique. No missing candle is forward-filled or interpolated.

## Data quality

- Duplicate source timestamps removed: `{manifest['duplicate_count']}`
- Incomplete candles removed from the frozen source response: `{manifest['incomplete_candles_removed']}`
- Missing expected hours: `{manifest['missing_timestamp_count']}`
- OHLCV validation passed: `{manifest['ohlcv_validation']['passed']}`
- Invalid numeric rows: `{manifest['ohlcv_validation']['invalid_numeric_count']}`
- Invalid OHLC/volume rows: `{manifest['ohlcv_validation']['invalid_ohlcv_count']}`

The validation requires high >= open/close/low, low <= open/close, high >= low, non-negative volume, and valid numeric values.

## Gap investigation

The official Binance spot API was queried around every gap and returned the surrounding candles but no candle for any of these seven timestamps:

{gaps}

The gaps are venue-history facts in the canonical data. They remain explicit.

## Differences from legacy caches

### Monday 1h cache

- Rows / coverage: `{monday['rows']}` / `{monday['first']}` through `{monday['last']}`
- Legacy file SHA-256: `{monday['sha256']}`
- Shared rows: `{monday['shared_rows']}`
- Canonical-only / legacy-only rows: `{monday['canonical_only_rows']}` / `{monday['legacy_only_rows']}`
- Shared rows with different OHLCV: `{monday['differing_shared_ohlcv_rows']}`

The one differing row is `2026-05-20 14:00 UTC`. The Monday cache captured it while forming; canonical v1 uses Binance's settled candle. The old file remains a legacy cache and was not overwritten.

### Crypto research lab 1h cache

- Rows / coverage: `{lab['rows']}` / `{lab['first']}` through `{lab['last']}`
- Legacy file SHA-256: `{lab['sha256']}`
- Shared rows: `{lab['shared_rows']}`
- Canonical-only / legacy-only rows: `{lab['canonical_only_rows']}` / `{lab['legacy_only_rows']}`
- Shared rows with different OHLCV: `{lab['differing_shared_ohlcv_rows']}`

The lab starts 24 hours later and ends 24 hours later. Its overlap agrees with canonical v1. The lab repository was read only throughout this work.

The separately downloaded Monday 4h file (SHA-256 `f7f09ff6b7acf69d1a2d2b7d1a667f7a256b81f819285c4a726b3d2c6ad00149`) and lab 1d file (SHA-256 `14775dac3c52ba4025583ff8b45135fcb68d762ecf24372ff7d4f4e566723592`) are legacy caches. Their final saved periods were incomplete. They are superseded for reproducible evaluation by deterministic views derived from canonical 1h data, but have not been deleted.

## Deterministic resampling

All bins use UTC `origin=start_day`, left labels, and left-closed intervals. Open is first, high is maximum, low is minimum, close is last, and volume is summed. A bin is kept only when every expected 1h timestamp is present. This drops initial/final partial bins and every 4h/daily period intersecting a genuine hourly gap.

- 4h: `{derived['4h']['rows']}` rows, `{derived['4h']['first']}` through `{derived['4h']['last']}`, deterministic frame SHA-256 `{derived['4h']['sha256']}`
- 1d: `{derived['1d']['rows']}` rows, `{derived['1d']['first']}` through `{derived['1d']['last']}`, deterministic frame SHA-256 `{derived['1d']['sha256']}`

Derived views are generated on demand and are not separate authoritative datasets.

## Data flow and provenance

- `run_backtest.py`, `run_parameter_sweep.py`, and `run_walk_forward.py` default to canonical v1.
- Higher-timeframe evaluation is derived from canonical 1h through `src.canonical_data.load_canonical_ohlcv`.
- Diagnostics use the dataset version recorded by the originating run.
- The dashboard Data Manager and paper cycle remain live/runtime paths. Their downloaded evaluation candles exclude nominally incomplete bars; the paper cycle also applies its existing close-time guard.
- The dashboard's standard backtest and walk-forward actions inherit the canonical runner defaults. Its standalone Research Lab and Range Sweep exploratory pages still read selected live/legacy caches; their outputs must not be treated as canonical evaluation until those callers explicitly select a canonical dataset.
- Historical `data/binance/*.csv`, saved backtests, sweeps, walk-forward outputs, and paper state remain legacy/runtime artifacts.

Future run configuration records dataset version/hash, code commit, strategy configuration, accounting version, risk base, leverage, fees, slippage, intrabar policy, timeframe, requested date range, and UTC run timestamp.

## Research interpretation

Old performance artifacts are retained but classified as legacy because they lack an immutable dataset identity and may include independently downloaded higher timeframes or a forming final candle. No performance search or strategy evaluation was run while producing this dataset.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable BTCUSDT Binance spot canonical dataset v1.")
    parser.add_argument("--source-csv", default=None, help="Verified raw API response; omit to download the frozen window.")
    parser.add_argument("--retrieval-timestamp", default=None)
    parser.add_argument("--monday-cache", default="data/binance/BTCUSDT_1h.csv")
    parser.add_argument("--lab-cache", required=True, help="Read-only lab BTCUSDT 1h cache used only for comparison.")
    parser.add_argument("--report", default="docs/data/btcusdt_binance_spot_1h_v1.md")
    args = parser.parse_args()

    retrieval = pd.Timestamp.now(tz="UTC") if args.retrieval_timestamp is None else pd.Timestamp(args.retrieval_timestamp)
    source = (
        pd.read_csv(args.source_csv)
        if args.source_csv
        else download_klines(
            "BTCUSDT", "1h", CANONICAL_START, CANONICAL_END_EXCLUSIVE,
            use_cache=False, include_incomplete=False, now_utc=retrieval,
        )
    )
    canonical, quality = prepare_canonical_frame(source, now_utc=retrieval)
    data_hash = write_canonical_csv(canonical)
    indexed = canonical.set_index("open_time")
    derived: dict[str, Any] = {}
    for interval in ("4h", "1d"):
        frame = resample_ohlcv(
            indexed, base_interval="1h", target_interval=interval,
            drop_incomplete_final=True,
        )
        derived[interval] = {
            "rows": int(len(frame)), "first": frame.index.min().isoformat(),
            "last": frame.index.max().isoformat(), "sha256": _frame_hash(frame),
            "dropped_incomplete_periods": frame.attrs.get("dropped_incomplete_periods", []),
        }
    manifest = build_manifest(
        canonical, data_hash=data_hash, quality=quality,
        retrieval_timestamp=retrieval, generator_commit=code_commit(),
        gap_investigation=GAP_INVESTIGATION,
    )
    manifest["derived_views"] = derived
    monday = _comparison(canonical, Path(args.monday_cache))
    lab = _comparison(canonical, Path(args.lab_cache))
    manifest["legacy_comparisons"] = {"monday_1h": monday, "lab_1h": lab}
    write_manifest(manifest)
    _report(
        manifest=manifest, monday=monday, lab=lab, derived=derived,
        report_path=Path(args.report),
    )
    print(json.dumps({"manifest": str(MANIFEST_FILE), **manifest}, indent=2))


if __name__ == "__main__":
    main()
