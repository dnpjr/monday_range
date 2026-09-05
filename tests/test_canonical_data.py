from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.canonical_data import (
    CANONICAL_COLUMNS,
    DATASET_VERSION,
    experiment_metadata,
    load_canonical_dataset,
    missing_timestamps,
    prepare_canonical_frame,
    remove_incomplete_candles,
    sha256_file,
    write_canonical_csv,
)


def candle_frame(times: list[str]) -> pd.DataFrame:
    opens = pd.to_datetime(times, utc=True)
    return pd.DataFrame(
        {
            "open_time": opens,
            "open": [100.0 + i for i in range(len(opens))],
            "high": [102.0 + i for i in range(len(opens))],
            "low": [99.0 + i for i in range(len(opens))],
            "close": [101.0 + i for i in range(len(opens))],
            "volume": [10.0] * len(opens),
            "close_time": opens + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "quote_asset_volume": [1000.0] * len(opens),
            "num_trades": [10] * len(opens),
            "taker_buy_base_asset_volume": [5.0] * len(opens),
            "taker_buy_quote_asset_volume": [500.0] * len(opens),
        }
    )[CANONICAL_COLUMNS]


class CanonicalDataTests(unittest.TestCase):
    def test_prepare_sorts_and_keeps_last_duplicate(self) -> None:
        source = candle_frame([
            "2024-01-01T01:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"
        ])
        source.loc[2, "close"] = 777.0
        source.loc[2, "high"] = 800.0
        prepared, quality = prepare_canonical_frame(
            source,
            start="2024-01-01T00:00:00Z",
            end_exclusive="2024-01-01T02:00:00Z",
            now_utc="2024-01-02T00:00:00Z",
        )
        self.assertEqual(list(prepared["open_time"]), list(pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")))
        self.assertEqual(float(prepared.iloc[1]["close"]), 777.0)
        self.assertEqual(quality["duplicate_count"], 1)

    def test_incomplete_final_candle_removed_at_utc_boundary(self) -> None:
        source = candle_frame(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"])
        complete, removed = remove_incomplete_candles(
            source, interval="1h", now_utc="2024-01-01T01:59:59.999Z"
        )
        self.assertEqual(list(complete["open_time"]), [pd.Timestamp("2024-01-01T00:00:00Z")])
        self.assertEqual(removed, 1)
        complete_at_boundary, removed_at_boundary = remove_incomplete_candles(
            source, interval="1h", now_utc="2024-01-01T02:00:00Z"
        )
        self.assertEqual(len(complete_at_boundary), 2)
        self.assertEqual(removed_at_boundary, 0)

    def test_gap_detection_does_not_forward_fill(self) -> None:
        source = candle_frame(["2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z"])
        gaps = missing_timestamps(
            source, interval="1h", start="2024-01-01T00:00:00Z",
            end_exclusive="2024-01-01T03:00:00Z",
        )
        self.assertEqual(gaps, [pd.Timestamp("2024-01-01T01:00:00Z")])
        prepared, quality = prepare_canonical_frame(
            source, start="2024-01-01T00:00:00Z",
            end_exclusive="2024-01-01T03:00:00Z", now_utc="2024-01-02T00:00:00Z",
        )
        self.assertEqual(len(prepared), 2)
        self.assertFalse(quality["forward_filled"])

    def test_hash_manifest_and_loading(self) -> None:
        source = candle_frame(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"])
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            data_path = root / f"{DATASET_VERSION}.csv"
            digest = write_canonical_csv(source, data_path)
            manifest = {
                "dataset_version": DATASET_VERSION,
                "data_file": data_path.name,
                "sha256": digest,
                "row_count": 2,
                "symbol": "BTCUSDT",
                "venue": "Binance spot",
                "interval": "1h",
                "first_timestamp": "2024-01-01T00:00:00+00:00",
                "last_timestamp": "2024-01-01T01:00:00+00:00",
            }
            (root / f"{DATASET_VERSION}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_canonical_dataset(data_dir=root)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(digest, sha256_file(data_path))
            with data_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_canonical_dataset(data_dir=root)

    def test_experiment_metadata_records_dataset_and_utc_run_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = {
                "dataset_version": DATASET_VERSION, "sha256": "abc123",
                "symbol": "BTCUSDT", "venue": "Binance spot", "interval": "1h",
                "first_timestamp": "2024-01-01T00:00:00+00:00",
                "last_timestamp": "2024-01-02T00:00:00+00:00",
            }
            (root / f"{DATASET_VERSION}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            meta = experiment_metadata(
                interval="4h", start="2024-01-01", end="2024-01-02", data_dir=root
            )
            self.assertEqual(meta["dataset_sha256"], "abc123")
            self.assertEqual(meta["timeframe"], "4h")
            self.assertEqual(pd.Timestamp(meta["run_timestamp_utc"]).tzinfo, pd.Timestamp.now(tz="UTC").tzinfo)


if __name__ == "__main__":
    unittest.main()
