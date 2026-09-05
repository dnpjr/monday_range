from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from download_binance_data import run_cache_update


def _sample_klines(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    open_time = pd.to_datetime([r[0] for r in rows], utc=True)
    close_time = open_time + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1.0] * len(rows),
            "close_time": close_time,
            "quote_asset_volume": [1.0] * len(rows),
            "num_trades": [1] * len(rows),
            "taker_buy_base_asset_volume": [1.0] * len(rows),
            "taker_buy_quote_asset_volume": [1.0] * len(rows),
        }
    )


class DownloadBinanceDataTests(unittest.TestCase):
    def test_run_cache_update_force_returns_summary_and_writes_cache(self) -> None:
        incoming = _sample_klines(
            [
                ("2024-01-01 00:00:00+00:00", 100.0, 101.0, 99.0, 100.5),
                ("2024-01-01 01:00:00+00:00", 101.0, 102.0, 100.0, 101.5),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, patch("download_binance_data.download_klines", return_value=incoming) as dl:
            result = run_cache_update(
                symbol="BTCUSDT",
                interval="1h",
                lookback_days=30,
                force=True,
                cache_dir=tmp,
            )
            self.assertEqual(dl.call_count, 1)
            self.assertEqual(result["symbol"], "BTCUSDT")
            self.assertEqual(result["interval"], "1h")
            self.assertEqual(result["rows_saved"], 2)
            self.assertEqual(result["duplicates_removed"], 0)
            self.assertTrue(pathlib.Path(result["cache_file"]).exists())

    def test_run_cache_update_incremental_dedupes(self) -> None:
        existing = _sample_klines([("2024-01-01 00:00:00+00:00", 100.0, 101.0, 99.0, 100.5)])
        incoming = _sample_klines(
            [
                ("2024-01-01 00:00:00+00:00", 111.0, 112.0, 110.0, 111.5),
                ("2024-01-01 01:00:00+00:00", 102.0, 103.0, 101.0, 102.5),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = pathlib.Path(tmp) / "BTCUSDT_1h.csv"
            existing.to_csv(cache_path, index=False)
            with patch("download_binance_data.download_klines", return_value=incoming):
                result = run_cache_update(
                    symbol="BTCUSDT",
                    interval="1h",
                    start="2024-01-01",
                    end="2024-01-02",
                    force=False,
                    cache_dir=tmp,
                )
            self.assertEqual(result["rows_saved"], 2)
            self.assertEqual(result["duplicates_removed"], 1)


if __name__ == "__main__":
    unittest.main()
