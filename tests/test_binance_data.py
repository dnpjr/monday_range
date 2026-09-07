from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.binance_data import _parse_klines_response, download_klines, filter_completed_klines


class BinanceParserTests(unittest.TestCase):
    def test_parse_klines_response_converts_types_and_dedupes(self) -> None:
        rows = [
            [1704067200000, "42000.0", "42100.0", "41950.0", "42050.0", "10.5", 1704070799999, "441000", 100, "5.0", "210000", "0"],
            [1704067200000, "42000.0", "42150.0", "41900.0", "42025.0", "11.0", 1704070799999, "442000", 101, "5.1", "211000", "0"],
            [1704070800000, "42050.0", "42200.0", "42000.0", "42120.0", "12.0", 1704074399999, "505000", 120, "6.0", "252000", "0"],
        ]

        df = _parse_klines_response(rows)

        self.assertEqual(len(df), 2)
        self.assertEqual(str(df["open_time"].dtype), "datetime64[ns, UTC]")
        self.assertEqual(str(df["close_time"].dtype), "datetime64[ns, UTC]")
        self.assertTrue(pd.api.types.is_float_dtype(df["open"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["high"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["low"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["close"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["volume"]))
        self.assertEqual(float(df.iloc[0]["high"]), 42150.0)


class BinanceDownloadTests(unittest.TestCase):
    def test_completed_filter_uses_nominal_interval_end(self) -> None:
        df = _parse_klines_response([
            [1704067200000, "100", "101", "99", "100", "1", 1704070799999, "100", 1, "0.5", "50", "0"],
            [1704070800000, "100", "101", "99", "100", "1", 1704074399999, "100", 1, "0.5", "50", "0"],
        ])
        before = filter_completed_klines(df, "1h", "2024-01-01T01:59:59.999Z")
        at_boundary = filter_completed_klines(df, "1h", "2024-01-01T02:00:00Z")
        self.assertEqual(len(before), 1)
        self.assertEqual(len(at_boundary), 2)

    def test_download_klines_paginates_1000_and_uses_cache_csv(self) -> None:
        start_ms = 1704067200000
        step_ms = 60 * 60 * 1000

        page1 = []
        for i in range(1000):
            open_ms = start_ms + i * step_ms
            close_ms = open_ms + step_ms - 1
            px = 100.0 + i
            page1.append([open_ms, str(px), str(px + 10), str(px - 10), str(px + 1), "10", close_ms, "1000", 10, "5", "500", "0"])

        open_ms_2 = start_ms + 1000 * step_ms
        close_ms_2 = open_ms_2 + step_ms - 1
        page2 = [[open_ms_2, "1100", "1110", "1090", "1101", "11", close_ms_2, "1100", 11, "6", "600", "0"]]

        responses = []
        for payload in (page1, page2):
            r = Mock()
            r.json.return_value = payload
            r.raise_for_status.return_value = None
            responses.append(r)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.binance_data._cache_dir", return_value=pathlib.Path(tmpdir)):
                with patch("src.binance_data.requests.get", side_effect=responses) as mock_get:
                    df = download_klines(
                        symbol="BTCUSDT",
                        interval="1h",
                        start=start_ms,
                        end=open_ms_2 + step_ms,
                        use_cache=True,
                        cache_format="csv",
                    )

                self.assertEqual(len(df), 1001)
                self.assertEqual(mock_get.call_count, 2)
                for call in mock_get.call_args_list:
                    self.assertEqual(call.kwargs["params"]["limit"], 1000)

                cache_files = list(pathlib.Path(tmpdir).glob("*.csv"))
                self.assertEqual(len(cache_files), 1)

                with patch("src.binance_data.requests.get") as mock_get_cached:
                    df_cached = download_klines(
                        symbol="BTCUSDT",
                        interval="1h",
                        start=start_ms,
                        end=open_ms_2 + step_ms,
                        use_cache=True,
                        cache_format="csv",
                    )
                    self.assertEqual(len(df_cached), 1001)
                    mock_get_cached.assert_not_called()

    def test_validation_rejects_unsupported_symbol_and_interval(self) -> None:
        with self.assertRaises(ValueError):
            download_klines("SOLUSDT", "1h", "2024-01-01", "2024-01-02", use_cache=False)

        with self.assertRaises(ValueError):
            download_klines("BTCUSDT", "5m", "2024-01-01", "2024-01-02", use_cache=False)


if __name__ == "__main__":
    unittest.main()
