from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.backtest_diagnostics import (
    compute_diagnostics,
    load_backtest_artifacts,
    save_diagnostics_outputs,
)


def run_backtest_diagnostics(
    *,
    backtest_dir: str,
    cache_dir: str = "data/binance",
) -> dict:
    trades, summary, config = load_backtest_artifacts(backtest_dir)
    diagnostics, tables = compute_diagnostics(
        trades,
        summary,
        config,
        cache_dir=cache_dir,
    )
    out_paths = save_diagnostics_outputs(
        backtest_dir=backtest_dir,
        diagnostics=diagnostics,
        tables=tables,
    )
    return {
        "diagnostics": diagnostics,
        "out_paths": {k: str(v) for k, v in out_paths.items()},
        "tables": tables,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Monday range backtest outputs and cost diagnostics.")
    ap.add_argument("--backtest_dir", required=True, help="Path containing trades.csv and summary.json")
    ap.add_argument("--cache_dir", default="data/binance", help="Path to cached Binance OHLCV CSV files")
    args = ap.parse_args()

    result = run_backtest_diagnostics(
        backtest_dir=args.backtest_dir,
        cache_dir=args.cache_dir,
    )

    print("=== Backtest diagnostics ===")
    print(json.dumps(result["diagnostics"], indent=2))
    for name, path in result["out_paths"].items():
        print(f"{name}: {Path(path)}")


if __name__ == "__main__":
    main()
