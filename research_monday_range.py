from __future__ import annotations
import argparse
import json
import pandas as pd

from src.data import download_ohlc
from src.features import add_monday_range
from src.research import analyze_weekly_sweep_signals, summarize_research

def main():
    ap = argparse.ArgumentParser(description="Research: Monday range sweep-and-fade statistics.")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out_csv", default="results/research_signals.csv")
    ap.add_argument("--out_json", default="results/research_summary.json")
    args = ap.parse_args()

    df = download_ohlc(args.symbol, interval=args.interval, period=args.period, start=args.start, end=args.end)
    df = add_monday_range(df)

    signals = analyze_weekly_sweep_signals(df)
    signals.to_csv(args.out_csv, index=False)

    summary = summarize_research(signals)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Research summary ===")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")

    print(f"\nSaved: {args.out_csv}")
    print(f"Saved: {args.out_json}")

if __name__ == "__main__":
    main()
