from __future__ import annotations

import argparse
import json

from src.paper_trader import run_paper_cycle


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper trading runner for Monday range strategy (no real orders).")
    ap.add_argument("--symbol", default="BTCUSDT", choices=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--interval", default="1h", choices=["15m", "1h", "4h", "1d"])
    ap.add_argument("--initial_cash", type=float, default=10_000.0)
    ap.add_argument("--risk_per_trade", type=float, default=100.0)
    ap.add_argument("--stop_mult", type=float, default=1.0)
    ap.add_argument("--tp1_frac", type=float, default=0.5)
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--lookback_days", type=int, default=30)
    ap.add_argument("--start", default=None, help="Optional fetch start date/time, e.g. 2024-01-01")
    ap.add_argument("--end", default=None, help="Optional fetch end date/time, e.g. 2025-01-01")
    ap.add_argument("--state_path", default="data/paper_state.json")
    ap.add_argument("--log_path", default="data/paper_trades.csv")
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="Process historical candles (simulation/backfill mode).")
    ap.add_argument("--friday_cutoff_hour_utc", type=int, default=23)
    ap.add_argument("--dry_run", action="store_true", help="Simulate cycle but do not write state/log files.")
    args = ap.parse_args()

    result = run_paper_cycle(
        symbol=args.symbol,
        interval=args.interval,
        initial_cash=args.initial_cash,
        risk_per_trade=args.risk_per_trade,
        stop_mult=args.stop_mult,
        tp1_frac=args.tp1_frac,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        lookback_days=args.lookback_days,
        start=args.start,
        end=args.end,
        state_path=args.state_path,
        log_path=args.log_path,
        use_cache=args.use_cache,
        dry_run=args.dry_run,
        backfill=args.backfill,
        friday_cutoff_hour_utc=args.friday_cutoff_hour_utc,
    )

    print(json.dumps(result, indent=2))
    if args.dry_run:
        print("\nDry run mode: no files were written.")
    else:
        print(f"\nUpdated state: {args.state_path}")
        print(f"Appended trade events: {args.log_path}")


if __name__ == "__main__":
    main()
