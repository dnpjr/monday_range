from __future__ import annotations
import argparse
import json
import matplotlib.pyplot as plt

from src.data import download_ohlc
from src.features import add_monday_range
from src.backtest import backtest_sweep_fade
from src.metrics import equity_metrics, trade_metrics

def bars_per_year_from_interval(interval: str) -> float:
    # crude mapping for Sharpe annualisation
    interval = interval.lower().strip()
    if interval.endswith("h"):
        hours = float(interval[:-1])
        return 365.0 * 24.0 / hours
    if interval.endswith("d"):
        days = float(interval[:-1])
        return 365.0 / days
    return 365.0 * 24.0 / 4.0  # fallback

def main():
    ap = argparse.ArgumentParser(description="Backtest: Monday range sweep-and-fade strategy.")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--initial_capital", type=float, default=10_000.0)
    ap.add_argument("--risk_per_trade", type=float, default=100.0)
    ap.add_argument("--stop_mult", type=float, default=1.0)
    ap.add_argument("--tp1_frac", type=float, default=0.5)
    ap.add_argument("--out_trades", default="results/trades.csv")
    ap.add_argument("--out_metrics", default="results/backtest_metrics.json")
    ap.add_argument("--out_plot", default="results/plots/equity_curve.png")
    args = ap.parse_args()

    df = download_ohlc(args.symbol, interval=args.interval, period=args.period, start=args.start, end=args.end)
    df = add_monday_range(df)

    df_out, trades = backtest_sweep_fade(
        df,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        stop_mult=args.stop_mult,
        tp1_frac=args.tp1_frac,
    )

    trades.to_csv(args.out_trades, index=False)

    bars_per_year = bars_per_year_from_interval(args.interval)
    em = equity_metrics(df_out["equity"], bars_per_year=bars_per_year)
    tm = trade_metrics(trades)
    metrics = {"equity": em, "trades": tm}

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # plot
    plt.figure()
    df_out["equity"].plot()
    plt.title(f"Equity curve: {args.symbol} ({args.interval})")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    plt.close()

    print("\n=== Backtest metrics ===")
    print(metrics)
    print(f"\nSaved: {args.out_trades}")
    print(f"Saved: {args.out_metrics}")
    print(f"Saved: {args.out_plot}")

if __name__ == "__main__":
    main()
