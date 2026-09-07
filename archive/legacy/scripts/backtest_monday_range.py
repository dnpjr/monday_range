from __future__ import annotations
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd

from src.data import download_ohlc
from src.binance_data import download_klines
from src.features import add_monday_range
from src.backtest import backtest_sweep_fade
from src.metrics import equity_metrics, trade_metrics

def _as_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

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
    ap.add_argument("--data_source", choices=["yahoo", "binance"], default="yahoo")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--initial_capital", type=float, default=10_000.0)
    ap.add_argument("--risk_per_trade", type=float, default=100.0)
    ap.add_argument("--stop_mult", type=float, default=1.0)
    ap.add_argument("--tp1_frac", type=float, default=0.5)
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--out_trades", default="results/trades.csv")
    ap.add_argument("--out_metrics", default="results/backtest_metrics.json")
    ap.add_argument("--out_plot", default="results/plots/equity_curve.png")
    args = ap.parse_args()

    if args.data_source == "yahoo":
        df = download_ohlc(args.symbol, interval=args.interval, period=args.period, start=args.start, end=args.end)
    else:
        end = _as_utc_timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC")
        start = _as_utc_timestamp(args.start) if args.start else (end - pd.DateOffset(years=5))
        df_binance = download_klines(
            symbol=args.symbol,
            interval=args.interval,
            start=start,
            end=end,
            use_cache=True,
            cache_format="parquet",
        )
        if df_binance.empty:
            raise ValueError("No Binance data downloaded for the requested range.")

        # Use only completed candles.
        now_utc = pd.Timestamp.now(tz="UTC")
        df_binance = df_binance[df_binance["close_time"] <= now_utc].copy()
        df = df_binance.set_index("open_time")[["open", "high", "low", "close"]].sort_index()

    df = add_monday_range(df)

    df_out, trades = backtest_sweep_fade(
        df,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        stop_mult=args.stop_mult,
        tp1_frac=args.tp1_frac,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    trades.to_csv(args.out_trades, index=False)

    bars_per_year = bars_per_year_from_interval(args.interval)
    em = equity_metrics(df_out["equity"], bars_per_year=bars_per_year)
    tm = trade_metrics(trades)
    bh_equity = args.initial_capital * (df_out["close"] / float(df_out["close"].iloc[0]))
    bm = equity_metrics(bh_equity, bars_per_year=bars_per_year)
    metrics = {"equity": em, "benchmark_buy_hold": bm, "trades": tm}

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
    print(
        "\nSummary:"
        f" total_return={em.get('total_return')},"
        f" annualized_return={em.get('annualized_return')},"
        f" sharpe={em.get('sharpe')},"
        f" max_drawdown={em.get('max_drawdown')},"
        f" win_rate={tm.get('win_rate')},"
        f" num_trades={tm.get('num_trades')},"
        f" avg_trade_return={tm.get('avg_trade_return')}"
    )
    print(f"\nSaved: {args.out_trades}")
    print(f"Saved: {args.out_metrics}")
    print(f"Saved: {args.out_plot}")

if __name__ == "__main__":
    main()
