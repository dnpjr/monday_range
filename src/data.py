from __future__ import annotations
import pandas as pd
import yfinance as yf

def download_ohlc(
    symbol: str,
    interval: str = "4h",
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download OHLC data using yfinance.

    Returns a DataFrame with columns: open, high, low, close and a tz-naive DatetimeIndex.
    """
    df = yf.download(
        symbol,
        interval=interval,
        period=None if (start and end) else period,
        start=start,
        end=end,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError("No data downloaded. Check symbol/interval/period.")

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df[["open", "high", "low", "close"]].copy()
    df = df.dropna()

    # normalize timezone to naive for consistent weekly grouping
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    return df.sort_index()
