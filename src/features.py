from __future__ import annotations
import pandas as pd

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weekday"] = out.index.weekday  # Monday=0
    iso = out.index.isocalendar()
    out["iso_year"] = iso.year.astype(int)
    out["iso_week"] = iso.week.astype(int)
    return out

def add_monday_range(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Monday high/low and derived features per ISO week."""
    out = add_calendar_features(df)

    monday = out[out["weekday"] == 0]
    stats = (
        monday.groupby(["iso_year", "iso_week"])[["high", "low"]]
        .agg(mon_high=("high", "max"), mon_low=("low", "min"))
    )

    out = out.join(stats, on=["iso_year", "iso_week"])
    out["mon_range"] = out["mon_high"] - out["mon_low"]
    out["mon_mid"] = (out["mon_high"] + out["mon_low"]) / 2.0
    out["is_tradeable"] = out["weekday"] > 0  # Tue+
    return out
