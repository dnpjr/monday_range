from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class SignalOutcome:
    iso_year: int
    iso_week: int
    side: str  # LONG/SHORT
    entry_time: pd.Timestamp
    entry_price: float
    mon_high: float
    mon_low: float
    mon_mid: float
    mon_range: float
    hit_mid: bool
    hit_full: bool
    mae_to_mid: float | None  # in range units
    mae_to_full: float | None # in range units
    reward_ratio: float       # (full target - entry)/range in "R" units
    sweep_retest: bool

def analyze_weekly_sweep_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Scan each ISO week and measure what happens after the first sweep signal.

    - Entry: next bar open after a sweep-and-close-back bar.
    - Mid target: Monday mid.
    - Full target: opposite side of Monday range (Mon high for long, Mon low for short).
    - MAE: maximum adverse excursion before target, in units of Monday range.
    """
    if df.empty:
        return pd.DataFrame()

    out_rows: List[Dict[str, Any]] = []

    weeks = df[["iso_year", "iso_week"]].drop_duplicates().itertuples(index=False, name=None)

    for iso_year, iso_week in weeks:
        week = df[(df["iso_year"] == iso_year) & (df["iso_week"] == iso_week)].copy()
        if week.empty:
            continue

        monday = week[week["weekday"] == 0]
        if monday.empty:
            continue

        mon_high = float(monday["high"].max())
        mon_low  = float(monday["low"].min())
        mon_mid  = (mon_high + mon_low) / 2.0
        rng = mon_high - mon_low
        if rng <= 0:
            continue

        tradeable = week[week["weekday"] > 0]
        if tradeable.empty:
            continue

        # Find first signal
        entry_time = None
        entry_side = None
        entry_price = None
        sweep_extreme = None

        for ts, row in tradeable.iterrows():
            # previous bar within the week
            loc = week.index.get_loc(ts)
            if loc == 0:
                continue
            prev = week.iloc[loc - 1]

            # long: sweep below low then close back inside
            if (prev["low"] < mon_low) and (prev["close"] > mon_low):
                entry_side = "LONG"
                entry_time = ts
                entry_price = float(row["open"])
                sweep_extreme = float(prev["low"])
                break

            # short: sweep above high then close back inside
            if (prev["high"] > mon_high) and (prev["close"] < mon_high):
                entry_side = "SHORT"
                entry_time = ts
                entry_price = float(row["open"])
                sweep_extreme = float(prev["high"])
                break

        if entry_time is None:
            continue

        forward = week.loc[entry_time:].iloc[1:]  # after entry bar
        if forward.empty:
            continue

        target_mid = mon_mid
        target_full = mon_high if entry_side == "LONG" else mon_low

        reward_ratio = (target_full - entry_price) / rng if entry_side == "LONG" else (entry_price - target_full) / rng

        # hit mid
        hit_mid = False
        mae_mid = None
        for t, r in forward.iterrows():
            hit = (r["high"] >= target_mid) if entry_side == "LONG" else (r["low"] <= target_mid)
            if hit:
                hit_mid = True
                path = forward.loc[:t]
                if entry_side == "LONG":
                    worst = float(path["low"].min())
                    dd = max(0.0, entry_price - worst)
                else:
                    worst = float(path["high"].max())
                    dd = max(0.0, worst - entry_price)
                mae_mid = dd / rng
                break

        # hit full + sweep retest
        hit_full = False
        mae_full = None
        sweep_retest = False
        for t, r in forward.iterrows():
            if entry_side == "LONG":
                if float(r["low"]) < float(sweep_extreme):
                    sweep_retest = True
            else:
                if float(r["high"]) > float(sweep_extreme):
                    sweep_retest = True

            hit = (r["high"] >= target_full) if entry_side == "LONG" else (r["low"] <= target_full)
            if hit:
                hit_full = True
                path = forward.loc[:t]
                if entry_side == "LONG":
                    worst = float(path["low"].min())
                    dd = max(0.0, entry_price - worst)
                    if float(path["low"].min()) < float(sweep_extreme):
                        sweep_retest = True
                else:
                    worst = float(path["high"].max())
                    dd = max(0.0, worst - entry_price)
                    if float(path["high"].max()) > float(sweep_extreme):
                        sweep_retest = True
                mae_full = dd / rng
                break

        out_rows.append(dict(
            iso_year=iso_year,
            iso_week=iso_week,
            side=entry_side,
            entry_time=entry_time,
            entry_price=entry_price,
            mon_high=mon_high,
            mon_low=mon_low,
            mon_mid=mon_mid,
            mon_range=rng,
            hit_mid=hit_mid,
            hit_full=hit_full,
            mae_mid=mae_mid,
            mae_full=mae_full,
            reward_ratio=reward_ratio,
            sweep_retest=sweep_retest,
        ))

    return pd.DataFrame(out_rows)

def summarize_research(signals: pd.DataFrame) -> dict:
    if signals.empty:
        return {"total_signals": 0}

    total = len(signals)
    mid_hit = float(signals["hit_mid"].mean())
    full_hit = float(signals["hit_full"].mean())

    mae_mid = signals.loc[signals["hit_mid"], "mae_mid"]
    mae_full = signals.loc[signals["hit_full"], "mae_full"]

    sweep_break = float(signals["sweep_retest"].mean())
    winners = signals[signals["hit_full"]]
    winners_sweep_break = float(winners["sweep_retest"].mean()) if len(winners) else 0.0

    return {
        "total_signals": int(total),
        "p_hit_mid": mid_hit,
        "p_hit_full": full_hit,
        "avg_mae_mid": float(mae_mid.mean()) if len(mae_mid) else None,
        "p90_mae_mid": float(mae_mid.quantile(0.90)) if len(mae_mid) else None,
        "avg_mae_full": float(mae_full.mean()) if len(mae_full) else None,
        "p90_mae_full": float(mae_full.quantile(0.90)) if len(mae_full) else None,
        "avg_reward_ratio": float(signals["reward_ratio"].mean()),
        "p_sweep_retest": sweep_break,
        "p_sweep_retest_winners": winners_sweep_break,
    }
