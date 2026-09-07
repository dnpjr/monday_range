from __future__ import annotations

import pandas as pd


def sweep_rejection_signal(prev: pd.Series, mon_low: float, mon_high: float) -> str | None:
    """Return signal side from previous bar sweep rejection.

    Logic is intentionally identical to the previous inline implementation:
    - LONG: prev low sweeps below Monday low and closes back above it.
    - SHORT: prev high sweeps above Monday high and closes back below it.
    - If both are true, LONG wins (same precedence as before).
    """
    long_signal = (prev["low"] < mon_low) and (prev["close"] > mon_low)
    short_signal = (prev["high"] > mon_high) and (prev["close"] < mon_high)

    if long_signal:
        return "LONG"
    if short_signal:
        return "SHORT"
    return None
