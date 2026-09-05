from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .execution_helpers import calculate_tp_levels, is_friday_cutoff_bar, is_stop_hit, is_tp_hit
from .features import add_monday_range
from .research import analyze_weekly_sweep_signals, summarize_research
from .signals import sweep_rejection_signal


def _entry_cutoff_rows(week: pd.DataFrame, friday_cutoff_hour_utc: int) -> pd.DataFrame:
    if week.empty:
        return week
    freq_delta = (week.index[1] - week.index[0]) if len(week.index) > 1 else pd.Timedelta(hours=1)
    keep: list[bool] = []
    for ts in week.index:
        ts_utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        bar_open = ts_utc
        bar_close = ts_utc + freq_delta
        keep.append(ts_utc.weekday() <= 4)
        if is_friday_cutoff_bar(bar_open, bar_close, friday_cutoff_hour_utc=friday_cutoff_hour_utc):
            break
    return week.loc[week.index[: len(keep)]]


def _range_bucket(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    if v <= 0.025:
        return "0-2.5%"
    if v <= 0.050:
        return "2.5-5.0%"
    if v <= 0.075:
        return "5.0-7.5%"
    return "7.5%+"


def _resolve_tp_fractions(
    *,
    tp1_range_fraction: float | None,
    tp2_range_fraction: float | None,
    tp1_to_mid: float,
    tp2_to_full: float,
) -> tuple[float, float]:
    if tp1_range_fraction is None:
        tp1_range_fraction = 0.5 * float(tp1_to_mid)
    if tp2_range_fraction is None:
        tp2_range_fraction = 1.0 if float(tp2_to_full) >= 1.0 else (0.5 + 0.5 * float(tp2_to_full))
    return float(tp1_range_fraction), float(tp2_range_fraction)


def _summary_stats(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": None, "median": None, "std": None, "p05": None, "p75": None, "p90": None, "p95": None, "count": 0}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "p05": float(s.quantile(0.05)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "count": int(len(s)),
    }


def _positive_time_hours(series: pd.Series) -> pd.Series:
    """Keep strictly positive elapsed-hour values and drop invalids."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s[s > 0]


def _calc_excursions(path: pd.DataFrame, *, side: str, entry_price: float, mon_range: float) -> tuple[float | None, float | None]:
    if path.empty or mon_range <= 0:
        return None, None
    if side == "LONG":
        mfe = float((float(path["high"].max()) - entry_price) / mon_range)
        mae = float((entry_price - float(path["low"].min())) / mon_range)
    else:
        mfe = float((entry_price - float(path["low"].min())) / mon_range)
        mae = float((float(path["high"].max()) - entry_price) / mon_range)
    return mfe, mae


def _time_to_event(entry_time: pd.Timestamp, event_time: pd.Timestamp | None, freq_delta: pd.Timedelta) -> tuple[float | None, int | None]:
    if event_time is None:
        return None, None
    dt_hours = float((event_time - entry_time).total_seconds() / 3600.0)
    if freq_delta.total_seconds() <= 0:
        return dt_hours, None
    bars = int(round((event_time - entry_time) / freq_delta))
    return dt_hours, bars


def _compute_vol_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    ret = np.log(close).diff()
    out["_ret"] = ret
    out["_vol_24"] = ret.rolling(window=24, min_periods=12).std()
    out["_sma50"] = close.rolling(window=50, min_periods=50).mean()
    out["_sma50_slope5"] = out["_sma50"] - out["_sma50"].shift(5)
    out["_ret_7d"] = close.pct_change(periods=7 * 24)
    return out


def _trend_regime_from_preentry(pre_close: float | None, pre_sma50: float | None, pre_slope: float | None, pre_ret7d: float | None) -> str:
    if pre_close is None or pre_sma50 is None or pd.isna(pre_close) or pd.isna(pre_sma50):
        return "unknown"
    if pre_close > pre_sma50 and pre_slope is not None and not pd.isna(pre_slope) and pre_slope > 0 and pre_ret7d is not None and not pd.isna(pre_ret7d) and pre_ret7d > 0:
        return "uptrend"
    if pre_close < pre_sma50 and pre_slope is not None and not pd.isna(pre_slope) and pre_slope < 0 and pre_ret7d is not None and not pd.isna(pre_ret7d) and pre_ret7d < 0:
        return "downtrend"
    return "sideways"


def analyze_original_sweep_retest_research(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Legacy compatibility path: reproduce original sweep/retest event analysis."""
    if ohlc.empty:
        return pd.DataFrame()
    return analyze_weekly_sweep_signals(add_monday_range(ohlc))


def summarize_original_sweep_retest_research(signals: pd.DataFrame) -> dict[str, Any]:
    """Legacy compatibility summary using original field names."""
    return summarize_research(signals)


def analyze_monday_range_research(
    ohlc: pd.DataFrame,
    *,
    tp1_range_fraction: float | None = 0.5,
    tp2_range_fraction: float | None = 1.0,
    tp1_to_mid: float = 1.0,  # legacy alias
    tp2_to_full: float = 1.0,  # legacy alias
    stop_mode: str = "opposite_boundary",
    stop_range_fraction: float = 1.0,
    stop_pct: float = 0.01,
    stop_mult: float = 1.0,
    friday_cutoff_hour_utc: int = 23,
    event_mode: str = "hypothetical_both_sides",
) -> pd.DataFrame:
    if ohlc.empty:
        return pd.DataFrame()
    df = _compute_vol_trend_features(add_monday_range(ohlc))
    tp1_rf, tp2_rf = _resolve_tp_fractions(
        tp1_range_fraction=tp1_range_fraction,
        tp2_range_fraction=tp2_range_fraction,
        tp1_to_mid=tp1_to_mid,
        tp2_to_full=tp2_to_full,
    )

    mode = str(event_mode).strip().lower()
    if mode == "strategy_entries":
        mode = "strategy_signal"
    elif mode == "sweep_events":
        mode = "hypothetical_both_sides"
    elif mode not in {"strategy_signal", "hypothetical_both_sides"}:
        raise ValueError(
            "Invalid event_mode. Use one of: strategy_entries, sweep_events, "
            "strategy_signal, hypothetical_both_sides."
        )

    out_rows: list[dict[str, Any]] = []
    weeks = df[["iso_year", "iso_week"]].drop_duplicates().itertuples(index=False, name=None)
    for iso_year, iso_week in weeks:
        week = df[(df["iso_year"] == iso_year) & (df["iso_week"] == iso_week)].copy()
        if week.empty:
            continue
        mon_high = week["mon_high"].iloc[0]
        mon_low = week["mon_low"].iloc[0]
        mon_mid = week["mon_mid"].iloc[0]
        mon_range = week["mon_range"].iloc[0]
        if pd.isna(mon_high) or pd.isna(mon_low) or pd.isna(mon_mid) or mon_range <= 0:
            continue

        tradeable = week[week["weekday"] > 0].copy()
        if tradeable.empty:
            continue
        freq_delta = (tradeable.index[1] - tradeable.index[0]) if len(tradeable.index) > 1 else pd.Timedelta(hours=1)

        candidates: list[tuple[str, pd.Timestamp, float, pd.Timestamp | None]] = []
        week_has_long_sweep = False
        week_has_short_sweep = False
        for i in range(1, len(week)):
            prev = week.iloc[i - 1]
            signal_side = sweep_rejection_signal(prev, mon_low=float(mon_low), mon_high=float(mon_high))
            if signal_side == "LONG":
                week_has_long_sweep = True
            if signal_side == "SHORT":
                week_has_short_sweep = True

        if mode == "strategy_signal":
            for i in range(1, len(week)):
                row = week.iloc[i]
                prev = week.iloc[i - 1]
                ts = row.name
                if int(row["weekday"]) >= 4:
                    continue
                if int(row["weekday"]) <= 0:
                    continue
                signal_side = sweep_rejection_signal(prev, mon_low=float(mon_low), mon_high=float(mon_high))
                if signal_side is None:
                    continue
                candidates.append((signal_side, ts, float(row["open"]), prev.name))
                break
        elif mode == "hypothetical_both_sides":
            entry_time = tradeable.index.min()
            entry_price = float(tradeable.loc[entry_time]["open"])
            candidates.append(("LONG", entry_time, entry_price, None))
            candidates.append(("SHORT", entry_time, entry_price, None))
        else:
            raise ValueError("Invalid event_mode.")

        range_pct = float(mon_range / abs(mon_mid)) if mon_mid != 0 else np.nan
        range_bucket = _range_bucket(range_pct)

        for side, entry_time, entry_price, signal_time in candidates:
            path = _entry_cutoff_rows(tradeable.loc[entry_time:], friday_cutoff_hour_utc=friday_cutoff_hour_utc)
            if path.empty:
                continue
            # Use completed bars after the entry bar to avoid same-bar event ambiguity.
            eval_path = path.iloc[1:].copy()
            if eval_path.empty:
                continue

            tp1, tp2 = calculate_tp_levels(
                side,
                mon_low=float(mon_low),
                mon_mid=float(mon_mid),
                mon_high=float(mon_high),
                tp1_range_fraction=tp1_rf,
                tp2_range_fraction=tp2_rf,
            )
            if stop_mode == "opposite_boundary":
                stop = float(mon_low - stop_mult * mon_range) if side == "LONG" else float(mon_high + stop_mult * mon_range)
            elif stop_mode == "range_fraction":
                stop = float(mon_low - stop_range_fraction * mon_range) if side == "LONG" else float(mon_high + stop_range_fraction * mon_range)
            elif stop_mode == "fixed_pct":
                stop = float(entry_price * (1.0 - stop_pct)) if side == "LONG" else float(entry_price * (1.0 + stop_pct))
            else:
                raise ValueError("stop_mode must be one of opposite_boundary/range_fraction/fixed_pct")

            tp1_time = None
            tp2_time = None
            stop_time = None
            for ts, r in eval_path.iterrows():
                if stop_time is None and is_stop_hit(side, candle_high=float(r["high"]), candle_low=float(r["low"]), stop_price=float(stop)):
                    stop_time = ts
                if tp1_time is None and stop_time is None and is_tp_hit(side, candle_high=float(r["high"]), candle_low=float(r["low"]), tp_price=float(tp1)):
                    tp1_time = ts
                if tp2_time is None and stop_time is None and is_tp_hit(side, candle_high=float(r["high"]), candle_low=float(r["low"]), tp_price=float(tp2)):
                    tp2_time = ts

            hit_tp1 = tp1_time is not None
            hit_tp2 = tp2_time is not None
            hit_stop = stop_time is not None
            tp1_before_stop = hit_tp1 and (not hit_stop or tp1_time < stop_time)
            tp2_before_stop = hit_tp2 and (not hit_stop or tp2_time < stop_time)
            tp2_after_tp1 = hit_tp2 and hit_tp1 and tp1_before_stop and (tp2_time >= tp1_time)
            stop_before_tp1 = hit_stop and (not hit_tp1 or stop_time <= tp1_time)

            if hit_stop and not tp2_before_stop:
                outcome = "STOP" if not hit_tp1 else "TP1_THEN_STOP"
            elif tp2_before_stop:
                outcome = "TP2"
            elif tp1_before_stop:
                outcome = "TP1_ONLY"
            else:
                outcome = "NO_HIT_BY_FRIDAY"

            pre_candidates = week.loc[week.index < entry_time]
            pre_row = pre_candidates.iloc[-1] if len(pre_candidates) else None
            pre_close = float(pre_row["close"]) if pre_row is not None else None
            pre_sma50 = float(pre_row["_sma50"]) if pre_row is not None and pd.notna(pre_row["_sma50"]) else None
            pre_slope = float(pre_row["_sma50_slope5"]) if pre_row is not None and pd.notna(pre_row["_sma50_slope5"]) else None
            pre_ret7d = float(pre_row["_ret_7d"]) if pre_row is not None and pd.notna(pre_row["_ret_7d"]) else None
            pre_vol = float(pre_row["_vol_24"]) if pre_row is not None and pd.notna(pre_row["_vol_24"]) else np.nan
            trend_regime = _trend_regime_from_preentry(pre_close, pre_sma50, pre_slope, pre_ret7d)

            full_mfe_r, full_mae_r = _calc_excursions(eval_path, side=side, entry_price=entry_price, mon_range=float(mon_range))
            tp1_path = eval_path.loc[:tp1_time] if tp1_time is not None else eval_path.iloc[0:0]
            tp2_path = eval_path.loc[:tp2_time] if tp2_time is not None else eval_path.iloc[0:0]
            stop_path = eval_path.loc[:stop_time] if stop_time is not None else eval_path.iloc[0:0]

            mfe_before_tp1_r, mae_before_tp1_r = _calc_excursions(tp1_path, side=side, entry_price=entry_price, mon_range=float(mon_range))
            mfe_before_tp2_r, mae_before_tp2_r = _calc_excursions(tp2_path, side=side, entry_price=entry_price, mon_range=float(mon_range))
            mfe_before_stop_r, mae_before_stop_r = _calc_excursions(stop_path, side=side, entry_price=entry_price, mon_range=float(mon_range))

            mae_before_eventual_target_r = None
            if tp2_before_stop:
                mae_before_eventual_target_r = mae_before_tp2_r
            elif tp1_before_stop:
                mae_before_eventual_target_r = mae_before_tp1_r

            mfe_before_eventual_stop_r = mfe_before_stop_r if outcome in {"STOP", "TP1_THEN_STOP"} else None

            time_to_tp1_hours, time_to_tp1_bars = _time_to_event(entry_time, tp1_time, freq_delta)
            time_to_tp2_hours, time_to_tp2_bars = _time_to_event(entry_time, tp2_time, freq_delta)
            time_to_stop_hours, time_to_stop_bars = _time_to_event(entry_time, stop_time, freq_delta)

            out_rows.append(
                {
                    "iso_year": int(iso_year),
                    "iso_week": int(iso_week),
                    "side": side,
                    "event_mode": mode,
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "mon_high": float(mon_high),
                    "mon_low": float(mon_low),
                    "mon_mid": float(mon_mid),
                    "mon_range": float(mon_range),
                    "mon_range_pct": range_pct,
                    "range_bucket": range_bucket,
                    "tp1_price": float(tp1),
                    "tp2_price": float(tp2),
                    "stop_price": float(stop),
                    "hit_tp1": bool(hit_tp1),
                    "hit_tp2": bool(hit_tp2),
                    "hit_stop": bool(hit_stop),
                    "tp1_before_stop": bool(tp1_before_stop),
                    "tp2_after_tp1": bool(tp2_after_tp1),
                    "stop_before_tp1": bool(stop_before_tp1),
                    "tp2_before_stop": bool(tp2_before_stop),
                    "time_to_tp1_hours": time_to_tp1_hours,
                    "time_to_tp2_hours": time_to_tp2_hours,
                    "time_to_stop_hours": time_to_stop_hours,
                    "time_to_tp1_bars": time_to_tp1_bars,
                    "time_to_tp2_bars": time_to_tp2_bars,
                    "time_to_stop_bars": time_to_stop_bars,
                    "mfe_in_range": full_mfe_r,
                    "mae_in_range": full_mae_r,
                    "mfe_before_tp1_r": mfe_before_tp1_r,
                    "mae_before_tp1_r": mae_before_tp1_r,
                    "mfe_before_tp2_r": mfe_before_tp2_r,
                    "mae_before_tp2_r": mae_before_tp2_r,
                    "mfe_before_stop_r": mfe_before_stop_r,
                    "mae_before_stop_r": mae_before_stop_r,
                    "mae_before_eventual_target_r": mae_before_eventual_target_r,
                    "mfe_before_eventual_stop_r": mfe_before_eventual_stop_r,
                    "outcome": outcome,
                    "pre_entry_volatility_24": pre_vol,
                    "trend_regime": trend_regime,
                    "week_has_long_sweep": bool(week_has_long_sweep),
                    "week_has_short_sweep": bool(week_has_short_sweep),
                }
            )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    valid_vol = out["pre_entry_volatility_24"].dropna()
    if len(valid_vol) >= 6 and valid_vol.nunique() >= 3:
        q1 = float(valid_vol.quantile(0.33))
        q2 = float(valid_vol.quantile(0.66))

        def _vol_regime(v: float) -> str:
            if pd.isna(v):
                return "unknown"
            if v <= q1:
                return "low_vol"
            if v <= q2:
                return "mid_vol"
            return "high_vol"

        out["vol_regime"] = out["pre_entry_volatility_24"].map(_vol_regime)
    else:
        out["vol_regime"] = "unknown"
    return out


def summarize_monday_range_research(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {"total_rows": 0}

    avg = lambda s: float(s.mean()) if len(s) else None
    tp1_time = _summary_stats(_positive_time_hours(results["time_to_tp1_hours"]))
    tp2_time = _summary_stats(_positive_time_hours(results["time_to_tp2_hours"]))
    stop_time = _summary_stats(_positive_time_hours(results["time_to_stop_hours"]))
    mfe_stats = _summary_stats(results["mfe_in_range"])
    mae_stats = _summary_stats(results["mae_in_range"])
    mae_tp1_winners = _summary_stats(results.loc[results["tp1_before_stop"], "mae_before_tp1_r"])
    mae_tp2_winners = _summary_stats(results.loc[results["tp2_before_stop"], "mae_before_tp2_r"])
    mfe_failures = _summary_stats(results.loc[results["outcome"].isin(["STOP", "TP1_THEN_STOP", "NO_HIT_BY_FRIDAY"]), "mfe_before_eventual_stop_r"])

    date_min = pd.to_datetime(results["entry_time"], utc=True, errors="coerce").min()
    date_max = pd.to_datetime(results["entry_time"], utc=True, errors="coerce").max()
    weeks = int(len(results[["iso_year", "iso_week"]].drop_duplicates()))
    long_count = int((results["side"] == "LONG").sum())
    short_count = int((results["side"] == "SHORT").sum())

    wk = results[["iso_year", "iso_week", "week_has_long_sweep", "week_has_short_sweep"]].drop_duplicates()
    weeks_with_any_sweep = int((wk["week_has_long_sweep"] | wk["week_has_short_sweep"]).sum()) if len(wk) else 0

    stop_guidance = {
        "tp1_winners_mae_r": mae_tp1_winners,
        "tp2_winners_mae_r": mae_tp2_winners,
        "failures_mfe_r": mfe_failures,
        "suggested_stop_tp1_conservative_r": mae_tp1_winners["p90"],
        "suggested_stop_tp2_conservative_r": mae_tp2_winners["p90"],
        "suggested_stop_very_conservative_r": max(
            [x for x in [mae_tp1_winners["p95"], mae_tp2_winners["p95"]] if x is not None],
            default=None,
        ),
    }

    return {
        "total_rows": int(len(results)),
        "total_weeks": weeks,
        "long_candidates": long_count,
        "short_candidates": short_count,
        "weeks_with_any_sweep": weeks_with_any_sweep,
        "p_week_with_sweep": (weeks_with_any_sweep / weeks) if weeks > 0 else None,
        "start_entry_time": (date_min.isoformat() if pd.notna(date_min) else None),
        "end_entry_time": (date_max.isoformat() if pd.notna(date_max) else None),
        "p_tp1_hit": avg(results["hit_tp1"].astype(float)),
        "p_tp2_hit": avg(results["hit_tp2"].astype(float)),
        "p_stop_hit": avg(results["hit_stop"].astype(float)),
        "p_tp1_before_stop": avg(results["tp1_before_stop"].astype(float)),
        "p_tp2_after_tp1": avg(results["tp2_after_tp1"].astype(float)),
        "p_stop_before_tp1": avg(results["stop_before_tp1"].astype(float)),
        "tp1_time_hours": tp1_time,
        "tp2_time_hours": tp2_time,
        "stop_time_hours": stop_time,
        "mfe_in_range_stats": mfe_stats,
        "mae_in_range_stats": mae_stats,
        "mae_before_tp1_winners_stats": mae_tp1_winners,
        "mae_before_tp2_winners_stats": mae_tp2_winners,
        "mfe_before_failure_stats": mfe_failures,
        "stop_guidance": stop_guidance,
    }


def _agg_prob_table(g: pd.core.groupby.DataFrameGroupBy, key: str) -> pd.DataFrame:
    out = (
        g.agg(
            rows=("side", "count"),
            p_tp1_hit=("hit_tp1", "mean"),
            p_tp2_hit=("hit_tp2", "mean"),
            p_stop_before_tp1=("stop_before_tp1", "mean"),
            p_tp1_before_stop=("tp1_before_stop", "mean"),
            p_tp2_after_tp1=("tp2_after_tp1", "mean"),
            median_mae_before_tp1_r=("mae_before_tp1_r", "median"),
            p90_mae_before_tp1_r=("mae_before_tp1_r", lambda s: float(pd.Series(s).quantile(0.90)) if len(pd.Series(s).dropna()) else np.nan),
            median_time_to_tp1_hours=("time_to_tp1_hours", lambda s: _positive_time_hours(pd.Series(s)).median()),
            median_time_to_tp2_hours=("time_to_tp2_hours", lambda s: _positive_time_hours(pd.Series(s)).median()),
            median_time_to_stop_hours=("time_to_stop_hours", lambda s: _positive_time_hours(pd.Series(s)).median()),
        )
        .reset_index()
    )
    if key == "range_bucket":
        order = pd.CategoricalDtype(["0-2.5%", "2.5-5.0%", "5.0-7.5%", "7.5%+", "unknown"], ordered=True)
        out["range_bucket"] = out["range_bucket"].astype(order)
    if key == "vol_regime":
        order = pd.CategoricalDtype(["low_vol", "mid_vol", "high_vol", "unknown"], ordered=True)
        out["vol_regime"] = out["vol_regime"].astype(order)
    if key == "trend_regime":
        order = pd.CategoricalDtype(["uptrend", "sideways", "downtrend", "unknown"], ordered=True)
        out["trend_regime"] = out["trend_regime"].astype(order)
    return out.sort_values(key).reset_index(drop=True)


def probability_tables(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if results.empty:
        empty = pd.DataFrame()
        return {
            "by_direction": empty,
            "by_year": empty,
            "by_range_bucket": empty,
            "by_vol_regime": empty,
            "by_trend_regime": empty,
        }
    return {
        "by_direction": _agg_prob_table(results.groupby("side"), "side"),
        "by_year": _agg_prob_table(results.groupby("iso_year"), "iso_year"),
        "by_range_bucket": _agg_prob_table(results.groupby("range_bucket"), "range_bucket"),
        "by_vol_regime": _agg_prob_table(results.groupby("vol_regime"), "vol_regime"),
        "by_trend_regime": _agg_prob_table(results.groupby("trend_regime"), "trend_regime"),
    }
