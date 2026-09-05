from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .timeframe_utils import can_resample, resample_ohlcv, timeframe_to_timedelta


REFERENCE_TIMEFRAMES = {"previous_day", "previous_week", "previous_month"}
EXECUTION_TIMEFRAMES = {"1h", "2h", "4h", "6h", "12h", "1d"}
CONFIRMATION_MODES = {"wick_only", "close_back_inside", "close_outside"}
EVENT_MODES = {"first_sweep_per_reference_period", "all_sweeps"}
HOLDING_WINDOW_MODES = {"until_end_of_reference_period", "fixed_n_candles", "fixed_hours"}


@dataclass(frozen=True)
class SweepAnalysisConfig:
    reference_timeframe: str = "previous_day"
    execution_timeframe: str = "1h"
    confirmation_mode: str = "close_back_inside"
    event_mode: str = "first_sweep_per_reference_period"
    holding_window_mode: str = "until_end_of_reference_period"
    fixed_n_candles: int = 48
    fixed_hours: int = 48
    atr_period: int = 20
    sma_period: int = 50
    trend_return_period: int = 20


def _validate_config(config: SweepAnalysisConfig) -> None:
    if config.reference_timeframe not in REFERENCE_TIMEFRAMES:
        raise ValueError(f"reference_timeframe must be one of {sorted(REFERENCE_TIMEFRAMES)}")
    if config.execution_timeframe not in EXECUTION_TIMEFRAMES:
        raise ValueError(f"execution_timeframe must be one of {sorted(EXECUTION_TIMEFRAMES)}")
    if config.confirmation_mode not in CONFIRMATION_MODES:
        raise ValueError(f"confirmation_mode must be one of {sorted(CONFIRMATION_MODES)}")
    if config.event_mode not in EVENT_MODES:
        raise ValueError(f"event_mode must be one of {sorted(EVENT_MODES)}")
    if config.holding_window_mode not in HOLDING_WINDOW_MODES:
        raise ValueError(f"holding_window_mode must be one of {sorted(HOLDING_WINDOW_MODES)}")
    if config.fixed_n_candles <= 0:
        raise ValueError("fixed_n_candles must be positive.")
    if config.fixed_hours <= 0:
        raise ValueError("fixed_hours must be positive.")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    elif "open_time" in df.columns:
        out = df.copy()
        out["open_time"] = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
        out = out.dropna(subset=["open_time"]).set_index("open_time")
    else:
        raise ValueError("Input data must have DatetimeIndex or open_time column.")

    out = out.sort_index()
    for c in ("open", "high", "low", "close"):
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["open", "high", "low", "close"])


def _period_start(index: pd.DatetimeIndex, reference_timeframe: str) -> pd.Series:
    ts = pd.Series(index, index=index)
    if reference_timeframe == "previous_day":
        return ts.dt.floor("D")
    if reference_timeframe == "previous_week":
        floor_day = ts.dt.floor("D")
        return floor_day - pd.to_timedelta(floor_day.dt.weekday, unit="D")
    if reference_timeframe == "previous_month":
        # First day of current month in UTC.
        return pd.to_datetime(ts.dt.strftime("%Y-%m-01"), utc=True)
    raise ValueError(f"Unsupported reference_timeframe: {reference_timeframe}")


def _build_reference_ranges(exec_df: pd.DataFrame, reference_timeframe: str) -> pd.DataFrame:
    out = exec_df.copy()
    out["period_start"] = _period_start(out.index, reference_timeframe)

    grouped = out.groupby("period_start").agg(ref_high=("high", "max"), ref_low=("low", "min"))
    grouped["ref_mid"] = (grouped["ref_high"] + grouped["ref_low"]) / 2.0
    grouped["ref_range"] = grouped["ref_high"] - grouped["ref_low"]
    prev = grouped.shift(1).rename(
        columns={
            "ref_high": "prev_ref_high",
            "ref_low": "prev_ref_low",
            "ref_mid": "prev_ref_mid",
            "ref_range": "prev_ref_range",
        }
    )
    out = out.merge(prev, left_on="period_start", right_index=True, how="left")
    return out


def _range_bucket(range_pct: float | None) -> str:
    if range_pct is None or pd.isna(range_pct):
        return "unknown"
    x = float(range_pct)
    if x <= 0.025:
        return "0-2.5%"
    if x <= 0.05:
        return "2.5-5.0%"
    if x <= 0.075:
        return "5.0-7.5%"
    return "7.5%+"


def _confirmation_ok(side: str, row: pd.Series, confirmation_mode: str, ref_high: float, ref_low: float) -> bool:
    if confirmation_mode == "wick_only":
        return True
    close = float(row["close"])
    if side == "high_sweep":
        if confirmation_mode == "close_back_inside":
            return close < ref_high
        if confirmation_mode == "close_outside":
            return close > ref_high
    else:
        if confirmation_mode == "close_back_inside":
            return close > ref_low
        if confirmation_mode == "close_outside":
            return close < ref_low
    return False


def _time_to_hours(start: pd.Timestamp, event_time: pd.Timestamp | None) -> float | None:
    if event_time is None:
        return None
    return float((event_time - start).total_seconds() / 3600.0)


def _first_hit_time(path: pd.DataFrame, predicate) -> pd.Timestamp | None:
    for ts, row in path.iterrows():
        if predicate(row):
            return ts
    return None


def _calc_excursions(path: pd.DataFrame, side: str, anchor_price: float, ref_range: float) -> tuple[float | None, float | None]:
    if path.empty or ref_range <= 0:
        return None, None
    if side == "high_sweep":
        mfe = (anchor_price - float(path["low"].min())) / ref_range
        mae = (float(path["high"].max()) - anchor_price) / ref_range
    else:
        mfe = (float(path["high"].max()) - anchor_price) / ref_range
        mae = (anchor_price - float(path["low"].min())) / ref_range
    return float(mfe), float(mae)


def _resolve_window_end_pos(
    frame: pd.DataFrame,
    *,
    event_pos: int,
    event_ts: pd.Timestamp,
    event_period_start: pd.Timestamp,
    reference_timeframe: str,
    holding_window_mode: str,
    fixed_n_candles: int,
    fixed_hours: int,
) -> int:
    if holding_window_mode == "fixed_n_candles":
        return min(len(frame) - 1, event_pos + fixed_n_candles)
    if holding_window_mode == "fixed_hours":
        end_ts = event_ts + pd.Timedelta(hours=fixed_hours)
        eligible = frame.index[(frame.index > event_ts) & (frame.index <= end_ts)]
        if len(eligible) == 0:
            return event_pos
        return int(frame.index.get_indexer([eligible[-1]])[0])

    if reference_timeframe == "previous_day":
        key = frame.index.floor("D")
        same = np.where(key == event_period_start)[0]
    elif reference_timeframe == "previous_week":
        week_start = frame.index.floor("D") - pd.to_timedelta(frame.index.weekday, unit="D")
        same = np.where(week_start == event_period_start)[0]
    else:
        month_start = pd.to_datetime(frame.index.strftime("%Y-%m-01"), utc=True)
        same = np.where(month_start == event_period_start)[0]
    same = same[same > event_pos]
    if len(same) == 0:
        return event_pos
    return int(same[-1])


def _build_pre_event_regimes(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    valid_vol = out["pre_event_volatility"].dropna()
    if len(valid_vol) >= 6 and valid_vol.nunique() >= 3:
        q1 = float(valid_vol.quantile(1 / 3))
        q2 = float(valid_vol.quantile(2 / 3))

        def vol_bucket(v: float) -> str:
            if pd.isna(v):
                return "unknown"
            if v <= q1:
                return "low_vol"
            if v <= q2:
                return "mid_vol"
            return "high_vol"

        out["vol_regime"] = out["pre_event_volatility"].map(vol_bucket)
    else:
        out["vol_regime"] = "unknown"
    return out


def _compute_trend_label(pre_close: float | None, pre_sma: float | None, pre_slope: float | None, pre_ret: float | None) -> str:
    if pre_close is None or pre_sma is None or pd.isna(pre_close) or pd.isna(pre_sma):
        return "unknown"
    if pre_close > pre_sma and pre_slope is not None and not pd.isna(pre_slope) and pre_slope > 0 and pre_ret is not None and not pd.isna(pre_ret) and pre_ret > 0:
        return "uptrend"
    if pre_close < pre_sma and pre_slope is not None and not pd.isna(pre_slope) and pre_slope < 0 and pre_ret is not None and not pd.isna(pre_ret) and pre_ret < 0:
        return "downtrend"
    return "neutral"


def analyze_range_sweeps(
    ohlcv: pd.DataFrame,
    *,
    base_interval: str,
    config: SweepAnalysisConfig = SweepAnalysisConfig(),
) -> pd.DataFrame:
    _validate_config(config)
    raw = _normalize_ohlcv(ohlcv)
    if raw.empty:
        return pd.DataFrame()
    if not can_resample(base_interval, config.execution_timeframe):
        raise ValueError(
            f"Execution timeframe {config.execution_timeframe} is lower than cached base interval {base_interval}."
        )

    exec_df = resample_ohlcv(
        raw,
        base_interval=base_interval,
        target_interval=config.execution_timeframe,
        drop_incomplete_final=True,
    )
    if exec_df.empty:
        return pd.DataFrame()

    exec_df = _build_reference_ranges(exec_df, config.reference_timeframe)
    close = pd.to_numeric(exec_df["close"], errors="coerce")
    returns = np.log(close).diff()
    exec_df["_vol"] = returns.rolling(window=config.atr_period, min_periods=max(5, config.atr_period // 2)).std().shift(1)
    exec_df["_sma"] = close.rolling(window=config.sma_period, min_periods=config.sma_period).mean().shift(1)
    exec_df["_sma_slope"] = (exec_df["_sma"] - exec_df["_sma"].shift(5)).shift(1)
    exec_df["_ret_n"] = close.pct_change(config.trend_return_period).shift(1)
    exec_df["_close_pre"] = close.shift(1)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[pd.Timestamp, str]] = set()
    step = timeframe_to_timedelta(config.execution_timeframe)

    for pos, (ts, row) in enumerate(exec_df.iterrows()):
        ref_high = row.get("prev_ref_high")
        ref_low = row.get("prev_ref_low")
        ref_mid = row.get("prev_ref_mid")
        ref_range = row.get("prev_ref_range")
        period_start = row.get("period_start")
        if pd.isna(ref_high) or pd.isna(ref_low) or pd.isna(ref_mid) or pd.isna(ref_range) or float(ref_range) <= 0:
            continue
        ref_high = float(ref_high)
        ref_low = float(ref_low)
        ref_mid = float(ref_mid)
        ref_range = float(ref_range)

        sides: list[str] = []
        if float(row["high"]) > ref_high and _confirmation_ok("high_sweep", row, config.confirmation_mode, ref_high, ref_low):
            sides.append("high_sweep")
        if float(row["low"]) < ref_low and _confirmation_ok("low_sweep", row, config.confirmation_mode, ref_high, ref_low):
            sides.append("low_sweep")

        for side in sides:
            key = (period_start, side)
            if config.event_mode == "first_sweep_per_reference_period" and key in seen:
                continue
            seen.add(key)

            start_pos = pos + 1
            if start_pos >= len(exec_df):
                continue
            window_end_pos = _resolve_window_end_pos(
                exec_df,
                event_pos=pos,
                event_ts=ts,
                event_period_start=period_start,
                reference_timeframe=config.reference_timeframe,
                holding_window_mode=config.holding_window_mode,
                fixed_n_candles=config.fixed_n_candles,
                fixed_hours=config.fixed_hours,
            )
            if window_end_pos < start_pos:
                continue
            path = exec_df.iloc[start_pos : window_end_pos + 1].copy()
            if path.empty:
                continue

            sweep_close = float(row["close"])
            overshoot_r = ((float(row["high"]) - ref_high) / ref_range) if side == "high_sweep" else ((ref_low - float(row["low"])) / ref_range)
            midpoint_hit = _first_hit_time(path, lambda r: float(r["low"]) <= ref_mid if side == "high_sweep" else float(r["high"]) >= ref_mid)
            opposite_hit = _first_hit_time(path, lambda r: float(r["low"]) <= ref_low if side == "high_sweep" else float(r["high"]) >= ref_high)
            cont05_level = ref_high + 0.5 * ref_range if side == "high_sweep" else ref_low - 0.5 * ref_range
            cont10_level = ref_high + 1.0 * ref_range if side == "high_sweep" else ref_low - 1.0 * ref_range
            cont05_hit = _first_hit_time(path, lambda r: float(r["high"]) >= cont05_level if side == "high_sweep" else float(r["low"]) <= cont05_level)
            cont10_hit = _first_hit_time(path, lambda r: float(r["high"]) >= cont10_level if side == "high_sweep" else float(r["low"]) <= cont10_level)
            reclaim_hit = _first_hit_time(path, lambda r: float(r["close"]) < ref_high if side == "high_sweep" else float(r["close"]) > ref_low)

            mfe_r, mae_r = _calc_excursions(path, side, sweep_close, ref_range)
            path_to_mid = path.loc[:midpoint_hit] if midpoint_hit is not None else path.iloc[0:0]
            path_to_opposite = path.loc[:opposite_hit] if opposite_hit is not None else path.iloc[0:0]
            mfe_mid_r, mae_mid_r = _calc_excursions(path_to_mid, side, sweep_close, ref_range)
            mfe_opp_r, mae_opp_r = _calc_excursions(path_to_opposite, side, sweep_close, ref_range)

            if reclaim_hit is None:
                max_overshoot_before_reclaim = None
            else:
                over_path = exec_df.iloc[pos : int(exec_df.index.get_indexer([reclaim_hit])[0]) + 1]
                if side == "high_sweep":
                    max_overshoot_before_reclaim = float((over_path["high"].max() - ref_high) / ref_range)
                else:
                    max_overshoot_before_reclaim = float((ref_low - over_path["low"].min()) / ref_range)

            if opposite_hit is not None:
                final_outcome = "hit_opposite_boundary"
            elif midpoint_hit is not None:
                final_outcome = "hit_midpoint_only"
            elif cont10_hit is not None:
                final_outcome = "continued_1.0R"
            elif cont05_hit is not None:
                final_outcome = "continued_0.5R"
            else:
                final_outcome = "no_major_level"

            ref_range_pct = (ref_range / abs(ref_mid)) if ref_mid != 0 else np.nan
            pre_close = row.get("_close_pre")
            pre_sma = row.get("_sma")
            pre_slope = row.get("_sma_slope")
            pre_ret = row.get("_ret_n")
            trend_regime = _compute_trend_label(pre_close, pre_sma, pre_slope, pre_ret)

            rows.append(
                {
                    "reference_timeframe": config.reference_timeframe,
                    "execution_timeframe": config.execution_timeframe,
                    "confirmation_mode": config.confirmation_mode,
                    "event_mode": config.event_mode,
                    "holding_window_mode": config.holding_window_mode,
                    "side": side,
                    "sweep_time": ts,
                    "sweep_close": sweep_close,
                    "swept_level": (ref_high if side == "high_sweep" else ref_low),
                    "ref_high": ref_high,
                    "ref_low": ref_low,
                    "ref_mid": ref_mid,
                    "ref_range": ref_range,
                    "ref_range_pct": ref_range_pct,
                    "range_bucket": _range_bucket(ref_range_pct),
                    "period_start": period_start,
                    "year": int(ts.year),
                    "overshoot_r": float(overshoot_r),
                    "hit_midpoint": midpoint_hit is not None,
                    "time_to_midpoint_hours": _time_to_hours(ts, midpoint_hit),
                    "hit_opposite_boundary": opposite_hit is not None,
                    "time_to_opposite_boundary_hours": _time_to_hours(ts, opposite_hit),
                    "continued_0_5r": cont05_hit is not None,
                    "time_to_continuation_0_5r_hours": _time_to_hours(ts, cont05_hit),
                    "continued_1_0r": cont10_hit is not None,
                    "time_to_continuation_1_0r_hours": _time_to_hours(ts, cont10_hit),
                    "reclaim": reclaim_hit is not None,
                    "time_to_reclaim_hours": _time_to_hours(ts, reclaim_hit),
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "mfe_before_midpoint_r": mfe_mid_r,
                    "mae_before_midpoint_r": mae_mid_r,
                    "mfe_before_opposite_r": mfe_opp_r,
                    "mae_before_opposite_r": mae_opp_r,
                    "max_overshoot_before_reclaim_r": max_overshoot_before_reclaim,
                    "final_outcome": final_outcome,
                    "window_end_time": path.index[-1],
                    "window_bars": int(len(path)),
                    "window_hours": float(len(path) * step.total_seconds() / 3600.0),
                    "pre_event_volatility": row.get("_vol"),
                    "trend_regime": trend_regime,
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = _build_pre_event_regimes(out)
    return out


def _safe_stats(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": None, "median": None, "p10": None, "p90": None, "p95": None}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
    }


def summarize_range_sweeps(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"num_sweeps": 0}
    return {
        "num_sweeps": int(len(events)),
        "p_reclaim": float(events["reclaim"].mean()),
        "p_hit_midpoint": float(events["hit_midpoint"].mean()),
        "p_hit_opposite_boundary": float(events["hit_opposite_boundary"].mean()),
        "p_continue_0_5r": float(events["continued_0_5r"].mean()),
        "p_continue_1_0r": float(events["continued_1_0r"].mean()),
        "time_to_midpoint_hours": _safe_stats(events["time_to_midpoint_hours"]),
        "time_to_opposite_boundary_hours": _safe_stats(events["time_to_opposite_boundary_hours"]),
        "time_to_reclaim_hours": _safe_stats(events["time_to_reclaim_hours"]),
        "mfe_r": _safe_stats(events["mfe_r"]),
        "mae_r": _safe_stats(events["mae_r"]),
        "overshoot_r": _safe_stats(events["overshoot_r"]),
        "median_mae_r": _safe_stats(events["mae_r"])["median"],
        "p90_mae_r": _safe_stats(events["mae_r"])["p90"],
        "start_sweep_time": str(pd.to_datetime(events["sweep_time"], utc=True).min()),
        "end_sweep_time": str(pd.to_datetime(events["sweep_time"], utc=True).max()),
    }


def _agg_table(events: pd.DataFrame, by: str) -> pd.DataFrame:
    grouped = (
        events.groupby(by)
        .agg(
            samples=("side", "count"),
            p_reclaim=("reclaim", "mean"),
            p_hit_midpoint=("hit_midpoint", "mean"),
            p_hit_opposite_boundary=("hit_opposite_boundary", "mean"),
            p_continue_0_5r=("continued_0_5r", "mean"),
            p_continue_1_0r=("continued_1_0r", "mean"),
            median_time_to_midpoint_h=("time_to_midpoint_hours", "median"),
            median_mfe_r=("mfe_r", "median"),
            median_mae_r=("mae_r", "median"),
            p90_mae_r=("mae_r", lambda s: float(pd.Series(s).dropna().quantile(0.90)) if len(pd.Series(s).dropna()) else np.nan),
            median_overshoot_r=("overshoot_r", "median"),
        )
        .reset_index()
    )
    if by == "range_bucket":
        order = pd.CategoricalDtype(["0-2.5%", "2.5-5.0%", "5.0-7.5%", "7.5%+", "unknown"], ordered=True)
        grouped["range_bucket"] = grouped["range_bucket"].astype(order)
        grouped = grouped.sort_values("range_bucket")
    return grouped.reset_index(drop=True)


def range_sweep_tables(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if events.empty:
        empty = pd.DataFrame()
        return {
            "by_side": empty,
            "by_year": empty,
            "by_vol_regime": empty,
            "by_trend_regime": empty,
            "by_range_bucket": empty,
            "by_reference_timeframe": empty,
            "by_execution_timeframe": empty,
            "by_confirmation_mode": empty,
        }
    return {
        "by_side": _agg_table(events, "side"),
        "by_year": _agg_table(events, "year"),
        "by_vol_regime": _agg_table(events, "vol_regime"),
        "by_trend_regime": _agg_table(events, "trend_regime"),
        "by_range_bucket": _agg_table(events, "range_bucket"),
        "by_reference_timeframe": _agg_table(events, "reference_timeframe"),
        "by_execution_timeframe": _agg_table(events, "execution_timeframe"),
        "by_confirmation_mode": _agg_table(events, "confirmation_mode"),
    }
