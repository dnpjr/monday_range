from __future__ import annotations

import streamlit as st
import pandas as pd
import traceback

from src.dashboard_data import (
    load_paper_state,
    load_paper_trades,
    find_latest_binance_cache,
    load_binance_cache,
    find_backtest_runs,
    load_backtest_run,
    build_state_snapshot,
    compute_data_health,
    reconstruct_completed_trades,
)
from src.dashboard_actions import (
    update_data_cache,
    get_cache_status,
    build_dataset_label,
    get_date_preset_range,
    validate_backtest_date_range,
    build_backtest_run_label,
    format_probability_pct,
    format_hours,
    format_float,
    STOP_MODE_LABEL_TO_VALUE,
    stop_mode_value,
    stop_mode_visible_inputs,
    run_research_lab_analysis,
    default_stop_mode_for_context,
    estimate_sweep_retest_signals,
    run_train_validation_split,
    run_cost_sensitivity,
    run_neighbourhood_test,
    run_range_sweep_lab_analysis,
)
from src.dashboard_actions import (
    run_research_backtest,
    discover_backtest_runs,
    load_backtest_outputs,
    run_research_diagnostics,
    load_diagnostics_outputs,
)
from src.timeframe_utils import can_resample
from run_backtest import load_cached_ohlc, filter_ohlc_window
from run_walk_forward import run_walk_forward
from src.monday_range_research import summarize_monday_range_research, probability_tables
st.set_page_config(page_title="Monday Range Dashboard", layout="wide")
st.title("Monday Range Dashboard")
st.caption("Research mode dashboard. Live trading is disabled.")

state = load_paper_state("data/paper_state.json")
events = load_paper_trades("data/paper_trades.csv")
cache_meta = find_latest_binance_cache("data/binance")
candles = load_binance_cache(cache_meta)

if cache_meta is not None and len(events) and "symbol" not in events.columns:
    events = events.copy()
    events["symbol"] = cache_meta.symbol

interval = cache_meta.interval if cache_meta else None
health = compute_data_health(candles, interval=interval)
snapshot = build_state_snapshot(state, events, candles, friday_cutoff_hour_utc=23)
backtest_runs = find_backtest_runs("data/backtests")
selected_backtest_meta = None
if backtest_runs:
    run_labels = [r.name for r in backtest_runs]
    default_idx = 0
    selected_name = st.sidebar.selectbox("Backtest Run", run_labels, index=default_idx)
    selected_backtest_meta = next((r for r in backtest_runs if r.name == selected_name), None)
backtest_info = load_backtest_run(selected_backtest_meta)

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Data Manager",
        "Backtest Runner",
        "Walk-Forward",
        "Backtest Diagnostics",
        "Research Lab",
        "Range Sweep Lab",
        "Strategy State",
        "Trade Journal",
        "Performance",
        "Data Health",
        "Config",
    ],
)


def _render_dataset_controls(prefix: str, *, default_preset: str = "Full cached range") -> dict:
    s1, s2 = st.columns(2)
    symbol_mode = s1.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "Custom"], key=f"{prefix}_symbol_mode")
    symbol = symbol_mode if symbol_mode != "Custom" else s1.text_input("Custom Symbol", value="BTCUSDT", key=f"{prefix}_custom_symbol").strip().upper()
    interval_sel = s2.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1, key=f"{prefix}_interval")

    try:
        cache_status = get_cache_status(symbol=symbol, interval=interval_sel, cache_dir="data/binance")
    except Exception as exc:
        cache_status = {"cache_exists": False}
        st.error(f"Failed to inspect cache: {exc}")

    preset_options = [
        "Full cached range",
        "Last 12 months",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "2022-2023",
        "2021-2024 training",
        "2025-2026 validation",
        "Custom",
    ]
    default_idx = preset_options.index(default_preset) if default_preset in preset_options else 0
    preset = st.selectbox("Preset date range", preset_options, index=default_idx, key=f"{prefix}_preset")
    if cache_status.get("cache_exists"):
        st.info(build_dataset_label(symbol, interval_sel, cache_status))

    start_key = f"{prefix}_start_date"
    end_key = f"{prefix}_end_date"
    if start_key not in st.session_state:
        st.session_state[start_key] = pd.Timestamp.now(tz="UTC").date()
    if end_key not in st.session_state:
        st.session_state[end_key] = pd.Timestamp.now(tz="UTC").date()

    if cache_status.get("cache_exists") and preset != "Custom":
        preset_range = get_date_preset_range(preset, cache_status)
        if preset_range.get("start"):
            st.session_state[start_key] = pd.Timestamp(preset_range["start"], tz="UTC").date()
        if preset_range.get("end"):
            st.session_state[end_key] = pd.Timestamp(preset_range["end"], tz="UTC").date()
        if preset_range.get("warning"):
            st.warning(str(preset_range["warning"]))

    d1, d2 = st.columns(2)
    start_date = d1.date_input("Start date (UTC)", key=start_key)
    end_date = d2.date_input("End date (UTC)", key=end_key)
    if cache_status.get("cache_exists"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cached candles", int(cache_status.get("rows", 0)))
        c2.metric("Duplicates", int(cache_status.get("duplicate_count", 0)))
        c3.metric("First candle", str(cache_status.get("first_candle", "N/A"))[:10])
        c4.metric("Last candle", str(cache_status.get("last_candle", "N/A"))[:10])
        st.caption(f"Cache file: `{cache_status.get('cache_file')}`")
    else:
        st.error(f"No cached data found for `{symbol} {interval_sel}`. Use Data Manager first.")

    return {
        "symbol": symbol,
        "interval": interval_sel,
        "start_date": start_date,
        "end_date": end_date,
        "cache_status": cache_status,
    }


def _render_exit_mechanics_controls(
    prefix: str,
    *,
    friday_default: int,
    default_stop_mode_value: str = "swept_boundary_offset",
    default_stop_range_fraction: float = 1.0,
) -> dict:
    e1, e2 = st.columns(2)
    tp1_range_fraction = float(
        e1.number_input(
            "TP1 range fraction",
            value=0.5,
            step=0.05,
            key=f"{prefix}_tp1_range_fraction",
            help="0.5 = midpoint, 1.0 = full opposite side of Monday range.",
        )
    )
    tp2_range_fraction = float(
        e2.number_input(
            "TP2 range fraction",
            value=1.0,
            step=0.05,
            key=f"{prefix}_tp2_range_fraction",
            help="0.5 = midpoint, 1.0 = full opposite side of Monday range.",
        )
    )
    e3, e4 = st.columns(2)
    tp1_close_fraction = float(e3.number_input("TP1 close fraction", value=0.5, step=0.05, key=f"{prefix}_tp1_close_fraction"))
    friday_cutoff_hour_utc = int(e4.number_input("Friday cutoff hour (UTC)", min_value=0, max_value=23, value=friday_default, step=1, key=f"{prefix}_friday_cutoff"))
    stop_mode_label_selected = st.selectbox(
        "Stop mode",
        list(STOP_MODE_LABEL_TO_VALUE.keys()),
        index=list(STOP_MODE_LABEL_TO_VALUE.values()).index(default_stop_mode_value)
        if default_stop_mode_value in STOP_MODE_LABEL_TO_VALUE.values()
        else 0,
        key=f"{prefix}_stop_mode",
        help="Swept boundary uses the swept Monday boundary; the offset mode places the stop a configured Monday-range width beyond it.",
    )
    stop_mode = stop_mode_value(stop_mode_label_selected)
    visible = stop_mode_visible_inputs(stop_mode)
    stop_range_fraction = None
    stop_pct = None
    if visible["show_stop_range_fraction"]:
        stop_range_fraction = float(
            st.number_input(
                "Stop range fraction",
                value=float(default_stop_range_fraction),
                step=0.1,
                key=f"{prefix}_stop_range_fraction",
                help="Stop distance as a fraction of Monday range.",
            )
        )
    elif visible["show_stop_pct"]:
        stop_pct = float(
            st.number_input(
                "Stop percent from entry",
                value=0.01,
                step=0.001,
                format="%.4f",
                key=f"{prefix}_stop_pct",
                help="Stop distance as a percent of entry price, e.g. 0.01 = 1%.",
            )
        )

    e5, e6 = st.columns(2)
    move_stop_to_breakeven_after_tp1 = bool(e5.checkbox("Move stop to breakeven after TP1", value=True, key=f"{prefix}_move_be"))
    breakeven_includes_fees = bool(e6.checkbox("Breakeven includes fees", value=False, key=f"{prefix}_be_fees"))

    e7, e8 = st.columns(2)
    exit_style_label = e7.selectbox(
        "Exit style",
        ["Partial TP1 then TP2", "Single target"],
        index=0,
        key=f"{prefix}_exit_style",
    )
    exit_style = "single_target" if exit_style_label == "Single target" else "partial_tp2"
    single_target_level = e8.selectbox(
        "Single target level",
        ["TP1/midpoint", "TP2/full reversal"],
        index=1,
        key=f"{prefix}_single_target_level",
        disabled=(exit_style != "single_target"),
    )
    single_target_level_value = "tp1" if single_target_level == "TP1/midpoint" else "tp2"
    single_target_mode = bool(exit_style == "single_target")
    return {
        "tp1_range_fraction": tp1_range_fraction,
        "tp2_range_fraction": tp2_range_fraction,
        "tp1_close_fraction": tp1_close_fraction,
        "friday_cutoff_hour_utc": friday_cutoff_hour_utc,
        "stop_mode": stop_mode,
        "stop_range_fraction": (float(default_stop_range_fraction) if stop_range_fraction is None else stop_range_fraction),
        "stop_pct": (0.01 if stop_pct is None else stop_pct),
        "move_stop_to_breakeven_after_tp1": move_stop_to_breakeven_after_tp1,
        "breakeven_includes_fees": breakeven_includes_fees,
        "exit_style": exit_style,
        "single_target_mode": single_target_mode,
        "single_target_level": single_target_level_value,
    }


def _format_research_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "side": "Direction",
        "iso_year": "Year",
        "range_bucket": "Range Bucket",
        "vol_regime": "Vol Regime",
        "rows": "Samples",
        "p_tp1_hit": "P(TP1 hit)",
        "p_tp2_hit": "P(TP2 hit)",
        "p_stop_hit": "P(STOP hit)",
        "p_tp1_before_stop": "P(TP1 before STOP)",
        "p_tp2_after_tp1": "P(TP2 after TP1)",
        "p_stop_before_tp1": "P(STOP before TP1)",
        "median_mae_before_tp1_r": "Median MAE before TP1 (R)",
        "p90_mae_before_tp1_r": "P90 MAE before TP1 (R)",
        "median_time_to_tp1_hours": "Median time to TP1 (h)",
        "median_time_to_tp2_hours": "Median time to TP2 (h)",
        "median_time_to_stop_hours": "Median time to STOP (h)",
    }
    out = out.rename(columns=rename_map)
    pct_cols = [c for c in out.columns if c.startswith("P(")]
    for c in pct_cols:
        out[c] = out[c].map(lambda v: format_probability_pct(v))
    for c in [c for c in out.columns if "(h)" in c]:
        out[c] = out[c].map(lambda v: format_hours(v))
    for c in [c for c in out.columns if "(R)" in c]:
        out[c] = out[c].map(lambda v: format_float(v, 2))
    if "Direction" in out.columns:
        out["Direction"] = out["Direction"].replace({"LONG": "Long", "SHORT": "Short"})
    return out


def _format_r(value: float | None, decimals: int = 2) -> str:
    base = format_float(value, decimals)
    return f"{base}R" if base != "N/A" else "N/A"

if page == "Overview":
    st.subheader("Overview")
    st.info("Mode: PAPER ONLY")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", cache_meta.symbol if cache_meta and cache_meta.symbol else "N/A")
    c2.metric("Interval", interval or "N/A")
    c3.metric("Current Position", snapshot["position_side"] or "FLAT")
    c4.metric("Latest Processed Candle", snapshot["last_processed_open_time"] or "N/A")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Cash", f"{snapshot['cash']:.2f}" if snapshot["cash"] is not None else "N/A")
    c6.metric("Equity", f"{snapshot['equity']:.2f}" if snapshot["equity"] is not None else "N/A")
    c7.metric("Realized PnL", f"{snapshot['realized_pnl']:.2f}" if snapshot["realized_pnl"] is not None else "N/A")
    c8.metric("Unrealized PnL", f"{snapshot['unrealized_pnl']:.2f}" if snapshot["unrealized_pnl"] is not None else "N/A")

    c9, c10, c11 = st.columns(3)
    c9.metric("Paper Event Count", snapshot["event_count"])
    c10.metric("Paper Trade Count", snapshot["trade_count"])
    c11.metric("Cached Candle Count", int(len(candles)))

    st.caption("Paper events come from `data/paper_trades.csv`.")
    st.caption("Candle history comes from `data/binance/{symbol}_{interval}.csv`.")
    st.caption("Backtest trades come from `data/backtests/<run>/trades.csv`.")

    if snapshot["last_event"] is not None:
        st.markdown("**Last Event**")
        st.json(snapshot["last_event"])
    else:
        st.info("No paper trade events found yet.")

    if backtest_info is not None:
        st.markdown("### Backtest Results")
        bsum = backtest_info["summary"]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Selected Run", backtest_info["name"])
        b2.metric("Backtest Trades", int(backtest_info["trades_count"]))
        b3.metric("Backtest Total Return %", f"{float(bsum.get('total_return_pct', 0.0)):.2f}")
        b4.metric("Backtest Win Rate", f"{float(bsum.get('win_rate', 0.0) * 100.0):.2f}%" if bsum.get("win_rate") is not None else "N/A")
        st.caption(f"Loaded from `{backtest_info['path']}`")

    age_sec = health.get("latest_candle_age_seconds")
    expected_sec = None
    if interval == "15m":
        expected_sec = 900
    elif interval == "1h":
        expected_sec = 3600
    elif interval == "4h":
        expected_sec = 14400
    elif interval == "1d":
        expected_sec = 86400
    if age_sec is not None and expected_sec is not None and age_sec > expected_sec * 1.5:
        st.warning(f"Stale data warning: latest candle age is {age_sec:.0f}s (expected around {expected_sec}s).")

if page == "Data Manager":
    st.subheader("Data Manager")
    st.warning("RESEARCH MODE ONLY — NO LIVE TRADING")
    st.caption("Public Binance market data only. No API keys and no order routing.")

    symbol_mode = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "Custom"])
    symbol = symbol_mode
    if symbol_mode == "Custom":
        symbol = st.text_input("Custom Symbol", value="BTCUSDT").strip().upper()

    interval_choice = st.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)
    window_mode = st.radio("Window Mode", ["lookback_days", "start_end"], horizontal=True)
    lookback_days = None
    start = None
    end = None
    if window_mode == "lookback_days":
        lookback_days = int(st.number_input("Lookback Days", min_value=1, max_value=36500, value=1825, step=1))
    else:
        start = st.text_input("Start (UTC)", value="", placeholder="YYYY-MM-DD")
        end = st.text_input("End (UTC)", value="", placeholder="YYYY-MM-DD")
        start = start.strip() or None
        end = end.strip() or None
    force_refresh = st.checkbox("Force Refresh", value=False)

    st.markdown("**Current Cache Status**")
    try:
        status = get_cache_status(symbol=symbol, interval=interval_choice, cache_dir="data/binance")
        if not status["cache_exists"]:
            st.info(f"No cache file yet for `{symbol} {interval_choice}` at `{status['cache_file']}`.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", int(status["rows"]))
            c2.metric("Duplicates", int(status["duplicate_count"]))
            c3.metric("Missing Candles (est.)", status["missing_candle_count"] if status["missing_candle_count"] is not None else "N/A")
            c4.metric("Latest Candle Age (s)", f"{status['latest_candle_age_seconds']:.0f}" if status["latest_candle_age_seconds"] is not None else "N/A")
            st.caption(f"Cache file: `{status['cache_file']}`")
            st.caption(f"First candle: `{status['first_candle']}`")
            st.caption(f"Last candle: `{status['last_candle']}`")
    except Exception as exc:
        st.error(f"Failed to read cache status: {exc}")

    if st.button("Download / Update Cache", type="primary"):
        try:
            result = update_data_cache(
                symbol=symbol,
                interval=interval_choice,
                lookback_days=lookback_days,
                start=start,
                end=end,
                force=force_refresh,
                cache_dir="data/binance",
            )
            st.success("Cache update complete.")
            st.json(
                {
                    "symbol": result["symbol"],
                    "interval": result["interval"],
                    "rows_saved": result["rows_saved"],
                    "first_candle": result["first_candle"],
                    "last_candle": result["last_candle"],
                    "duplicates_removed": result["duplicates_removed"],
                    "missing_candle_count": result["missing_candle_count"],
                    "cache_file": result["cache_file"],
                }
            )
        except Exception as exc:
            st.error(f"Cache update failed: {exc}")

if page == "Backtest Runner":
    st.subheader("Backtest Runner")
    st.caption("Run cached-data Monday range backtests. Research mode only. No live trading.")
    st.warning("RESEARCH MODE ONLY — NO LIVE TRADING")
    st.caption("Cached public market data only. No order routing.")

    with st.expander("Dataset", expanded=True):
        ds = _render_dataset_controls("bt")
    symbol = ds["symbol"]
    interval_bt = ds["interval"]
    start_d = ds["start_date"]
    end_d = ds["end_date"]
    bt_cache_status = ds["cache_status"]

    with st.expander("Core Assumptions", expanded=True):
        c1, c2 = st.columns(2)
        fee_bps = float(c1.number_input("Fee, bps", value=10.0, step=0.5, help="Round-trip trading fee assumption in basis points."))
        slippage_bps = float(c2.number_input("Slippage, bps", value=5.0, step=0.5, help="Entry/exit slippage assumption in basis points."))
        c3, c4, c5 = st.columns(3)
        risk_fraction = float(c3.number_input("Risk per trade", value=0.02, step=0.005, format="%.6f", help="Fraction of the selected portfolio equity base risked at the initial stop."))
        direction = c4.selectbox("Direction", ["both", "long_only", "short_only"], index=0)
        strategy_label = c5.selectbox(
            "Strategy",
            ["Current Monday range strategy", "Original sweep/retest strategy"],
            index=0,
            help="Original sweep/retest strategy uses only actual legacy sweep/retest signals.",
        )
        strategy = "sweep_retest" if strategy_label == "Original sweep/retest strategy" else "current_monday_range"
        c6, c7, c8 = st.columns(3)
        risk_base = c6.selectbox("Risk base", ["current_equity", "initial_capital"], index=0, help="Current equity compounds risk; initial capital preserves fixed-base fractional sizing.")
        max_leverage = float(c7.number_input("Maximum gross leverage", min_value=0.1, value=1.0, step=0.25, help="Caps entry notional as a multiple of current equity."))
        intrabar_policy = c8.selectbox("Intrabar policy", ["conservative_stop_first", "target_first"], index=0, help="OHLC cannot resolve stop/target order; conservative stop-first is the research default.")

    if strategy == "sweep_retest":
        st.info("Original sweep/retest strategy: trades only actual legacy sweep/retest signals (first signal per week).")

    with st.expander("Exit Mechanics", expanded=True):
        default_stop_mode = "swept_boundary_offset"
        default_stop_frac = 0.5 if strategy == "sweep_retest" else 1.0
        exit_cfg = _render_exit_mechanics_controls(
            "bt",
            friday_default=20,
            default_stop_mode_value=default_stop_mode,
            default_stop_range_fraction=default_stop_frac,
        )
    tp1_range_fraction = float(exit_cfg["tp1_range_fraction"])
    tp2_range_fraction = float(exit_cfg["tp2_range_fraction"])
    tp1_close_fraction = float(exit_cfg["tp1_close_fraction"])
    friday_cutoff_hour_utc = int(exit_cfg["friday_cutoff_hour_utc"])
    stop_mode = str(exit_cfg["stop_mode"])
    stop_range_fraction = float(exit_cfg["stop_range_fraction"])
    stop_pct = float(exit_cfg["stop_pct"])
    move_stop_to_breakeven_after_tp1 = bool(exit_cfg["move_stop_to_breakeven_after_tp1"])
    breakeven_includes_fees = bool(exit_cfg["breakeven_includes_fees"])
    exit_style = str(exit_cfg["exit_style"])
    single_target_mode = bool(exit_cfg["single_target_mode"])
    single_target_level = str(exit_cfg["single_target_level"])

    with st.expander("Optional Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        min_range_raw = f1.text_input("Min range %", value="", placeholder="e.g. 0.01")
        max_range_raw = f2.text_input("Max range %", value="", placeholder="e.g. 0.06")
        sma_raw = f3.text_input("SMA period", value="", placeholder="e.g. 50")
        f4, f5 = st.columns(2)
        max_entry_day_raw = f4.text_input("Max entry day UTC (0-6)", value="", placeholder="e.g. 2")
        max_entry_hour_raw = f5.text_input("Max entry hour UTC (0-23)", value="", placeholder="e.g. 23")

    date_validation = validate_backtest_date_range(
        str(start_d),
        str(end_d),
        bt_cache_status if isinstance(bt_cache_status, dict) else {},
    )
    if date_validation.get("warning"):
        st.warning(str(date_validation["warning"]))
    if fee_bps == 0.0 or slippage_bps == 0.0:
        st.warning("Cost warning: fee_bps or slippage_bps is zero. Results may overestimate deployable performance.")
    if strategy == "sweep_retest":
        try:
            if bt_cache_status.get("cache_exists") and date_validation.get("valid"):
                ds_ohlc = load_cached_ohlc(symbol, interval_bt)
                ds_ohlc = filter_ohlc_window(ds_ohlc, str(start_d), str(end_d))
                legacy_signals = estimate_sweep_retest_signals(ds_ohlc)
                st.metric("Legacy sweep/retest signals in selected range", legacy_signals)
        except Exception as exc:
            st.warning(f"Could not estimate legacy signal count: {exc}")

    def _parse_opt_float_local(v: str) -> float | None:
        t = v.strip()
        return float(t) if t else None

    def _parse_opt_int_local(v: str) -> int | None:
        t = v.strip()
        return int(t) if t else None

    try:
        min_range_pct_preview = _parse_opt_float_local(min_range_raw)
        max_range_pct_preview = _parse_opt_float_local(max_range_raw)
        sma_period_preview = _parse_opt_int_local(sma_raw)
        max_entry_day_preview = _parse_opt_int_local(max_entry_day_raw)
        max_entry_hour_preview = _parse_opt_int_local(max_entry_hour_raw)
        parse_error = None
    except ValueError as exc:
        min_range_pct_preview = None
        max_range_pct_preview = None
        sma_period_preview = None
        max_entry_day_preview = None
        max_entry_hour_preview = None
        parse_error = str(exc)

    if parse_error:
        st.warning(f"Optional filter parse warning: {parse_error}")
    bt_config = {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval_bt,
        "start": str(start_d),
        "end": str(end_d),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "risk_fraction": risk_fraction,
        "risk_base": risk_base,
        "max_leverage": max_leverage,
        "intrabar_policy": intrabar_policy,
        "tp1_range_fraction": tp1_range_fraction,
        "tp2_range_fraction": tp2_range_fraction,
        "tp2_to_full": 1.0,
        "friday_cutoff_hour_utc": friday_cutoff_hour_utc,
        "min_range_pct": min_range_pct_preview,
        "max_range_pct": max_range_pct_preview,
        "direction": direction,
        "max_entry_day_utc": max_entry_day_preview,
        "max_entry_hour_utc": max_entry_hour_preview,
        "sma_period": sma_period_preview,
        "tp1_to_mid": 1.0,
        "tp1_close_fraction": tp1_close_fraction,
        "stop_mode": stop_mode,
        "stop_range_fraction": stop_range_fraction,
        "stop_pct": stop_pct,
        "move_stop_to_breakeven_after_tp1": move_stop_to_breakeven_after_tp1,
        "breakeven_includes_fees": breakeven_includes_fees,
        "exit_style": exit_style,
        "single_target_mode": single_target_mode,
        "single_target_level": single_target_level,
        "output_dir": "data/backtests",
    }

    st.markdown("### Run")
    run_disabled = (not bt_cache_status.get("cache_exists", False)) or (not bool(date_validation.get("valid")))
    run_clicked = st.button("Run Backtest", type="primary", disabled=run_disabled)
    if not date_validation.get("valid"):
        st.error(str(date_validation.get("error")))

    if run_clicked:
        try:
            min_range_pct = float(min_range_raw) if min_range_raw.strip() else None
            max_range_pct = float(max_range_raw) if max_range_raw.strip() else None
            sma_period = int(sma_raw) if sma_raw.strip() else None
            max_entry_day_utc = int(max_entry_day_raw) if max_entry_day_raw.strip() else None
            max_entry_hour_utc = int(max_entry_hour_raw) if max_entry_hour_raw.strip() else None

            result = run_research_backtest(
                strategy=strategy,
                symbol=symbol,
                interval=interval_bt,
                start=str(start_d),
                end=str(end_d),
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                risk_fraction=risk_fraction,
                risk_base=risk_base,
                max_leverage=max_leverage,
                intrabar_policy=intrabar_policy,
                tp1_range_fraction=tp1_range_fraction,
                tp2_range_fraction=tp2_range_fraction,
                friday_cutoff_hour_utc=friday_cutoff_hour_utc,
                min_range_pct=min_range_pct,
                max_range_pct=max_range_pct,
                direction=direction,
                max_entry_day_utc=max_entry_day_utc,
                max_entry_hour_utc=max_entry_hour_utc,
                sma_period=sma_period,
                tp1_close_fraction=tp1_close_fraction,
                stop_mode=stop_mode,
                stop_range_fraction=stop_range_fraction,
                stop_pct=stop_pct,
                move_stop_to_breakeven_after_tp1=move_stop_to_breakeven_after_tp1,
                breakeven_includes_fees=breakeven_includes_fees,
                exit_style=exit_style,
                single_target_mode=single_target_mode,
                single_target_level=single_target_level,
                output_dir="data/backtests",
            )
            st.session_state["bt_last_result"] = result
            st.success("Backtest completed.")
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")

    st.markdown("### Focused Validation (Sweep/Retest)")
    st.caption("Runs additional robustness checks for the currently selected configuration.")
    if strategy != "sweep_retest":
        st.info("Switch strategy to `Original sweep/retest strategy` to run this workflow.")
    fv1, fv2, fv3 = st.columns(3)
    tv_clicked = fv1.button("Run Train/Validation Split", disabled=(run_disabled or strategy != "sweep_retest"))
    cs_clicked = fv2.button("Run Cost Sensitivity", disabled=(run_disabled or strategy != "sweep_retest"))
    nb_clicked = fv3.button("Run Neighbourhood Test", disabled=(run_disabled or strategy != "sweep_retest"))

    if tv_clicked:
        try:
            latest_cached = str(bt_cache_status.get("last_candle", ""))[:10]
            split = run_train_validation_split(config=bt_config, latest_cached_date=latest_cached)
            st.session_state["bt_train_validation"] = split
            st.success("Train/validation split completed.")
        except Exception as exc:
            st.error(f"Train/validation split failed: {exc}")

    if cs_clicked:
        try:
            cs_df = run_cost_sensitivity(config=bt_config, fee_bps_values=[0.0, 7.5, 10.0], slippage_bps_values=[0.0, 1.0, 2.5, 5.0])
            st.session_state["bt_cost_sensitivity"] = cs_df
            st.success("Cost sensitivity completed.")
        except Exception as exc:
            st.error(f"Cost sensitivity failed: {exc}")

    if nb_clicked:
        try:
            nb_df = run_neighbourhood_test(config=bt_config)
            st.session_state["bt_neighbourhood"] = nb_df
            st.success("Neighbourhood test completed.")
        except Exception as exc:
            st.error(f"Neighbourhood test failed: {exc}")

    tv_payload = st.session_state.get("bt_train_validation")
    if isinstance(tv_payload, dict):
        st.markdown("**Train vs Validation**")
        tv_table = tv_payload.get("table")
        if isinstance(tv_table, pd.DataFrame) and len(tv_table):
            st.dataframe(tv_table, use_container_width=True)

    cs_payload = st.session_state.get("bt_cost_sensitivity")
    if isinstance(cs_payload, pd.DataFrame) and len(cs_payload):
        st.markdown("**Cost Sensitivity Results**")
        st.dataframe(cs_payload, use_container_width=True)

    nb_payload = st.session_state.get("bt_neighbourhood")
    if isinstance(nb_payload, pd.DataFrame) and len(nb_payload):
        st.markdown("**Neighbourhood Results**")
        st.dataframe(nb_payload, use_container_width=True)

    st.markdown("### Results")
    result = st.session_state.get("bt_last_result")
    if isinstance(result, dict):
        st.code(result["out_dir"])
        s = result["summary"]
        st.caption(
            f"Strategy: `{s.get('strategy', 'N/A')}` | Signals: `{s.get('signal_count', 'N/A')}` | "
            f"Exit style: `{s.get('exit_style', 'N/A')}`"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final equity", f"{float(s.get('final_equity', 0.0)):.2f}")
        m2.metric("Total return %", f"{float(s.get('total_return_pct', 0.0)):.2f}")
        m3.metric("Trades", int(s.get("num_trades", 0)))
        m4.metric("Win rate", f"{float(s.get('win_rate', 0.0) * 100.0):.2f}%" if s.get("win_rate") is not None else "N/A")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg trade PnL", f"{float(s.get('average_trade_pnl', 0.0)):.4f}" if s.get("average_trade_pnl") is not None else "N/A")
        m6.metric("Best trade", f"{float(s.get('best_trade', 0.0)):.4f}" if s.get("best_trade") is not None else "N/A")
        m7.metric("Worst trade", f"{float(s.get('worst_trade', 0.0)):.4f}" if s.get("worst_trade") is not None else "N/A")
        m8.metric("Total fees paid", f"{float(s.get('total_fees_paid', 0.0)):.4f}" if s.get("total_fees_paid") is not None else "N/A")
        m9, m10 = st.columns(2)
        m9.metric("Max drawdown", f"{float(s.get('max_drawdown', 0.0)):.4f}" if s.get("max_drawdown") is not None else "N/A")
        m10.metric("Start/end used", f"{s.get('start_used', 'N/A')} -> {s.get('end_used', 'N/A')}")

        loaded = load_backtest_outputs(result["out_dir"])
        trades_df = loaded.get("trades")
        if isinstance(trades_df, pd.DataFrame) and len(trades_df):
            st.dataframe(trades_df, use_container_width=True)

        with st.expander("Show Raw Summary / Config JSON", expanded=False):
            st.json(loaded.get("summary") or {})
            st.json(loaded.get("config") or {})

        out_dir = result["out_dir"]
        trades_path = f"{out_dir}/trades.csv"
        summary_path = f"{out_dir}/summary.json"
        config_path = f"{out_dir}/config.json"
        db1, db2, db3 = st.columns(3)
        if loaded.get("trades_exists"):
            with open(trades_path, "rb") as f:
                db1.download_button("Download trades.csv", data=f.read(), file_name="trades.csv", mime="text/csv")
        if loaded.get("summary_exists"):
            with open(summary_path, "rb") as f:
                db2.download_button("Download summary.json", data=f.read(), file_name="summary.json", mime="application/json")
        if loaded.get("config_exists"):
            with open(config_path, "rb") as f:
                db3.download_button("Download config.json", data=f.read(), file_name="config.json", mime="application/json")
    else:
        st.info("Run a backtest to see results here.")

    st.markdown("### Saved Backtest Runs")
    try:
        recent_runs = discover_backtest_runs("data/backtests")
        if not recent_runs:
            st.info("No backtest runs found under `data/backtests`.")
        else:
            chosen = st.selectbox(
                "Select saved run",
                recent_runs,
                key="bt_recent_run",
                format_func=build_backtest_run_label,
            )
            st.caption(f"Path: `{chosen}`")
            loaded_recent = load_backtest_outputs(chosen)
            rs = loaded_recent.get("summary")
            if isinstance(rs, dict):
                with st.expander("Show selected run summary JSON", expanded=False):
                    st.json(rs)
            rt = loaded_recent.get("trades")
            if isinstance(rt, pd.DataFrame) and len(rt):
                st.dataframe(rt, use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to load saved backtest runs: {exc}")

if page == "Walk-Forward":
    st.subheader("Walk-Forward")
    st.warning("RESEARCH MODE ONLY — NO LIVE TRADING")
    st.caption("Runs sweep_retest walk-forward optimisation on cached data only.")

    with st.expander("Dataset", expanded=True):
        wf_ds = _render_dataset_controls("wf")
    wf_symbol = wf_ds["symbol"]
    wf_interval = wf_ds["interval"]
    wf_cache_status = wf_ds["cache_status"]

    c1, c2, c3 = st.columns(3)
    wf_fee_bps = float(c1.number_input("Fee, bps", value=10.0, step=0.5, key="wf_fee_bps"))
    wf_slippage_bps = float(c2.number_input("Slippage, bps", value=5.0, step=0.5, key="wf_slippage_bps"))
    wf_top_n = int(c3.number_input("Top N configs per fold", value=3, min_value=1, max_value=20, step=1, key="wf_top_n"))

    if wf_fee_bps == 0.0 or wf_slippage_bps == 0.0:
        st.warning("Cost warning: fee_bps or slippage_bps is zero. Results may overestimate deployable performance.")

    run_wf_disabled = not bool(wf_cache_status.get("cache_exists", False))
    if st.button("Run Default Sweep/Retest Walk-Forward", type="primary", disabled=run_wf_disabled, key="wf_run"):
        try:
            wf_result = run_walk_forward(
                strategy="sweep_retest",
                symbol=wf_symbol,
                interval=wf_interval,
                yearly_mode=True,
                fee_bps=wf_fee_bps,
                slippage_bps=wf_slippage_bps,
                top_n=wf_top_n,
                output_dir="data/walk_forward",
            )
            st.session_state["wf_last_result"] = wf_result
            st.success("Walk-forward completed.")
        except Exception as exc:
            st.error(f"Walk-forward failed: {exc}")

    wf_payload = st.session_state.get("wf_last_result")
    if isinstance(wf_payload, dict):
        st.code(wf_payload.get("out_dir", ""))
        wf_summary = wf_payload.get("walk_forward_results")
        fold_test = wf_payload.get("fold_test_results")
        fold_train = wf_payload.get("fold_train_results")
        fold_summary = wf_payload.get("fold_summary")

        if isinstance(wf_summary, pd.DataFrame) and len(wf_summary):
            row = wf_summary.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Folds", int(row.get("num_folds", 0)))
            m2.metric("Positive OOS folds", f"{float(row.get('percentage_positive_out_of_sample', 0.0)):.1f}%")
            m3.metric("Avg validation return", f"{float(row.get('average_out_of_sample_return_pct', 0.0)):.2f}%")
            m4.metric("Worst validation return", f"{float(row.get('worst_out_of_sample_return_pct', 0.0)):.2f}%")
            m5.metric("Worst validation drawdown", f"{float(row.get('worst_out_of_sample_drawdown', 0.0)):.4f}" if row.get("worst_out_of_sample_drawdown") is not None else "N/A")

        st.markdown("### Fold Summary")
        if isinstance(fold_summary, pd.DataFrame) and len(fold_summary):
            fs = fold_summary.copy()
            if "test_positive" in fs.columns:
                fs["validation_fold"] = fs["test_positive"].map(lambda v: "POSITIVE" if bool(v) else "NEGATIVE")
            st.dataframe(fs, use_container_width=True)
        else:
            st.info("No fold summary available.")

        st.markdown("### Selected Parameters per Fold")
        if isinstance(fold_test, pd.DataFrame) and len(fold_test):
            primary = fold_test[fold_test["is_primary"] == True].copy() if "is_primary" in fold_test.columns else fold_test.copy()
            cols = [
                "fold_id",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "tp1_range_fraction",
                "stop_range_fraction",
                "tp1_close_fraction",
                "friday_cutoff_hour_utc",
                "exit_style",
                "test_total_return_pct",
                "test_max_drawdown",
                "test_num_trades",
            ]
            show_cols = [c for c in cols if c in primary.columns]
            st.dataframe(primary[show_cols], use_container_width=True)
        else:
            st.info("No fold test rows available.")

        with st.expander("All Fold Train Results", expanded=False):
            if isinstance(fold_train, pd.DataFrame) and len(fold_train):
                st.dataframe(fold_train, use_container_width=True)
        with st.expander("All Fold Test Results", expanded=False):
            if isinstance(fold_test, pd.DataFrame) and len(fold_test):
                st.dataframe(fold_test, use_container_width=True)

if page == "Backtest Diagnostics":
    st.subheader("Backtest Diagnostics")
    st.warning("RESEARCH MODE ONLY — NO LIVE TRADING")
    st.caption("Runs diagnostics on existing cached-data backtest outputs. No order routing.")

    runs = discover_backtest_runs("data/backtests")
    if not runs:
        st.info("No backtest runs found under `data/backtests/`. Run a backtest first.")
    else:
        selected_run = st.selectbox("Select Backtest Run", runs, key="diag_select_run")
        loaded_run = load_backtest_outputs(selected_run)
        summary = loaded_run.get("summary") if isinstance(loaded_run, dict) else None
        config = loaded_run.get("config") if isinstance(loaded_run, dict) else None
        summary = summary if isinstance(summary, dict) else {}
        config = config if isinstance(config, dict) else {}

        st.markdown("**Selected Run Metadata**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Symbol", str(summary.get("symbol") or config.get("symbol") or "N/A"))
        m2.metric("Interval", str(summary.get("interval") or config.get("interval") or "N/A"))
        m3.metric("Start", str(summary.get("start_used") or config.get("start") or "N/A"))
        m4.metric("End", str(summary.get("end_used") or config.get("end") or "N/A"))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Fee, bps", str(config.get("fee_bps", "N/A")))
        m6.metric("Slippage, bps", str(config.get("slippage_bps", "N/A")))
        m7.metric("Risk per trade", str(config.get("risk_fraction", "N/A")))
        m8.metric("TP2 range fraction", str(config.get("tp2_range_fraction", config.get("tp2_to_full", "N/A"))))

        m9, m10 = st.columns(2)
        m9.metric("friday_cutoff_hour_utc", str(config.get("friday_cutoff_hour_utc", "N/A")))
        m10.metric("Output Path", selected_run)

        st.caption(
            "Filters used: "
            f"min_range_pct={config.get('min_range_pct')}, "
            f"max_range_pct={config.get('max_range_pct')}, "
            f"direction={config.get('direction')}, "
            f"max_entry_day_utc={config.get('max_entry_day_utc')}, "
            f"max_entry_hour_utc={config.get('max_entry_hour_utc')}, "
            f"sma_period={config.get('sma_period')}"
        )

        if st.button("Run / Refresh Diagnostics", type="primary", key="diag_run"):
            try:
                run_research_diagnostics(selected_run, cache_dir="data/binance")
                st.success("Diagnostics refreshed.")
            except Exception as exc:
                st.error(f"Diagnostics run failed: {exc}")

        diag_loaded = load_diagnostics_outputs(selected_run)
        if not diag_loaded.get("diagnostics_exists"):
            st.info("No diagnostics files found yet. Click 'Run / Refresh Diagnostics' to generate them.")
        else:
            diagnostics = diag_loaded.get("diagnostics")
            if isinstance(diagnostics, dict):
                st.markdown("### Diagnostics Summary")
                st.json(diagnostics)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Gross PnL", f"{float(diagnostics.get('gross_pnl_total', 0.0)):.4f}" if diagnostics.get("gross_pnl_total") is not None else "N/A")
                c2.metric("Net PnL", f"{float(diagnostics.get('net_pnl_total', 0.0)):.4f}")
                c3.metric("Fee Drag %", f"{float(diagnostics.get('fee_drag_pct_of_initial_cash', 0.0)):.4f}" if diagnostics.get("fee_drag_pct_of_initial_cash") is not None else "N/A")
                c4.metric("Avg Holding Hours", f"{float(diagnostics.get('average_holding_hours', 0.0)):.2f}" if diagnostics.get("average_holding_hours") is not None else "N/A")

                bm = diagnostics.get("buy_hold_benchmark")
                if isinstance(bm, dict):
                    st.markdown("### Buy-and-Hold Benchmark")
                    st.json(bm)

                st.markdown("### PnL by Entry Weekday / Hour")
                st.write("Weekday:", diagnostics.get("pnl_by_entry_day", {}))
                st.write("Hour (UTC):", diagnostics.get("pnl_by_entry_hour_utc", {}))

            by_year = diag_loaded.get("by_year")
            if isinstance(by_year, pd.DataFrame) and len(by_year):
                st.markdown("### PnL by Year")
                st.dataframe(by_year, use_container_width=True)
                if "exit_year" in by_year.columns and "net_pnl" in by_year.columns:
                    chart = by_year[["exit_year", "net_pnl"]].copy().set_index("exit_year")
                    st.bar_chart(chart)
            else:
                st.info("No yearly diagnostics table available.")

            by_direction = diag_loaded.get("by_direction")
            if isinstance(by_direction, pd.DataFrame) and len(by_direction):
                st.markdown("### PnL by Direction")
                st.dataframe(by_direction, use_container_width=True)
                if "side" in by_direction.columns and "net_pnl" in by_direction.columns:
                    chart = by_direction[["side", "net_pnl"]].copy().set_index("side")
                    st.bar_chart(chart)
            else:
                st.info("No direction diagnostics table available.")

            by_exit = diag_loaded.get("by_exit_reason")
            if isinstance(by_exit, pd.DataFrame) and len(by_exit):
                st.markdown("### PnL by Exit Reason")
                st.dataframe(by_exit, use_container_width=True)
                if "reason" in by_exit.columns and "net_pnl" in by_exit.columns:
                    chart = by_exit[["reason", "net_pnl"]].copy().set_index("reason")
                    st.bar_chart(chart)
            else:
                st.info("No exit-reason diagnostics table available.")

            by_bucket = diag_loaded.get("by_range_bucket")
            if isinstance(by_bucket, pd.DataFrame) and len(by_bucket):
                st.markdown("### PnL by Monday Range Bucket")
                st.dataframe(by_bucket, use_container_width=True)
            else:
                st.info("No range-bucket diagnostics table available.")

            st.markdown("### Download Diagnostics")
            d1, d2, d3, d4, d5 = st.columns(5)
            diagnostics_path = f"{selected_run}/diagnostics.json"
            by_year_path = f"{selected_run}/diagnostics_by_year.csv"
            by_direction_path = f"{selected_run}/diagnostics_by_direction.csv"
            by_exit_path = f"{selected_run}/diagnostics_by_exit_reason.csv"
            by_bucket_path = f"{selected_run}/diagnostics_by_range_bucket.csv"

            if diag_loaded.get("diagnostics_exists"):
                with open(diagnostics_path, "rb") as f:
                    d1.download_button("diagnostics.json", data=f.read(), file_name="diagnostics.json", mime="application/json")
            if diag_loaded.get("by_year_exists"):
                with open(by_year_path, "rb") as f:
                    d2.download_button("diagnostics_by_year.csv", data=f.read(), file_name="diagnostics_by_year.csv", mime="text/csv")
            if diag_loaded.get("by_direction_exists"):
                with open(by_direction_path, "rb") as f:
                    d3.download_button("diagnostics_by_direction.csv", data=f.read(), file_name="diagnostics_by_direction.csv", mime="text/csv")
            if diag_loaded.get("by_exit_reason_exists"):
                with open(by_exit_path, "rb") as f:
                    d4.download_button("diagnostics_by_exit_reason.csv", data=f.read(), file_name="diagnostics_by_exit_reason.csv", mime="text/csv")
            if diag_loaded.get("by_range_bucket_exists"):
                with open(by_bucket_path, "rb") as f:
                    d5.download_button("diagnostics_by_range_bucket.csv", data=f.read(), file_name="diagnostics_by_range_bucket.csv", mime="text/csv")

if page == "Research Lab":
    st.subheader("Research Lab")
    st.warning("RESEARCH MODE ONLY — NO LIVE TRADING")
    st.caption("Research Lab estimates how price paths behave after Monday range events. It is used to choose TP/STOP settings before running full backtests.")
    mode_label = st.selectbox(
        "Research calculation mode",
        ["Hypothetical weekly path analysis", "Original sweep/retest mode"],
        index=0,
        help="Original mode reproduces legacy sweep/retest statistics. New mode runs expanded path-analysis metrics.",
    )
    lab_analysis_mode = "new_path_analysis" if mode_label == "Hypothetical weekly path analysis" else "original_sweep_retest"

    with st.expander("Dataset", expanded=True):
        lab_ds = _render_dataset_controls("lab")
    lab_symbol = lab_ds["symbol"]
    lab_interval = lab_ds["interval"]
    lab_start = lab_ds["start_date"]
    lab_end = lab_ds["end_date"]
    lab_cache_status = lab_ds["cache_status"]
    lab_event_mode = "sweep_events"
    lab_exit = {
        "tp1_range_fraction": 0.5,
        "tp2_range_fraction": 1.0,
        "stop_mode": "swept_boundary_offset",
        "stop_range_fraction": 1.0,
        "stop_pct": 0.01,
        "friday_cutoff_hour_utc": 23,
    }
    if lab_analysis_mode == "new_path_analysis":
        st.info("Hypothetical weekly path analysis creates long and short candidates each week. It is not a tradable signal by itself.")
        with st.expander("Exit Mechanics", expanded=True):
            lab_exit = _render_exit_mechanics_controls(
                "lab",
                friday_default=23,
                default_stop_mode_value=default_stop_mode_for_context("research_lab"),
            )
        with st.expander("Advanced Event Settings", expanded=False):
            lab_event_mode = st.selectbox(
                "Event construction mode",
                ["sweep_events", "strategy_entries"],
                index=0,
                help="sweep_events: build long+short candidates each week. strategy_entries: only strategy-style sweep signal entries.",
            )
    else:
        st.info(
            "Original sweep/retest mode uses legacy fixed definitions (first weekly signal, "
            "mid/full targets, and legacy MAE calculations). Exit-mechanics controls are not used."
        )

    run_lab = st.button("Run Research Lab", type="primary", key="lab_run")
    if run_lab:
        try:
            lab_ohlc = load_cached_ohlc(lab_symbol, lab_interval)
            lab_ohlc = filter_ohlc_window(lab_ohlc, str(lab_start), str(lab_end))
            if lab_ohlc.empty:
                st.error("No cached candles in selected window. Use Data Manager to update cache or broaden dates.")
            else:
                lab_out = run_research_lab_analysis(
                    ohlc=lab_ohlc,
                    analysis_mode=lab_analysis_mode,
                    tp1_range_fraction=float(lab_exit["tp1_range_fraction"]),
                    tp2_range_fraction=float(lab_exit["tp2_range_fraction"]),
                    stop_mode=str(lab_exit["stop_mode"]),
                    stop_range_fraction=float(lab_exit["stop_range_fraction"]),
                    stop_pct=float(lab_exit["stop_pct"]),
                    friday_cutoff_hour_utc=int(lab_exit["friday_cutoff_hour_utc"]),
                    event_mode=lab_event_mode,
                )
                res = lab_out["results"]
                if res.empty:
                    st.info("No analyzable weeks in selected window.")
                else:
                    st.session_state["lab_last_results"] = lab_out
        except Exception as exc:
            st.error(f"Research Lab failed: {exc}")
            with st.expander("Debug details", expanded=False):
                st.code(traceback.format_exc())

    lab_payload = st.session_state.get("lab_last_results")
    if isinstance(lab_payload, dict):
        payload_mode = str(lab_payload.get("analysis_mode", "new_path_analysis"))
        res = lab_payload.get("results")
        if payload_mode == "new_path_analysis" and isinstance(res, pd.DataFrame):
            # Recompute presentation summaries from raw results to avoid stale metrics
            # after code changes or session reloads.
            summ = summarize_monday_range_research(res)
            tables = probability_tables(res)
        else:
            summ = lab_payload.get("summary", {})
            tables = lab_payload.get("tables", {})

        if payload_mode == "original_sweep_retest":
            st.markdown("### Original Sweep/Retest Research Metrics")
            st.caption("These metrics follow legacy research definitions from the original sweep/retest script.")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("total_signals", int(summ.get("total_signals", 0)))
            c2.metric("p_hit_mid", format_probability_pct(summ.get("p_hit_mid")))
            c3.metric("p_hit_full", format_probability_pct(summ.get("p_hit_full")))
            c4.metric("avg_mae_mid", _format_r(summ.get("avg_mae_mid"), 3))
            c5.metric("p90_mae_mid", _format_r(summ.get("p90_mae_mid"), 3))
            c6, c7, c8, c9, c10 = st.columns(5)
            c6.metric("avg_mae_full", _format_r(summ.get("avg_mae_full"), 3))
            c7.metric("p90_mae_full", _format_r(summ.get("p90_mae_full"), 3))
            c8.metric("avg_reward_ratio", _format_r(summ.get("avg_reward_ratio"), 3))
            c9.metric("p_sweep_retest", format_probability_pct(summ.get("p_sweep_retest")))
            c10.metric("p_sweep_retest_winners", format_probability_pct(summ.get("p_sweep_retest_winners")))
            if isinstance(res, pd.DataFrame) and len(res):
                st.markdown("### All Sweep/Retest Signals")
                f1, f2, f3, f4 = st.columns(4)
                side_filter = f1.selectbox("Side filter", ["All", "LONG", "SHORT"], index=0, key="lab_orig_side")
                year_values = sorted(res["iso_year"].dropna().astype(int).unique().tolist()) if "iso_year" in res.columns else []
                year_options = ["All"] + [str(y) for y in year_values]
                year_filter = f2.selectbox("Year filter", year_options, index=0, key="lab_orig_year")
                hit_mid_filter = f3.selectbox("Hit midpoint", ["All", "True", "False"], index=0, key="lab_orig_hit_mid")
                hit_full_filter = f4.selectbox("Hit full reversal", ["All", "True", "False"], index=0, key="lab_orig_hit_full")

                filt = res.copy()
                if side_filter != "All" and "side" in filt.columns:
                    filt = filt[filt["side"] == side_filter]
                if year_filter != "All" and "iso_year" in filt.columns:
                    filt = filt[filt["iso_year"].astype(int) == int(year_filter)]
                if hit_mid_filter != "All" and "hit_mid" in filt.columns:
                    filt = filt[filt["hit_mid"] == (hit_mid_filter == "True")]
                if hit_full_filter != "All" and "hit_full" in filt.columns:
                    filt = filt[filt["hit_full"] == (hit_full_filter == "True")]

                cols = [
                    "signal_time",
                    "side",
                    "entry_time",
                    "entry_price",
                    "mon_high",
                    "mon_low",
                    "mon_mid",
                    "hit_mid",
                    "hit_full",
                    "mae_mid",
                    "mae_full",
                    "reward_ratio",
                    "iso_year",
                    "iso_week",
                ]
                show_cols = [c for c in cols if c in filt.columns]
                st.dataframe(filt[show_cols], use_container_width=True)
            with st.expander("Raw original summary", expanded=False):
                st.json(summ)
        else:
            st.markdown("### New Path-Analysis Metrics")
            g1, g2, g3, g4, g5 = st.tabs(["TP1 Behaviour", "TP2 Behaviour", "STOP Behaviour", "Path / Risk", "Sample Size"])
            with g1:
                c1, c2 = st.columns(2)
                c1.metric("P(TP1 hit)", format_probability_pct(summ.get("p_tp1_hit")))
                c2.metric("P(TP1 before STOP)", format_probability_pct(summ.get("p_tp1_before_stop")))
                c3, c4 = st.columns(2)
                tp1_stats = summ.get("tp1_time_hours", {}) if isinstance(summ.get("tp1_time_hours"), dict) else {}
                c3.metric("Median time to TP1", format_hours(tp1_stats.get("median")))
                c4.metric("90% interval time to TP1", f"{format_hours(tp1_stats.get('p05'))} to {format_hours(tp1_stats.get('p95'))}")
            with g2:
                c1, c2 = st.columns(2)
                c1.metric("P(TP2 hit)", format_probability_pct(summ.get("p_tp2_hit")))
                c2.metric("P(TP2 after TP1)", format_probability_pct(summ.get("p_tp2_after_tp1")))
                c3, c4 = st.columns(2)
                tp2_stats = summ.get("tp2_time_hours", {}) if isinstance(summ.get("tp2_time_hours"), dict) else {}
                c3.metric("Median time to TP2", format_hours(tp2_stats.get("median")))
                c4.metric("90% interval time to TP2", f"{format_hours(tp2_stats.get('p05'))} to {format_hours(tp2_stats.get('p95'))}")
            with g3:
                c1, c2, c3 = st.columns(3)
                c1.metric("P(STOP hit)", format_probability_pct(summ.get("p_stop_hit")))
                c2.metric("P(STOP before TP1)", format_probability_pct(summ.get("p_stop_before_tp1")))
                stop_stats = summ.get("stop_time_hours", {}) if isinstance(summ.get("stop_time_hours"), dict) else {}
                c3.metric("Median time to STOP", format_hours(stop_stats.get("median")))
                c4, c5 = st.columns(2)
                c4.metric("90% interval time to STOP", f"{format_hours(stop_stats.get('p05'))} to {format_hours(stop_stats.get('p95'))}")
                c5.metric("Median MFE before STOP/fail", _format_r((summ.get("mfe_before_failure_stats", {}) or {}).get("median"), 2))
            with g4:
                mfe_stats = summ.get("mfe_in_range_stats", {}) if isinstance(summ.get("mfe_in_range_stats"), dict) else {}
                mae_stats = summ.get("mae_in_range_stats", {}) if isinstance(summ.get("mae_in_range_stats"), dict) else {}
                tp1_mae = summ.get("mae_before_tp1_winners_stats", {}) if isinstance(summ.get("mae_before_tp1_winners_stats"), dict) else {}
                tp2_mae = summ.get("mae_before_tp2_winners_stats", {}) if isinstance(summ.get("mae_before_tp2_winners_stats"), dict) else {}
                c1, c2, c3 = st.columns(3)
                c1.metric("Median MAE before TP1", _format_r(tp1_mae.get("median"), 2))
                c2.metric("P90 MAE before TP1", _format_r(tp1_mae.get("p90"), 2))
                c3.metric("P95 MAE before TP1", _format_r(tp1_mae.get("p95"), 2))
                c4, c5, c6 = st.columns(3)
                c4.metric("Median MAE before TP2", _format_r(tp2_mae.get("median"), 2))
                c5.metric("P90 MAE before TP2", _format_r(tp2_mae.get("p90"), 2))
                c6.metric("P95 MAE before TP2", _format_r(tp2_mae.get("p95"), 2))
                c7, c8, c9 = st.columns(3)
                c7.metric("Median MFE (overall)", _format_r(mfe_stats.get("median"), 2))
                c8.metric("Median MAE (overall)", _format_r(mae_stats.get("median"), 2))
                c9.metric("Std MAE (overall)", _format_r(mae_stats.get("std"), 2))
                sg = summ.get("stop_guidance", {}) if isinstance(summ.get("stop_guidance"), dict) else {}
                st.markdown("**Stop Placement Guidance**")
                g4c1, g4c2, g4c3 = st.columns(3)
                g4c1.metric("Conservative stop for TP1 trades", _format_r(sg.get("suggested_stop_tp1_conservative_r"), 2))
                g4c2.metric("Conservative stop for TP2 trades", _format_r(sg.get("suggested_stop_tp2_conservative_r"), 2))
                g4c3.metric("Very conservative stop", _format_r(sg.get("suggested_stop_very_conservative_r"), 2))
            with g5:
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", int(summ.get("total_rows", 0)))
                c2.metric("Weeks", int(summ.get("total_weeks", 0)))
                c3.metric("Weeks with sweep", int(summ.get("weeks_with_any_sweep", 0)))
                c4, c5, c6 = st.columns(3)
                c4.metric("Long candidates", int(summ.get("long_candidates", 0)))
                c5.metric("Short candidates", int(summ.get("short_candidates", 0)))
                c6.metric("Dataset", build_dataset_label(lab_symbol, lab_interval, lab_cache_status))
                st.caption(f"Entry window: {summ.get('start_entry_time', 'N/A')} -> {summ.get('end_entry_time', 'N/A')}")

            st.markdown("### Grouped Tables")
            by_dir = tables.get("by_direction")
            if isinstance(by_dir, pd.DataFrame) and len(by_dir):
                st.markdown("**By Direction**")
                st.dataframe(_format_research_table(by_dir), use_container_width=True)
            by_year = tables.get("by_year")
            if isinstance(by_year, pd.DataFrame) and len(by_year):
                st.markdown("**By Year**")
                st.dataframe(_format_research_table(by_year), use_container_width=True)
            by_bucket = tables.get("by_range_bucket")
            if isinstance(by_bucket, pd.DataFrame) and len(by_bucket):
                st.markdown("**By Monday Range Bucket**")
                st.dataframe(_format_research_table(by_bucket), use_container_width=True)
            by_vol = tables.get("by_vol_regime")
            if isinstance(by_vol, pd.DataFrame) and len(by_vol):
                st.markdown("**By Volatility Regime**")
                formatted = _format_research_table(by_vol)
                if "Vol Regime" in formatted.columns:
                    def _row_style(row: pd.Series) -> list[str]:
                        regime = str(row.get("Vol Regime", "")).strip().lower()
                        if regime == "low_vol":
                            css = "background-color: #0f2f4a; color: #e8f3ff;"
                        elif regime == "mid_vol":
                            css = "background-color: #2b2f36; color: #f3f4f6;"
                        elif regime == "high_vol":
                            css = "background-color: #4a2614; color: #fff1e8;"
                        else:
                            css = ""
                        return [css] * len(row)

                    styled = formatted.style.apply(_row_style, axis=1)
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.dataframe(formatted, use_container_width=True)

            by_trend = tables.get("by_trend_regime")
            if isinstance(by_trend, pd.DataFrame) and len(by_trend):
                st.markdown("**By Trend Regime**")
                st.dataframe(_format_research_table(by_trend), use_container_width=True)

            if isinstance(res, pd.DataFrame) and len(res):
                st.markdown("### All Candidate Outcomes")
                show_all = st.checkbox("Show all candidate outcomes", value=True, key="lab_show_all_candidates")
                if show_all:
                    st.dataframe(res, use_container_width=True)
                else:
                    max_rows = st.number_input("Rows to show", min_value=10, max_value=2000, value=200, step=10, key="lab_candidate_rows")
                    st.dataframe(res.head(int(max_rows)), use_container_width=True)

            with st.expander("Raw probability summary", expanded=False):
                st.json(summ)

if page == "Range Sweep Lab":
    st.subheader("Range Sweep Lab")
    st.warning("RESEARCH MODE ONLY — NO LIVE TRADING")
    st.caption(
        "Range Sweep Lab studies what happens after price sweeps previous day/week/month highs or lows. "
        "It uses cached public OHLCV data only."
    )

    with st.expander("Dataset", expanded=True):
        rs_ds = _render_dataset_controls("range_sweep", default_preset="Last 12 months")
    rs_symbol = rs_ds["symbol"]
    rs_base_interval = rs_ds["interval"]
    rs_start = rs_ds["start_date"]
    rs_end = rs_ds["end_date"]

    with st.expander("Sweep Setup", expanded=True):
        s1, s2 = st.columns(2)
        ref_label = s1.selectbox(
            "Reference range timeframe",
            ["Previous day", "Previous week", "Previous month"],
            index=0,
        )
        ref_map = {
            "Previous day": "previous_day",
            "Previous week": "previous_week",
            "Previous month": "previous_month",
        }
        reference_timeframe = ref_map[ref_label]

        execution_timeframe = s2.selectbox(
            "Execution / confirmation timeframe",
            ["1h", "2h", "4h", "6h", "12h", "1d"],
            index=0,
            help="Must be equal or higher resolution than cached base interval.",
        )

        c1, c2 = st.columns(2)
        confirmation_mode_label = c1.selectbox(
            "Confirmation mode",
            ["Close back inside", "Wick only", "Close outside / breakout continuation"],
            index=0,
        )
        confirmation_mode = {
            "Wick only": "wick_only",
            "Close back inside": "close_back_inside",
            "Close outside / breakout continuation": "close_outside",
        }[confirmation_mode_label]

        event_mode_label = c2.selectbox(
            "Event mode",
            ["First sweep per reference period", "All sweeps"],
            index=0,
        )
        event_mode = {
            "First sweep per reference period": "first_sweep_per_reference_period",
            "All sweeps": "all_sweeps",
        }[event_mode_label]

    with st.expander("Holding Window", expanded=True):
        h_mode_label = st.selectbox(
            "Holding window mode",
            [
                "Until end of reference period",
                "Fixed N candles",
                "Fixed hours",
            ],
            index=0,
        )
        holding_mode = {
            "Until end of reference period": "until_end_of_reference_period",
            "Fixed N candles": "fixed_n_candles",
            "Fixed hours": "fixed_hours",
        }[h_mode_label]
        fixed_n_candles = 48
        fixed_hours = 48
        if holding_mode == "fixed_n_candles":
            fixed_n_candles = int(st.number_input("Fixed N candles", min_value=1, max_value=5000, value=48, step=1))
        if holding_mode == "fixed_hours":
            fixed_hours = int(st.number_input("Fixed hours", min_value=1, max_value=5000, value=48, step=1))

    if not can_resample(rs_base_interval, execution_timeframe):
        st.error(
            f"Execution timeframe `{execution_timeframe}` is not compatible with cached base interval `{rs_base_interval}`. "
            "Choose equal or higher-resolution execution timeframe."
        )

    run_range_sweep = st.button("Run Range Sweep Analysis", type="primary")
    if run_range_sweep:
        try:
            if not can_resample(rs_base_interval, execution_timeframe):
                st.error("Cannot run analysis because timeframe selection is invalid.")
            else:
                rs_ohlc = load_cached_ohlc(rs_symbol, rs_base_interval)
                rs_ohlc = filter_ohlc_window(rs_ohlc, str(rs_start), str(rs_end))
                if rs_ohlc.empty:
                    st.error("No cached candles in selected date window.")
                else:
                    rs_out = run_range_sweep_lab_analysis(
                        ohlcv=rs_ohlc,
                        base_interval=rs_base_interval,
                        reference_timeframe=reference_timeframe,
                        execution_timeframe=execution_timeframe,
                        confirmation_mode=confirmation_mode,
                        event_mode=event_mode,
                        holding_window_mode=holding_mode,
                        fixed_n_candles=fixed_n_candles,
                        fixed_hours=fixed_hours,
                    )
                    st.session_state["range_sweep_last"] = rs_out
        except Exception as exc:
            st.error(f"Range Sweep Lab failed: {exc}")
            with st.expander("Debug details", expanded=False):
                st.code(traceback.format_exc())

    rs_payload = st.session_state.get("range_sweep_last")
    if isinstance(rs_payload, dict):
        rs_events = rs_payload.get("events")
        rs_summary = rs_payload.get("summary", {})
        rs_tables = rs_payload.get("tables", {})
        if isinstance(rs_events, pd.DataFrame) and len(rs_events):
            st.markdown("### Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Number of sweeps", int(rs_summary.get("num_sweeps", 0)))
            m2.metric("P(reclaim)", format_probability_pct(rs_summary.get("p_reclaim")))
            m3.metric("P(hit midpoint)", format_probability_pct(rs_summary.get("p_hit_midpoint")))
            m4.metric("P(hit opposite boundary)", format_probability_pct(rs_summary.get("p_hit_opposite_boundary")))
            m5.metric("P(continue 0.5R)", format_probability_pct(rs_summary.get("p_continue_0_5r")))
            m6, m7, m8, m9 = st.columns(4)
            m6.metric("P(continue 1.0R)", format_probability_pct(rs_summary.get("p_continue_1_0r")))
            m7.metric("Median time to midpoint", format_hours((rs_summary.get("time_to_midpoint_hours") or {}).get("median")))
            m8.metric("Median MAE", format_float(rs_summary.get("median_mae_r"), 2))
            m9.metric("P90 MAE", format_float(rs_summary.get("p90_mae_r"), 2))

            def _format_range_sweep_table(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                rename = {
                    "side": "Side",
                    "year": "Year",
                    "vol_regime": "Volatility Regime",
                    "trend_regime": "Trend Regime",
                    "range_bucket": "Range Bucket",
                    "samples": "Samples",
                    "p_reclaim": "P(reclaim)",
                    "p_hit_midpoint": "P(hit midpoint)",
                    "p_hit_opposite_boundary": "P(hit opposite)",
                    "p_continue_0_5r": "P(continue 0.5R)",
                    "p_continue_1_0r": "P(continue 1.0R)",
                    "median_time_to_midpoint_h": "Median time to midpoint (h)",
                    "median_mfe_r": "Median MFE (R)",
                    "median_mae_r": "Median MAE (R)",
                    "p90_mae_r": "P90 MAE (R)",
                    "median_overshoot_r": "Median overshoot (R)",
                }
                out = out.rename(columns=rename)
                for c in [c for c in out.columns if c.startswith("P(")]:
                    out[c] = out[c].map(format_probability_pct)
                for c in [c for c in out.columns if "(h)" in c]:
                    out[c] = out[c].map(format_hours)
                for c in [c for c in out.columns if "(R)" in c]:
                    out[c] = out[c].map(lambda x: format_float(x, 2))
                return out

            st.markdown("### Grouped Tables")
            t_side = rs_tables.get("by_side")
            if isinstance(t_side, pd.DataFrame) and len(t_side):
                st.markdown("**By Side**")
                st.dataframe(_format_range_sweep_table(t_side), use_container_width=True)
            t_year = rs_tables.get("by_year")
            if isinstance(t_year, pd.DataFrame) and len(t_year):
                st.markdown("**By Year**")
                st.dataframe(_format_range_sweep_table(t_year), use_container_width=True)
            t_vol = rs_tables.get("by_vol_regime")
            if isinstance(t_vol, pd.DataFrame) and len(t_vol):
                st.markdown("**By Volatility Regime**")
                st.dataframe(_format_range_sweep_table(t_vol), use_container_width=True)
            t_trend = rs_tables.get("by_trend_regime")
            if isinstance(t_trend, pd.DataFrame) and len(t_trend):
                st.markdown("**By Trend Regime**")
                st.dataframe(_format_range_sweep_table(t_trend), use_container_width=True)
            t_bucket = rs_tables.get("by_range_bucket")
            if isinstance(t_bucket, pd.DataFrame) and len(t_bucket):
                st.markdown("**By Range Size Bucket**")
                st.dataframe(_format_range_sweep_table(t_bucket), use_container_width=True)

            st.markdown("### All Sweep Events")
            f1, f2, f3, f4, f5 = st.columns(5)
            side_filter = f1.selectbox("Side", ["All", "high_sweep", "low_sweep"], index=0, key="rs_side")
            years = sorted(rs_events["year"].dropna().astype(int).unique().tolist()) if "year" in rs_events.columns else []
            year_filter = f2.selectbox("Year", ["All"] + [str(y) for y in years], index=0, key="rs_year")
            confirm_filter = f3.selectbox("Confirmation", ["All", "wick_only", "close_back_inside", "close_outside"], index=0, key="rs_confirm")
            hit_mid_filter = f4.selectbox("Hit midpoint", ["All", "True", "False"], index=0, key="rs_hit_mid")
            hit_opp_filter = f5.selectbox("Hit opposite", ["All", "True", "False"], index=0, key="rs_hit_opp")
            view = rs_events.copy()
            if side_filter != "All":
                view = view[view["side"] == side_filter]
            if year_filter != "All":
                view = view[view["year"].astype(int) == int(year_filter)]
            if confirm_filter != "All":
                view = view[view["confirmation_mode"] == confirm_filter]
            if hit_mid_filter != "All":
                view = view[view["hit_midpoint"] == (hit_mid_filter == "True")]
            if hit_opp_filter != "All":
                view = view[view["hit_opposite_boundary"] == (hit_opp_filter == "True")]
            st.dataframe(view, use_container_width=True)

            with st.expander("Raw summary JSON", expanded=False):
                st.json(rs_summary)
        else:
            st.info("No sweep events found for the current settings.")

if page == "Strategy State":
    st.subheader("Strategy State")
    if state is None:
        st.info("No paper state file found at `data/paper_state.json`.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Monday High", f"{snapshot['mon_high']:.4f}" if snapshot["mon_high"] is not None else "N/A")
    c2.metric("Monday Low", f"{snapshot['mon_low']:.4f}" if snapshot["mon_low"] is not None else "N/A")
    c3.metric("Monday Mid", f"{snapshot['mon_mid']:.4f}" if snapshot["mon_mid"] is not None else "N/A")

    c4, c5, c6 = st.columns(3)
    c4.metric("Current Price", f"{snapshot['current_price']:.4f}" if snapshot["current_price"] is not None else "N/A")
    c5.metric("Pending Entry", str(snapshot["pending_entry"]) if snapshot["pending_entry"] is not None else "None")
    c6.metric("Entry Price", f"{float(snapshot['entry_price']):.4f}" if snapshot["entry_price"] is not None else "N/A")

    c7, c8, c9 = st.columns(3)
    c7.metric("Stop", f"{float(snapshot['stop']):.4f}" if snapshot["stop"] is not None else "N/A")
    c8.metric("TP1", f"{float(snapshot['tp1']):.4f}" if snapshot["tp1"] is not None else "N/A")
    c9.metric("TP2", f"{float(snapshot['tp2']):.4f}" if snapshot["tp2"] is not None else "N/A")

    st.metric("Friday Cutoff Hour (UTC)", snapshot["friday_cutoff_hour_utc"])

if page == "Trade Journal":
    st.subheader("Trade Journal")
    if events.empty:
        st.info("No paper trade log found at `data/paper_trades.csv`.")
    else:
        df = events.copy()
        if "timestamp" in df.columns:
            min_d = df["timestamp"].min().date()
            max_d = df["timestamp"].max().date()
        else:
            min_d = max_d = pd.Timestamp.now(tz="UTC").date()

        col1, col2, col3, col4 = st.columns(4)
        event_opts = ["All"] + sorted(df["event"].dropna().astype(str).unique().tolist()) if "event" in df.columns else ["All"]
        side_opts = ["All"] + sorted(df["side"].dropna().astype(str).unique().tolist()) if "side" in df.columns else ["All"]
        symbol_opts = ["All"] + sorted(df["symbol"].dropna().astype(str).unique().tolist()) if "symbol" in df.columns else ["All"]
        event_f = col1.selectbox("Event", event_opts)
        side_f = col2.selectbox("Side", side_opts)
        symbol_f = col3.selectbox("Symbol", symbol_opts)
        date_f = col4.date_input("Date Range", value=(min_d, max_d))

        if event_f != "All" and "event" in df.columns:
            df = df[df["event"] == event_f]
        if side_f != "All" and "side" in df.columns:
            df = df[df["side"] == side_f]
        if symbol_f != "All" and "symbol" in df.columns:
            df = df[df["symbol"] == symbol_f]
        if isinstance(date_f, tuple) and len(date_f) == 2 and "timestamp" in df.columns:
            d0 = pd.Timestamp(date_f[0], tz="UTC")
            d1 = pd.Timestamp(date_f[1], tz="UTC") + pd.Timedelta(days=1)
            df = df[(df["timestamp"] >= d0) & (df["timestamp"] < d1)]

        st.dataframe(df, use_container_width=True)
        st.caption("Events include ENTRY / TP1 / TP2 / STOP / FRIDAY.")

if page == "Performance":
    st.subheader("Performance")
    if events.empty:
        st.info("No events available to compute performance.")
    else:
        perf = events.copy()
        perf = perf.sort_values("timestamp")
        if "realized_pnl" in perf.columns:
            perf["cum_realized_pnl"] = perf["realized_pnl"].cumsum()
            st.line_chart(perf.set_index("timestamp")[["realized_pnl", "cum_realized_pnl"]])
        else:
            st.info("No `realized_pnl` column available.")

        completed = reconstruct_completed_trades(perf)
        total_fees = float(perf["fee"].sum()) if "fee" in perf.columns else 0.0
        wins = int((completed["pnl"] > 0).sum()) if len(completed) else 0
        losses = int((completed["pnl"] <= 0).sum()) if len(completed) else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Completed Trades", int(len(completed)))
        c2.metric("Wins", wins)
        c3.metric("Losses", losses)
        c4.metric("Total Fees Paid", f"{total_fees:.4f}")

        if len(completed):
            st.dataframe(completed, use_container_width=True)

if page == "Data Health":
    st.subheader("Data Health")
    if not health["has_candles"]:
        st.info("No local Binance cache found under `data/binance/`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Candle Age (s)", f"{health['latest_candle_age_seconds']:.0f}" if health["latest_candle_age_seconds"] is not None else "N/A")
        c2.metric("Duplicate Candles", health["duplicate_candles"])
        c3.metric("Missing Candles (est.)", health["missing_candles_estimate"] if health["missing_candles_estimate"] is not None else "N/A")
        c4.metric("UTC Timestamps", "OK" if health["timestamps_utc_ok"] else "Check")

        if cache_meta is not None:
            st.caption(f"Using cache file: `{cache_meta.path}`")

if page == "Config":
    st.subheader("Config")
    st.warning("LIVE TRADING DISABLED")
    st.json(
        {
            "mode": "PAPER ONLY",
            "default_symbol": "BTCUSDT",
            "default_interval": "1h",
            "default_risk_per_trade": 100.0,
            "default_stop_mult": 1.0,
            "default_tp1_range_fraction": 0.5,
            "default_tp2_range_fraction": 1.0,
            "default_fee_bps": 10.0,
            "default_slippage_bps": 5.0,
            "default_friday_cutoff_hour_utc": 23,
            "state_file": "data/paper_state.json",
            "trade_log_file": "data/paper_trades.csv",
            "cache_dir": "data/binance",
            "backtest_dir": "data/backtests",
        }
    )
