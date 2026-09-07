from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping

import altair as alt
import pandas as pd
import streamlit as st

from src.portfolio_results import (
    EXPECTED_CANDIDATE_ID,
    HOLDOUT_START,
    RUN_ID,
    ResearchArtifactError,
    equity_frame,
    load_canonical_research,
    run_exploratory_backtest,
    walk_forward_table,
    weekly_return_frame,
)

st.set_page_config(
    page_title="Monday Range Research",
    page_icon="↔",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#3676d8"
AMBER = "#b7791f"
NEGATIVE = "#ba4a4a"
MUTED = "#738096"
REPO_ROOT = Path(__file__).resolve().parent

st.markdown(
    """
    <style>
    :root { --content-width: 1180px; }
    .block-container { max-width: var(--content-width); padding-top: 3.25rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] { border-right: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent); }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    h1 { letter-spacing: -0.04em; font-weight: 680; }
    h2 { letter-spacing: -0.025em; margin-top: 2rem; }
    h3 { letter-spacing: -0.015em; }
    .eyebrow { color: var(--primary-color); font-size: .76rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .55rem; }
    .hero-copy { color: color-mix(in srgb, var(--text-color) 72%, transparent); font-size: 1.13rem; line-height: 1.65; max-width: 780px; margin: -.4rem 0 1.35rem; }
    .conclusion { border-left: 4px solid #b7791f; background: color-mix(in srgb, #b7791f 8%, var(--background-color)); border-radius: 0 .7rem .7rem 0; padding: 1rem 1.15rem; line-height: 1.55; margin: .8rem 0 1.5rem; }
    .metric-card { min-height: 126px; border: 1px solid color-mix(in srgb, var(--text-color) 11%, transparent); border-radius: .85rem; padding: 1rem 1.05rem; background: color-mix(in srgb, var(--secondary-background-color) 74%, transparent); }
    .metric-label { color: color-mix(in srgb, var(--text-color) 63%, transparent); font-size: .78rem; font-weight: 650; letter-spacing: .045em; text-transform: uppercase; }
    .metric-value { font-size: 1.75rem; font-weight: 680; letter-spacing: -.035em; line-height: 1.2; margin: .32rem 0 .28rem; }
    .metric-note { color: color-mix(in srgb, var(--text-color) 60%, transparent); font-size: .80rem; line-height: 1.35; }
    .tone-amber { color: #b7791f; }
    .tone-blue { color: #3676d8; }
    .tone-red { color: #ba4a4a; }
    .tone-neutral { color: var(--text-color); }
    .section-kicker { color: color-mix(in srgb, var(--text-color) 58%, transparent); font-size: .82rem; font-weight: 650; letter-spacing: .07em; text-transform: uppercase; margin-top: 2.25rem; }
    .section-copy { color: color-mix(in srgb, var(--text-color) 70%, transparent); max-width: 800px; line-height: 1.6; }
    .flow-card { min-height: 140px; border-top: 3px solid #3676d8; border-radius: .55rem; padding: .9rem .8rem; background: color-mix(in srgb, var(--secondary-background-color) 78%, transparent); }
    .flow-number { color: #3676d8; font-size: .72rem; font-weight: 750; letter-spacing: .09em; }
    .flow-title { font-size: .95rem; font-weight: 680; margin: .4rem 0 .3rem; }
    .flow-text { color: color-mix(in srgb, var(--text-color) 63%, transparent); font-size: .79rem; line-height: 1.35; }
    .research-label { display: inline-block; border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent); border-radius: 99px; padding: .24rem .58rem; font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; margin-bottom: .6rem; }
    .label-in { color: #8a5b16; background: color-mix(in srgb, #b7791f 8%, transparent); }
    .label-oos { color: #306bbf; background: color-mix(in srgb, #3676d8 8%, transparent); }
    .label-final { color: #7446a8; background: color-mix(in srgb, #8557b5 8%, transparent); }
    .footnote { color: color-mix(in srgb, var(--text-color) 55%, transparent); font-size: .76rem; line-height: 1.5; }
    .sidebar-brand { font-size: 1.02rem; font-weight: 720; letter-spacing: -.02em; margin-bottom: .2rem; }
    .sidebar-sub { color: color-mix(in srgb, var(--text-color) 55%, transparent); font-size: .76rem; line-height: 1.4; margin-bottom: 1.2rem; }
    .status-dot { color: #b7791f; font-size: .72rem; margin-right: .35rem; }
    @media (max-width: 700px) {
      .block-container { padding-top: 2.5rem; }
      .metric-card { min-height: 108px; margin-bottom: .45rem; }
      .metric-value { font-size: 1.45rem; }
      .flow-card { min-height: auto; margin-bottom: .4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _research():
    return load_canonical_research()


@st.cache_data(show_spinner=False)
def _run_explore_cached(
    start_iso: str,
    end_iso: str,
    stop_offset: float,
    latest_entry_day: int,
    target_plan: str,
    fee_bps: float,
    slippage_bps: float,
    intrabar_policy: str,
    direction: str,
):
    return run_exploratory_backtest(
        start_inclusive=start_iso,
        end_exclusive=end_iso,
        stop_offset=stop_offset,
        latest_entry_day=latest_entry_day,
        target_plan=target_plan,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        intrabar_policy=intrabar_policy,
        direction=direction,
    )


try:
    research = _research()
except ResearchArtifactError as exc:
    st.error("The canonical research bundle could not be verified.")
    st.code(str(exc))
    st.stop()


def pct(value: Any, digits: int = 2, sign: bool = True) -> str:
    if value is None or pd.isna(value):
        return "—"
    prefix = "+" if sign and float(value) > 0 else ""
    return f"{prefix}{float(value) * 100:.{digits}f}%"


def num(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{float(value):,.2f}"


def metric_card(label: str, value: str, note: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{escape(label)}</div>
          <div class="metric-value tone-{escape(tone)}">{escape(value)}</div>
          <div class="metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_intro(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="hero-copy">{escape(copy)}</div>', unsafe_allow_html=True)


def section(title: str, copy: str | None = None, label: str | None = None, label_class: str = "") -> None:
    if label:
        st.markdown(
            f'<div class="research-label {escape(label_class)}">{escape(label)}</div>',
            unsafe_allow_html=True,
        )
    st.subheader(title)
    if copy:
        st.markdown(f'<div class="section-copy">{escape(copy)}</div>', unsafe_allow_html=True)


def daily_curve(result: Mapping[str, Any]) -> pd.DataFrame:
    curve = equity_frame(result)
    if curve.empty:
        return curve
    return (
        curve.set_index("timestamp")
        .resample("1D")
        .agg({"equity": "last", "cash": "last", "drawdown": "min"})
        .dropna(subset=["equity"])
        .reset_index()
    )


def equity_chart(result: Mapping[str, Any], title: str, color: str = ACCENT) -> alt.Chart:
    curve = daily_curve(result)
    return (
        alt.Chart(curve, title=title)
        .mark_line(color=color, strokeWidth=2.2)
        .encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y=alt.Y("equity:Q", title="Portfolio equity", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("timestamp:T", title="Date"), alt.Tooltip("equity:Q", title="Equity", format=",.2f")],
        )
        .properties(height=300)
    )


def drawdown_chart(result: Mapping[str, Any], title: str) -> alt.Chart:
    curve = daily_curve(result).copy()
    curve["drawdown_pct"] = curve["drawdown"] * 100
    return (
        alt.Chart(curve, title=title)
        .mark_area(color=NEGATIVE, opacity=0.28, line={"color": NEGATIVE, "strokeWidth": 1.3})
        .encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y=alt.Y("drawdown_pct:Q", title="Drawdown (%)"),
            tooltip=[alt.Tooltip("timestamp:T", title="Date"), alt.Tooltip("drawdown_pct:Q", title="Drawdown", format=".2f")],
        )
        .properties(height=210)
    )


def weekly_histogram(result: Mapping[str, Any]) -> alt.Chart:
    weekly = weekly_return_frame(result).copy()
    weekly["return_pct"] = weekly["net_return"] * 100
    return (
        alt.Chart(weekly, title="Distribution of 52 weekly net returns")
        .mark_bar(color=ACCENT, opacity=0.75)
        .encode(
            x=alt.X("return_pct:Q", bin=alt.Bin(maxbins=16), title="Weekly return (%)"),
            y=alt.Y("count():Q", title="Weeks"),
            tooltip=[alt.Tooltip("count():Q", title="Weeks")],
        )
        .properties(height=210)
    )


def result_metrics_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows = [
        ("Net total return", pct(metrics.get("total_return"), 4)),
        ("CAGR", pct(metrics.get("net_cagr"), 4)),
        ("Maximum drawdown", pct(metrics.get("max_drawdown"), 4)),
        ("Sharpe", num(metrics.get("sharpe"), 3)),
        ("Sortino", num(metrics.get("sortino"), 3)),
        ("Profit factor", num(metrics.get("profit_factor"), 3)),
        ("Trades", f"{int(metrics.get('num_trades', 0))}"),
        ("Win rate", pct(metrics.get("win_rate"), 2, sign=False)),
        ("Average trade return", pct(metrics.get("mean_net_trade_return"), 4)),
        ("Median trade return", pct(metrics.get("median_net_trade_return"), 4)),
        ("Time exposed", pct(metrics.get("time_exposed_fraction"), 2, sign=False)),
        ("Annualized turnover", f"{num(metrics.get('annualized_turnover_multiple'), 2)}×"),
        ("Fees", num(metrics.get("fees"), 2)),
        ("Explicit slippage", num(metrics.get("total_slippage_cost"), 2)),
        ("Combined execution cost", num(metrics.get("combined_execution_cost"), 2)),
        ("Maximum marked leverage", f"{num(metrics.get('maximum_leverage_used'), 3)}×"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def scenario_output(payload: Mapping[str, Any], scenario_id: str) -> Mapping[str, Any]:
    for output in payload.get("outputs", []):
        if output.get("scenario", {}).get("id") == scenario_id:
            return output["result"]
    raise KeyError(scenario_id)


def provenance_expander() -> None:
    manifest = research.dataset_manifest
    with st.expander("Data and experiment provenance"):
        left, right = st.columns(2)
        with left:
            st.markdown("**Canonical market data**")
            st.write(f"{manifest['symbol']} · {manifest['venue']} · {manifest['interval']} · UTC")
            st.write(f"{manifest['first_timestamp']} → {manifest['last_timestamp']}")
            st.write(f"{int(manifest['row_count']):,} completed candles · {manifest['missing_timestamp_count']} genuine missing hours")
            st.code(manifest["sha256"], language=None)
        with right:
            st.markdown("**Sealed experiment**")
            st.write(f"Run `{RUN_ID}`")
            st.write(f"Code commit `{research.provenance['git_commit'][:7]}` · accounting `{research.provenance['accounting_version']}`")
            st.write(f"Holdout opened once at {research.access['authorized_at_utc']}")
            st.code(research.access["protocol_hash"], language=None)
        st.caption("Every JSON artifact is checked against its semantic SHA-256 sidecar when the application loads.")


with st.sidebar:
    st.markdown('<div class="sidebar-brand">Monday Range Research</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">A reproducible BTC mean-reversion study</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "Strategy & Method", "Research Results", "Robustness & Uncertainty", "Explore"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<span class="status-dot">●</span> **Protocol V1 complete**', unsafe_allow_html=True)
    st.caption("Final conclusion: null not rejected")
    st.caption(f"Run `{RUN_ID}`")

holdout_metrics = research.selected["metrics"]
bootstrap = research.holdout_uncertainty

if page == "Overview":
    page_intro(
        "Final · Protocol V1",
        "Monday Range Research",
        "Testing whether BTC mean-reverts after sweeping the completed Monday trading range.",
    )
    st.markdown(
        """
        <div class="conclusion"><strong>The holdout point estimate was positive, but the evidence was not strong enough.</strong><br>
        The selected strategy returned +1.52% on the untouched 52-week holdout with a −2.02% maximum drawdown. Its preregistered bootstrap interval crossed zero, and the result disappeared under 1.5× transaction costs. Protocol V1 therefore did not establish a reliable positive edge.</div>
        """,
        unsafe_allow_html=True,
    )
    cards = st.columns(4)
    with cards[0]:
        metric_card("Holdout return", pct(holdout_metrics["total_return"], 2), "Untouched 52-week test", "blue")
    with cards[1]:
        metric_card("Maximum drawdown", pct(holdout_metrics["max_drawdown"], 2), "Hourly marked equity", "red")
    with cards[2]:
        metric_card("Completed trades", str(holdout_metrics["num_trades"]), "At most one per ISO week", "neutral")
    with cards[3]:
        metric_card("Bootstrap test", "Null not rejected", "95% interval includes zero", "amber")

    st.markdown('<div class="section-kicker">Untouched holdout</div>', unsafe_allow_html=True)
    chart_left, chart_right = st.columns([1.65, 1])
    with chart_left:
        st.altair_chart(equity_chart(research.selected, "Holdout marked-equity curve"), width="stretch")
        st.caption("Daily display samples from the sealed hourly mark-to-market series. Initial capital: 10,000 quote units.")
    with chart_right:
        st.altair_chart(weekly_histogram(research.selected), width="stretch")
        st.caption(f"Mean: {pct(bootstrap['point_estimate'], 5)} · 95% block-bootstrap interval: {pct(bootstrap['percentile_2_5'], 5)} to {pct(bootstrap['percentile_97_5'], 5)}.")
    st.altair_chart(drawdown_chart(research.selected, "Holdout drawdown"), width="stretch")

    st.info("Short trades are synthetic research positions evaluated on a Binance spot BTCUSDT price series; they are not spot short executions.", icon="ℹ️")
    provenance_expander()

elif page == "Strategy & Method":
    page_intro(
        "Strategy",
        "One weekly setup, tested as specified",
        "A completed Monday sets the reference range. From Tuesday through Wednesday, the strategy fades the first sweep that closes back inside that range.",
    )
    flow = [
        ("01", "Build the range", "Monday's completed UTC candles define the weekly high and low."),
        ("02", "Sweep a boundary", "Price trades beyond the Monday high or low during an eligible day."),
        ("03", "Close back inside", "A completed hourly candle must return inside the range."),
        ("04", "Enter next bar", "The fade enters at the following hourly open, after adverse slippage."),
        ("05", "Manage risk", "The stop sits 1.25 Monday-range widths beyond the swept boundary."),
        ("06", "Exit", "Close at the midpoint, or on Friday at 23:00 UTC if still open."),
    ]
    cols = st.columns(3)
    for idx, (number, title, text) in enumerate(flow):
        with cols[idx % 3]:
            st.markdown(
                f'<div class="flow-card"><div class="flow-number">{number}</div><div class="flow-title">{escape(title)}</div><div class="flow-text">{escape(text)}</div></div>',
                unsafe_allow_html=True,
            )
            if idx == 2:
                st.write("")

    section("Selected configuration", "Chosen mechanically from the frozen 18-candidate grid using only pre-holdout data.", "Sealed candidate", "label-final")
    config_cols = st.columns(5)
    with config_cols[0]: metric_card("Stop", "1.25R", "Beyond swept boundary", "neutral")
    with config_cols[1]: metric_card("Entry window", "Tue–Wed", "Next hourly open", "neutral")
    with config_cols[2]: metric_card("Target", "Midpoint", "Full position exit", "neutral")
    with config_cols[3]: metric_card("Risk", "1.00%", "Current marked equity", "neutral")
    with config_cols[4]: metric_card("Leverage cap", "1.00×", "Gross notional", "neutral")

    left, right = st.columns(2)
    with left, st.expander("Execution conventions", expanded=True):
        st.markdown(
            """
            - **Signal timing:** a completed prior bar generates the signal; entry occurs at the next bar open.
            - **Costs:** 10 bps fee and 5 bps adverse slippage on every fill.
            - **Gaps:** an opening price through a stop or target is used before adverse slippage.
            - **Intrabar ambiguity:** if stop and target are both touched, the conservative stop-first path applies.
            - **Frequency:** at most one position-producing signal per ISO week.
            """
        )
    with right, st.expander("Portfolio accounting", expanded=True):
        st.markdown(
            """
            - **Mark:** open positions are marked at every hourly close.
            - **Equity:** cash plus marked position value, including unrealized P&L.
            - **Sizing:** 1% of current marked equity is put at risk, bounded by 1× gross notional.
            - **Costs:** fees and explicit slippage are recorded separately and deducted from equity.
            - **Terminal handling:** any remaining position closes on the final completed bar with normal exit costs.
            """
        )
    with st.expander("Frozen objective and research controls"):
        st.code("score = net CAGR − 0.50 × |max drawdown| − annualized cost drag − 0.0005 × annualized turnover", language=None)
        st.write("Candidates needed at least 30 completed training trades. Ties were broken by Sortino, absolute drawdown, turnover, then candidate ID. Validation performance never selected a training winner.")
    provenance_expander()

elif page == "Research Results":
    page_intro(
        "Evidence",
        "Three distinct research stages",
        "Development selected parameters, walk-forward measured repeated out-of-sample behavior, and the sealed holdout supplied the final untouched test. Their curves are kept separate.",
    )

    dev_top = research.development_ranking["candidate_ranking"][0]
    section("Development", "This period was used for parameter selection and is not out-of-sample evidence.", "In sample", "label-in")
    dcols = st.columns(4)
    with dcols[0]: metric_card("Top candidate return", pct(dev_top["total_return"]), "2021-05-31 → 2024-01-01", "neutral")
    with dcols[1]: metric_card("Top candidate", "Stop 1.00R", "Thursday · opposite boundary", "neutral")
    with dcols[2]: metric_card("Trades", str(dev_top["num_trades"]), "Eligibility minimum: 30", "neutral")
    with dcols[3]: metric_card("Objective score", num(dev_top["score"], 4), "Frozen risk/cost objective", "neutral")

    wf_metrics = research.walk_forward["metrics"]
    section("Walk-forward validation", "Each 24-week fold selected one winner from training data, sealed it, then evaluated only that candidate on the next unseen window.", "Out of sample", "label-oos")
    wcols = st.columns(4)
    with wcols[0]: metric_card("Aggregate return", pct(wf_metrics["total_return"]), "72 non-overlapping weeks", "red")
    with wcols[1]: metric_card("Maximum drawdown", pct(wf_metrics["maximum_drawdown"]), "Hourly marked aggregate", "red")
    with wcols[2]: metric_card("Trades", str(wf_metrics["trade_count"]), "Across three folds", "neutral")
    with wcols[3]: metric_card("Mean weekly return", pct(wf_metrics["mean_weekly_net_return"], 3), "All validation weeks", "red")

    fold_table = walk_forward_table(research).copy()
    display_folds = fold_table.copy()
    display_folds["Validation return"] = display_folds["Validation return"].map(lambda value: pct(value, 2))
    display_folds["Max drawdown"] = display_folds["Max drawdown"].map(lambda value: pct(value, 2))
    display_folds["Profit factor"] = display_folds["Profit factor"].map(lambda value: num(value, 3))
    st.dataframe(display_folds, hide_index=True, width="stretch")
    wf_curve = pd.DataFrame(research.walk_forward["equity_curve"])
    wf_curve["timestamp"] = pd.to_datetime(wf_curve["timestamp"], utc=True)
    wf_daily = wf_curve.set_index("timestamp")["equity"].resample("1D").last().dropna().reset_index()
    wf_chart = alt.Chart(wf_daily, title="Walk-forward aggregate equity — validation only").mark_line(color=MUTED, strokeWidth=2).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0)),
        y=alt.Y("equity:Q", title="Chained equity", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("timestamp:T", title="Date"), alt.Tooltip("equity:Q", format=",.2f")],
    ).properties(height=250)
    st.altair_chart(wf_chart, width="stretch")

    section("Final untouched holdout", "The final candidate was fixed using all pre-holdout data, then this interval was opened once under the executor lock.", "Final test", "label-final")
    compare = pd.DataFrame(
        [
            {"Configuration": "Selected candidate", "Return": research.selected["metrics"]["total_return"] * 100},
            {"Configuration": "Frozen baseline", "Return": research.baseline["metrics"]["total_return"] * 100},
        ]
    )
    compare_chart = alt.Chart(compare, title="Holdout comparison").mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Configuration:N", title=None, sort=None),
        x=alt.X("Return:Q", title="Net total return (%)"),
        color=alt.Color("Configuration:N", scale=alt.Scale(domain=["Selected candidate", "Frozen baseline"], range=[ACCENT, MUTED]), legend=None),
        tooltip=["Configuration:N", alt.Tooltip("Return:Q", format="+.2f")],
    ).properties(height=175)
    main, details = st.columns([1.55, 1])
    with main:
        st.altair_chart(equity_chart(research.selected, "Selected candidate — untouched holdout"), width="stretch")
        st.altair_chart(compare_chart, width="stretch")
    with details:
        st.dataframe(result_metrics_table(holdout_metrics), hide_index=True, width="stretch", height=605)
    st.warning("The selected candidate beat the frozen baseline in the holdout, but the bootstrap interval includes zero. This comparison does not change the preregistered statistical conclusion.", icon="⚠️")
    provenance_expander()

elif page == "Robustness & Uncertainty":
    page_intro(
        "Sensitivity",
        "A positive estimate with fragile margins",
        "The predefined checks probe costs, intrabar ordering, direction, time periods, data gaps, and the frozen parameter neighborhood. They diagnose the selected candidate; none can replace it.",
    )

    section("Cost sensitivity", "Moderately higher transaction costs erased the holdout gain.")
    cost_rows = [
        {"Scenario": "Baseline costs", "Return": holdout_metrics["total_return"] * 100, "Cost": holdout_metrics["combined_execution_cost"]},
    ]
    for scenario, label in (("costs_1_5x", "1.5× costs"), ("costs_2x", "2× costs")):
        item = scenario_output(research.holdout_robustness, scenario)["metrics"]
        cost_rows.append({"Scenario": label, "Return": item["total_return"] * 100, "Cost": item["combined_execution_cost"]})
    costs = pd.DataFrame(cost_rows)
    cost_chart = alt.Chart(costs, title="Untouched holdout under frozen cost stresses").mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X("Scenario:N", title=None, sort=["Baseline costs", "1.5× costs", "2× costs"]),
        y=alt.Y("Return:Q", title="Net total return (%)"),
        color=alt.condition(alt.datum.Return >= 0, alt.value(ACCENT), alt.value(NEGATIVE)),
        tooltip=["Scenario:N", alt.Tooltip("Return:Q", format="+.3f"), alt.Tooltip("Cost:Q", title="Execution cost", format=",.2f")],
    ).properties(height=280)
    st.altair_chart(cost_chart, width="stretch")
    st.caption("Baseline: +1.5248% · 1.5× costs: −0.0580% · 2× costs: −1.5955%.")

    section("Bootstrap uncertainty", "Four-week moving blocks preserve short-range dependence in the 52 weekly observations. The interval is the preregistered decision rule.")
    ci = pd.DataFrame([{
        "Statistic": "Mean weekly net return",
        "lower": bootstrap["percentile_2_5"] * 100,
        "upper": bootstrap["percentile_97_5"] * 100,
        "point": bootstrap["point_estimate"] * 100,
    }])
    zero = pd.DataFrame({"x": [0.0]})
    interval = alt.Chart(ci).mark_rule(color=ACCENT, strokeWidth=6).encode(
        x=alt.X("lower:Q", title="Mean weekly net return (%)"), x2="upper:Q", y=alt.Y("Statistic:N", title=None)
    )
    point = alt.Chart(ci).mark_point(color="#172033", filled=True, size=150).encode(x="point:Q", y="Statistic:N", tooltip=[alt.Tooltip("point:Q", format="+.5f"), alt.Tooltip("lower:Q", format="+.5f"), alt.Tooltip("upper:Q", format="+.5f")])
    zero_rule = alt.Chart(zero).mark_rule(color=NEGATIVE, strokeDash=[5, 4]).encode(x="x:Q")
    st.altair_chart((interval + point + zero_rule).properties(height=100, title="95% moving-block bootstrap interval"), width="stretch")
    st.markdown(
        f"""<div class="conclusion"><strong>Protocol V1 does not reject the null.</strong><br>
        Point estimate: {pct(bootstrap['point_estimate'], 5)} per week. 95% interval: {pct(bootstrap['percentile_2_5'], 5)} to {pct(bootstrap['percentile_97_5'], 5)}. The lower bound is below zero.</div>""",
        unsafe_allow_html=True,
    )

    section("Other predefined diagnostics")
    long_short = holdout_metrics["long_short_decomposition"]
    direction = pd.DataFrame([
        {"Direction": "Long", "Net contribution": long_short.get("LONG", {}).get("net_pnl", 0.0)},
        {"Direction": "Synthetic short", "Net contribution": long_short.get("SHORT", {}).get("net_pnl", 0.0)},
    ])
    exits = pd.DataFrame([
        {"Exit": key.title(), "Trades": value}
        for key, value in holdout_metrics["exit_reason_distribution"].items()
    ])
    c1, c2 = st.columns(2)
    with c1:
        chart = alt.Chart(direction, title="Holdout contribution by direction").mark_bar(cornerRadiusEnd=4, color=ACCENT).encode(
            x=alt.X("Net contribution:Q", title="Net P&L (quote units)"), y=alt.Y("Direction:N", title=None, sort=None), tooltip=["Direction:N", alt.Tooltip("Net contribution:Q", format="+.2f")]
        ).properties(height=150)
        st.altair_chart(chart, width="stretch")
    with c2:
        chart = alt.Chart(exits, title="Holdout exit reasons").mark_bar(cornerRadiusEnd=4, color=MUTED).encode(
            x=alt.X("Trades:Q", title="Trades"), y=alt.Y("Exit:N", title=None, sort="-x"), tooltip=["Exit:N", "Trades:Q"]
        ).properties(height=150)
        st.altair_chart(chart, width="stretch")

    pre = research.preholdout_robustness
    year_output = scenario_output(pre, "calendar_year_subperiods")
    years = pd.DataFrame(year_output["subperiods"])
    years["return_pct"] = years["total_return"] * 100
    year_chart = alt.Chart(years, title="Pre-holdout calendar-year slices").mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("year:O", title=None), y=alt.Y("return_pct:Q", title="Return (%)"),
        color=alt.condition(alt.datum.return_pct >= 0, alt.value(ACCENT), alt.value(NEGATIVE)),
        tooltip=["year:O", alt.Tooltip("return_pct:Q", format="+.2f")],
    ).properties(height=220)
    st.altair_chart(year_chart, width="stretch")

    neighborhood = scenario_output(pre, "parameter_neighborhood")["candidate_diagnostics"]
    neighborhood_frame = pd.DataFrame(neighborhood).sort_values("total_return", ascending=True)
    neighborhood_frame["return_pct"] = neighborhood_frame["total_return"] * 100
    neighborhood_frame["Selected"] = neighborhood_frame["candidate_id"].eq(EXPECTED_CANDIDATE_ID)
    neighborhood_chart = alt.Chart(neighborhood_frame, title="Frozen-grid pre-holdout parameter neighborhood").mark_bar().encode(
        y=alt.Y("candidate_id:N", title=None, sort=alt.SortField("return_pct", order="descending"), axis=alt.Axis(labelLimit=290)),
        x=alt.X("return_pct:Q", title="Net total return (%)"),
        color=alt.condition("datum.Selected", alt.value(ACCENT), alt.value("#a8b0bd")),
        tooltip=["candidate_id:N", alt.Tooltip("return_pct:Q", format="+.2f"), alt.Tooltip("max_drawdown:Q", format=".3f"), "num_trades:Q"],
    ).properties(height=430)
    st.altair_chart(neighborhood_chart, width="stretch")

    gap = scenario_output(pre, "exclude_canonical_gap_weeks")["metrics"]
    target_first = scenario_output(research.holdout_robustness, "target_first")["metrics"]
    notes = st.columns(3)
    with notes[0]: metric_card("Gap-week exclusion", pct(gap["total_return"]), "Pre-holdout; baseline was −4.18%", "neutral")
    with notes[1]: metric_card("Intrabar target-first", pct(target_first["total_return"]), "Identical holdout result", "neutral")
    with notes[2]: metric_card("Leverage cap hits", pct(holdout_metrics["fraction_of_entries_capped_by_leverage"], 1, sign=False), "Maximum used: 0.534×", "neutral")
    st.caption("The pre-holdout result changed sign over time: positive in 2021–2022, negative from 2023 onward. Both directions contributed positively in the final holdout, but pre-holdout long-only and short-only diagnostics were negative.")
    provenance_expander()

elif page == "Explore":
    page_intro(
        "Sandbox",
        "Explore the frozen strategy family",
        "This page runs a single in-memory backtest on canonical pre-holdout data. It is exploratory, is not Protocol V1 evidence, and cannot write to or replace sealed research artifacts.",
    )
    st.warning("Exploratory — settings and outputs below are outside the canonical Protocol V1 result.", icon="⚠️")

    with st.form("explore_form"):
        dates = st.columns(2)
        with dates[0]:
            start_date = st.date_input("Start (Monday, UTC)", value=pd.Timestamp("2024-01-01").date(), min_value=pd.Timestamp("2021-05-31").date(), max_value=(HOLDOUT_START - pd.Timedelta(weeks=1)).date())
        with dates[1]:
            end_date = st.date_input("End, exclusive (Monday, UTC)", value=HOLDOUT_START.date(), min_value=pd.Timestamp("2021-06-07").date(), max_value=HOLDOUT_START.date())

        c1, c2, c3 = st.columns(3)
        with c1:
            stop_offset = st.selectbox("Stop offset", [0.75, 1.0, 1.25], index=2, format_func=lambda value: f"{value:.2f}R")
            latest_day_label = st.selectbox("Latest entry day", ["Wednesday", "Thursday"], index=0)
        with c2:
            target_label = st.selectbox("Target plan", ["Full exit at midpoint", "Half midpoint / half opposite", "Full exit at opposite boundary"], index=0)
            direction_label = st.selectbox("Direction", ["Long + synthetic short", "Long only", "Synthetic short only"], index=0)
        with c3:
            fee_bps = st.number_input("Fee per fill (bps)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
            slippage_bps = st.number_input("Adverse slippage per fill (bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
        intrabar_label = st.selectbox("Intrabar ambiguity", ["Conservative stop-first", "Target-first sensitivity"], index=0)
        submitted = st.form_submit_button("Run exploratory backtest", type="primary", width="stretch")

    if submitted:
        st.session_state.pop("explore_result", None)
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")
        if start.weekday() != 0 or end.weekday() != 0:
            st.error("Both boundaries must be Mondays so the evaluation contains complete ISO weeks.")
        elif end <= start:
            st.error("The end boundary must be after the start boundary.")
        else:
            targets = {
                "Full exit at midpoint": "full_at_midpoint",
                "Half midpoint / half opposite": "midpoint_half_then_opposite",
                "Full exit at opposite boundary": "full_at_opposite_boundary",
            }
            directions = {
                "Long + synthetic short": "both",
                "Long only": "long_only",
                "Synthetic short only": "short_only",
            }
            try:
                with st.spinner("Running one in-memory backtest on canonical 1h data…"):
                    st.session_state["explore_result"] = _run_explore_cached(
                        start.isoformat(),
                        end.isoformat(),
                        float(stop_offset),
                        2 if latest_day_label == "Wednesday" else 3,
                        targets[target_label],
                        float(fee_bps),
                        float(slippage_bps),
                        "conservative_stop_first" if intrabar_label.startswith("Conservative") else "target_first",
                        directions[direction_label],
                    )
            except Exception as exc:
                st.error(f"Exploratory backtest could not run: {exc}")

    result = st.session_state.get("explore_result")
    if result:
        metrics = result["metrics"]
        st.divider()
        section("Exploratory result", "Generated in memory for this app session. It does not alter the canonical conclusion or sealed bundle.")
        cards = st.columns(4)
        with cards[0]: metric_card("Net return", pct(metrics["total_return"]), "Exploratory window", "neutral")
        with cards[1]: metric_card("Max drawdown", pct(metrics["max_drawdown"]), "Marked hourly equity", "neutral")
        with cards[2]: metric_card("Trades", str(metrics["num_trades"]), "Completed positions", "neutral")
        with cards[3]: metric_card("Profit factor", num(metrics.get("profit_factor"), 3), "After selected costs", "neutral")
        curve = result["equity"].reset_index().rename(columns={result["equity"].index.name or "index": "timestamp"})
        if "timestamp" not in curve.columns:
            curve = curve.rename(columns={curve.columns[0]: "timestamp"})
        curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
        daily = curve.set_index("timestamp")["equity"].resample("1D").last().dropna().reset_index()
        chart = alt.Chart(daily, title="Exploratory marked-equity curve").mark_line(color=MUTED, strokeWidth=2).encode(
            x=alt.X("timestamp:T", title=None), y=alt.Y("equity:Q", title="Equity", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("timestamp:T"), alt.Tooltip("equity:Q", format=",.2f")],
        ).properties(height=280)
        st.altair_chart(chart, width="stretch")
        trades = result["trades"]
        if len(trades):
            columns = [column for column in ("entry_time", "exit_time", "side", "reason", "pnl", "return_pct", "holding_hours") if column in trades]
            shown = trades[columns].copy()
            if "return_pct" in shown:
                shown["return_pct"] = shown["return_pct"].map(lambda value: pct(value, 3))
            st.dataframe(shown, hide_index=True, width="stretch")
        else:
            st.info("No completed trades occurred in this exploratory window.")
        st.caption("No files were written. Refreshing the app clears this session result; the sealed canonical bundle remains unchanged.")

    with st.expander("Canonical reference settings"):
        st.write("The controls default to the sealed candidate: 1.25R stop, Wednesday cutoff, full midpoint exit, both directions, 10 bps fee, 5 bps adverse slippage, and conservative stop-first execution.")
        st.write("Risk remains fixed at 1% of current marked equity with a 1× leverage cap. These portfolio controls are intentionally not exposed here.")
    provenance_expander()

st.markdown("---")
st.markdown(
    '<div class="footnote">Research code and results are for methodological demonstration. They are not investment advice or a claim of a production-ready trading strategy.</div>',
    unsafe_allow_html=True,
)
