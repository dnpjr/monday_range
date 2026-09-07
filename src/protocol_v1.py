from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import backtest_sweep_fade
from .canonical_data import (
    DATA_FILE,
    MANIFEST_FILE,
    load_canonical_ohlcv,
    load_manifest,
    sha256_file,
)
from .features import add_monday_range
from .metrics import equity_metrics, trade_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "research" / "monday_range_protocol_v1.json"
EXPECTED_PROTOCOL_SHA256 = "da0d41ce67d445bcb308012bf323d2e573d246a7a1fd3eb57c9ee6bc951d4cf3"
ACCOUNTING_VERSION = "marked_equity_v1"
EXPERIMENT_ROOT = REPO_ROOT / "reports" / "experiments" / "monday_range_protocol_v1"
GLOBAL_HOLDOUT_ACCESS_PATH = EXPERIMENT_ROOT / "HOLDOUT_ACCESS.json"
SYNTHETIC_SHORT_LABEL = "synthetic_research_position_on_binance_spot_price_series"


class ProtocolError(RuntimeError):
    pass


class ArtifactSealedError(ProtocolError):
    pass


class HoldoutLockedError(ProtocolError):
    pass


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ProtocolError(f"Timestamp must be timezone-aware UTC: {value!r}")
    if ts.utcoffset() != pd.Timedelta(0):
        raise ProtocolError(f"Timestamp is not UTC: {value!r}")
    ts = ts.tz_convert("UTC")
    return ts


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def load_frozen_protocol(
    path: str | Path = PROTOCOL_PATH,
    *,
    expected_sha256: str = EXPECTED_PROTOCOL_SHA256,
) -> dict[str, Any]:
    source = Path(path)
    actual_hash = sha256_file(source)
    if actual_hash != expected_sha256:
        raise ProtocolError(
            f"Frozen protocol hash mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    with source.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_version") != "monday_range_protocol_v1":
        raise ProtocolError("Unsupported protocol version")
    if protocol.get("status") != "frozen_before_corrected_results":
        raise ProtocolError("Protocol is not in the frozen pre-results state")
    dataset = protocol.get("dataset", {})
    if dataset.get("timezone") != "UTC" or dataset.get("base_interval") != "1h":
        raise ProtocolError("Protocol requires UTC 1h canonical data")
    baseline = protocol.get("baseline", {})
    if baseline.get("accounting_version") != ACCOUNTING_VERSION:
        raise ProtocolError(
            f"Accounting version must be {ACCOUNTING_VERSION!r}"
        )
    if baseline.get("strategy") != "current_monday_range":
        raise ProtocolError("Frozen strategy cannot be represented by this executor")
    required_baseline = {
        "entry_days_utc": [1, 2, 3],
        "one_trade_per_iso_week": True,
        "signal_bar": "completed_previous_bar",
        "entry_price": "next_bar_open_after_adverse_slippage",
        "risk_base": "current_equity",
        "intrabar_policy": "conservative_stop_first",
        "mark_price": "bar_close",
        "terminal_exit": "end_of_data_with_normal_exit_costs",
    }
    mismatches = {
        key: (baseline.get(key), expected)
        for key, expected in required_baseline.items()
        if baseline.get(key) != expected
    }
    if mismatches:
        raise ProtocolError(f"Frozen baseline cannot be represented exactly: {mismatches}")

    names = ("development", "validation", "holdout")
    bounds: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for name in names:
        item = protocol.get("partitions", {}).get(name, {})
        start = _utc(item.get("start_inclusive"))
        end = _utc(item.get("end_exclusive"))
        if end <= start:
            raise ProtocolError(f"{name} partition is empty or reversed")
        if start.weekday() != 0 or start.hour != 0 or start.minute != 0:
            raise ProtocolError(f"{name} start is not Monday 00:00 UTC")
        if end.weekday() != 0 or end.hour != 0 or end.minute != 0:
            raise ProtocolError(f"{name} end is not Monday 00:00 UTC")
        expected_weeks = int((end - start) / pd.Timedelta(weeks=1))
        if expected_weeks != int(item.get("complete_iso_weeks", -1)):
            raise ProtocolError(f"{name} complete-week count is inconsistent")
        bounds.append((name, start, end))
    for (_, _, left_end), (right_name, right_start, _) in zip(bounds, bounds[1:]):
        if left_end != right_start:
            raise ProtocolError(f"Partition boundary before {right_name} has a gap or overlap")

    folds = protocol.get("walk_forward", {}).get("folds", [])
    if len(folds) != 3:
        raise ProtocolError("Protocol requires exactly three walk-forward folds")
    validation_start = bounds[1][1]
    validation_end = bounds[1][2]
    cursor = validation_start
    for fold in folds:
        train_start = _utc(fold["train_start_inclusive"])
        train_end = _utc(fold["train_end_exclusive"])
        fold_start = _utc(fold["validation_start_inclusive"])
        fold_end = _utc(fold["validation_end_exclusive"])
        if train_start != bounds[0][1] or train_end != fold_start:
            raise ProtocolError(f"Fold {fold['id']} is not an expanding, leakage-free fold")
        if fold_start != cursor or fold_end - fold_start != pd.Timedelta(weeks=24):
            raise ProtocolError(f"Fold {fold['id']} validation boundary is inconsistent")
        cursor = fold_end
    if cursor != validation_end:
        raise ProtocolError("Walk-forward folds do not cover the frozen validation partition")

    candidates = generate_candidates(protocol, validate=False)
    expected_count = int(protocol.get("parameter_search", {}).get("candidate_count", -1))
    if len(candidates) != expected_count or len({x["candidate_id"] for x in candidates}) != expected_count:
        raise ProtocolError("Frozen candidate grid is inconsistent")


def verify_canonical_dataset(
    protocol: Mapping[str, Any],
    *,
    data_path: str | Path = DATA_FILE,
    manifest_path: str | Path = MANIFEST_FILE,
) -> dict[str, Any]:
    expected = str(protocol["dataset"]["sha256"])
    actual = sha256_file(data_path)
    if actual != expected:
        raise ProtocolError(f"Canonical dataset hash mismatch: expected {expected}, got {actual}")
    manifest = load_manifest(manifest_path)
    if manifest.get("sha256") != expected:
        raise ProtocolError("Canonical manifest and protocol dataset hashes differ")
    if manifest.get("dataset_version") != protocol["dataset"]["version"]:
        raise ProtocolError("Canonical manifest dataset version differs from protocol")
    return manifest


def _number_id(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def generate_candidates(
    protocol: Mapping[str, Any], *, validate: bool = True
) -> list[dict[str, Any]]:
    if validate:
        # Avoid recursion when validate_protocol verifies the grid itself.
        if protocol.get("baseline", {}).get("strategy") != "current_monday_range":
            raise ProtocolError("Candidate grid requires current_monday_range")
    search = protocol["parameter_search"]["parameters"]
    baseline = dict(protocol["baseline"])
    candidates: list[dict[str, Any]] = []
    for stop in search["stop_range_fraction"]:
        for day in search["max_entry_day_utc"]:
            for target in search["target_plan"]:
                candidate_id = (
                    f"mr1_stop{_number_id(float(stop))}_day{int(day)}_"
                    f"{target['id']}"
                )
                config = dict(baseline)
                config.update(
                    {
                        "stop_range_fraction": float(stop),
                        "max_entry_day_utc": int(day),
                        "exit_style": target["exit_style"],
                        "tp1_range_fraction": float(target["tp1_range_fraction"]),
                        "tp2_range_fraction": float(target["tp2_range_fraction"]),
                        "tp1_close_fraction": float(target["tp1_close_fraction"]),
                        "single_target_level": target["single_target_level"],
                        "single_target_mode": target["exit_style"] == "single_target",
                    }
                )
                candidates.append(
                    {"candidate_id": candidate_id, "target_plan_id": target["id"], "config": config}
                )
    expected = int(protocol["parameter_search"]["candidate_count"])
    if len(candidates) != expected or len({x["candidate_id"] for x in candidates}) != expected:
        raise ProtocolError(f"Expected {expected} unique candidates, generated {len(candidates)}")
    return candidates


def partition_bounds(protocol: Mapping[str, Any], name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if name not in {"development", "validation", "holdout"}:
        raise ProtocolError(f"Unknown partition: {name}")
    part = protocol["partitions"][name]
    return _utc(part["start_inclusive"]), _utc(part["end_exclusive"])


def require_utc_index(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ProtocolError("OHLC data must have a DatetimeIndex")
    if frame.index.tz is None:
        raise ProtocolError("OHLC index must be timezone-aware UTC")
    if str(frame.index.tz) not in {"UTC", "UTC+00:00"}:
        raise ProtocolError("OHLC index must be normalized to UTC")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ProtocolError("OHLC index must be sorted and unique")


def feature_window_with_context(
    ohlc: pd.DataFrame,
    *,
    start_inclusive: Any,
    end_exclusive: Any,
    context_weeks: int = 1,
) -> pd.DataFrame:
    require_utc_index(ohlc)
    start, end = _utc(start_inclusive), _utc(end_exclusive)
    if end <= start:
        raise ProtocolError("Evaluation window is empty or reversed")
    if start.weekday() != 0 or end.weekday() != 0:
        raise ProtocolError("Evaluation windows must be Monday-aligned")
    context_start = start - pd.Timedelta(weeks=int(context_weeks))
    context = ohlc[(ohlc.index >= context_start) & (ohlc.index < end)].copy()
    if context.empty or not bool((context.index < start).any()):
        raise ProtocolError("At least one preceding ISO week of context is required")
    expected_context = pd.date_range(
        context_start, start - pd.Timedelta(hours=1), freq="1h"
    )
    available_context = context.index[(context.index >= context_start) & (context.index < start)]
    if len(expected_context.difference(available_context)):
        raise ProtocolError("The complete required preceding context week is unavailable")
    features = add_monday_range(context)
    features["protocol_entry_eligible"] = (features.index >= start) & (features.index < end)
    features.attrs.update(
        {
            "context_start_inclusive": context_start.isoformat(),
            "evaluation_start_inclusive": start.isoformat(),
            "evaluation_end_exclusive": end.isoformat(),
        }
    )
    return features


def _engine_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "strategy", "initial_capital", "risk_fraction", "risk_base", "max_leverage",
        "tp1_range_fraction", "tp2_range_fraction", "tp1_close_fraction",
        "stop_mode", "stop_range_fraction", "move_stop_to_breakeven_after_tp1",
        "breakeven_includes_fees", "single_target_mode", "single_target_level",
        "friday_cutoff_hour_utc", "min_range_pct", "max_range_pct", "direction",
        "max_entry_day_utc", "max_entry_hour_utc", "sma_period", "intrabar_policy",
    }
    result = {key: value for key, value in config.items() if key in allowed}
    result["fee_bps"] = float(config["fee_bps_per_fill"])
    result["slippage_bps"] = float(config["slippage_bps_per_fill"])
    result["exit_friday_close"] = True
    result["close_open_position_at_end"] = True
    return result


def evaluate_candidate_window(
    ohlc: pd.DataFrame,
    *,
    candidate: Mapping[str, Any],
    start_inclusive: Any,
    end_exclusive: Any,
    bars_per_year: float = 365.0 * 24.0,
) -> dict[str, Any]:
    start, end = _utc(start_inclusive), _utc(end_exclusive)
    featured = feature_window_with_context(
        ohlc, start_inclusive=start, end_exclusive=end, context_weeks=1
    )
    kwargs = _engine_kwargs(candidate["config"])
    df_out, trades = backtest_sweep_fade(
        featured,
        **kwargs,
        entry_start_utc=start,
        entry_end_exclusive_utc=end,
    )
    if df_out.attrs.get("accounting_version") != candidate["config"]["accounting_version"]:
        raise ProtocolError("Backtest engine accounting version differs from the frozen protocol")
    if df_out.attrs.get("strategy") != "current_monday_range":
        raise ProtocolError("Backtest engine did not execute the frozen canonical strategy")
    evaluated = df_out[(df_out.index >= start) & (df_out.index < end)].copy()
    if evaluated.empty:
        raise ProtocolError("No evaluation rows remain after exact partition filtering")
    if not trades.empty:
        entries = pd.to_datetime(trades["entry_time"], utc=True)
        if bool(((entries < start) | (entries >= end)).any()):
            raise ProtocolError("Context trade leaked into the evaluation window")
    em = equity_metrics(evaluated["equity"], bars_per_year=bars_per_year)
    tm = trade_metrics(trades)
    elapsed_years = float(em.get("elapsed_years", 0.0))
    average_equity = float(evaluated["equity"].dropna().mean())
    fees = float(evaluated["fees_paid"].sum())
    slippage = float(evaluated["slippage_cost"].sum())
    turnover = float(evaluated["turnover"].sum())
    denominator = average_equity * elapsed_years
    cost_drag = (fees + slippage) / denominator if denominator > 0 else math.inf
    annualized_turnover = turnover / denominator if denominator > 0 else math.inf
    equity_denominator = evaluated["equity"].replace(0.0, np.nan)
    gross_exposure = float((evaluated["gross_notional"] / equity_denominator).fillna(0.0).mean())
    net_exposure = float((evaluated["position_value"] / equity_denominator).fillna(0.0).mean())
    leverage_series = (evaluated["gross_notional"] / equity_denominator).replace([np.inf, -np.inf], np.nan)
    max_leverage_used = float(leverage_series.max()) if leverage_series.notna().any() else 0.0
    trade_returns = pd.to_numeric(trades.get("return_pct", pd.Series(dtype=float)), errors="coerce").dropna()
    trade_pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    holding = pd.to_numeric(trades.get("holding_hours", pd.Series(dtype=float)), errors="coerce").dropna()
    capped_fraction = (
        float((trades["sizing_limited_by"] == "max_leverage").mean())
        if len(trades) and "sizing_limited_by" in trades.columns else 0.0
    )
    long_short = {}
    if len(trades) and "side" in trades.columns:
        for side, group in trades.groupby("side"):
            long_short[str(side)] = {
                "trade_count": int(len(group)),
                "net_pnl": float(pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0).sum()),
            }
    profit_factor = tm.get("profit_factor")
    if profit_factor is not None and not math.isfinite(float(profit_factor)):
        profit_factor = None
    metrics = {
        "candidate_id": candidate["candidate_id"],
        "start_inclusive": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "context_start_inclusive": featured.attrs["context_start_inclusive"],
        "context_row_count": int((featured.index < start).sum()),
        "evaluation_row_count": int(len(evaluated)),
        "num_trades": int(len(trades)),
        "net_cagr": float(em.get("cagr", 0.0)),
        "total_return": float(em.get("total_return", 0.0)),
        "max_drawdown": float(em.get("max_drawdown", 0.0)),
        "annualized_volatility": float(em.get("annualized_volatility", 0.0)),
        "sharpe": float(em.get("sharpe", 0.0)),
        "sortino": float(em.get("sortino", 0.0)),
        "profit_factor": profit_factor,
        "win_rate": tm.get("win_rate"),
        "mean_net_trade_return": float(trade_returns.mean()) if len(trade_returns) else None,
        "median_net_trade_return": float(trade_returns.median()) if len(trade_returns) else None,
        "mean_pnl": float(trade_pnl.mean()) if len(trade_pnl) else None,
        "median_pnl": float(trade_pnl.median()) if len(trade_pnl) else None,
        "mean_holding_hours": float(holding.mean()) if len(holding) else None,
        "median_holding_hours": float(holding.median()) if len(holding) else None,
        "exit_reason_distribution": tm.get("by_reason", {}),
        "long_short_decomposition": long_short,
        "average_marked_equity": average_equity,
        "fees": fees,
        "entry_slippage_cost": float(trades.get("entry_slippage_cost", pd.Series(dtype=float)).sum()),
        "exit_slippage_cost": float(trades.get("exit_slippage_cost", pd.Series(dtype=float)).sum()),
        "total_slippage_cost": slippage,
        "combined_execution_cost": fees + slippage,
        "annualized_cost_drag_fraction": float(cost_drag),
        "annualized_turnover_multiple": float(annualized_turnover),
        "total_traded_notional": turnover,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "time_exposed_fraction": float(evaluated["exposed"].mean()),
        "maximum_leverage_used": max_leverage_used,
        "fraction_of_entries_capped_by_leverage": capped_fraction,
        "short_position_interpretation": SYNTHETIC_SHORT_LABEL,
    }
    metrics["score"] = selection_score(metrics)
    return {
        "metrics": metrics,
        "equity": evaluated,
        "trades": trades,
        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "target_plan_id": candidate.get("target_plan_id"),
            "configuration": dict(candidate["config"]),
        },
    }


def selection_score(metrics: Mapping[str, Any]) -> float:
    return float(
        float(metrics["net_cagr"])
        - 0.50 * abs(float(metrics["max_drawdown"]))
        - float(metrics["annualized_cost_drag_fraction"])
        - 0.0005 * float(metrics["annualized_turnover_multiple"])
    )


def select_training_winner(
    evaluations: Sequence[Mapping[str, Any]], *, minimum_trades: int = 30
) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    for item in evaluations:
        row = dict(item)
        if row.get("scope", "training") != "training":
            continue
        if int(row.get("num_trades", 0)) < int(minimum_trades):
            continue
        row["score"] = selection_score(row)
        eligible.append(row)
    if not eligible:
        raise ProtocolError(f"No training candidate met the {minimum_trades}-trade minimum")

    def finite(value: Any, *, low: float = -math.inf) -> float:
        number = float(value)
        return number if math.isfinite(number) else low

    eligible.sort(
        key=lambda row: (
            -finite(row["score"]),
            -finite(row.get("sortino", 0.0)),
            abs(float(row["max_drawdown"])),
            float(row["annualized_turnover_multiple"]),
            str(row["candidate_id"]),
        )
    )
    return eligible[0]


class ExperimentBundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @classmethod
    def create(cls, root: str | Path) -> "ExperimentBundle":
        path = Path(root)
        path.mkdir(parents=True, exist_ok=False)
        for name in (
            "development", "fold_1", "fold_2", "fold_3", "final_fit", "holdout",
            "robustness", "uncertainty", "charts",
        ):
            (path / name).mkdir()
        return cls(path)

    def seal_json(
        self, relative_path: str | Path, payload: Mapping[str, Any], *, embed_hash: bool = False
    ) -> str:
        target = self.root / relative_path
        sidecar = target.with_suffix(target.suffix + ".sha256")
        if target.exists() or sidecar.exists():
            raise ArtifactSealedError(f"Refusing to overwrite sealed artifact: {target}")
        body = dict(payload)
        digest = payload_sha256(body)
        if embed_hash:
            body["artifact_hash"] = digest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        sidecar.write_text(digest + "\n", encoding="ascii")
        return digest

    def verify_json(self, relative_path: str | Path) -> dict[str, Any]:
        target = self.root / relative_path
        sidecar = target.with_suffix(target.suffix + ".sha256")
        if not target.exists() or not sidecar.exists():
            raise ProtocolError(f"Required sealed artifact is missing: {target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        embedded = payload.pop("artifact_hash", None)
        actual = payload_sha256(payload)
        expected = sidecar.read_text(encoding="ascii").strip()
        if actual != expected or (embedded is not None and embedded != actual):
            raise ProtocolError(f"Sealed artifact hash mismatch: {target}")
        if embedded is not None:
            payload["artifact_hash"] = embedded
        return payload


def collect_code_provenance(
    *, repo_root: str | Path = REPO_ROOT, require_clean: bool = True
) -> dict[str, Any]:
    root = Path(repo_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    dirty = bool(status.strip())
    if require_clean and dirty:
        raise ProtocolError("Canonical protocol execution requires a clean Git working tree")
    return {
        "git_commit": commit,
        "working_tree": "dirty" if dirty else "clean",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "accounting_version": ACCOUNTING_VERSION,
        "mark_price": "bar_close",
        "gap_fill_policy": "bar_open_when_opened_through_trigger_then_apply_adverse_slippage",
        "intrabar_policy": "conservative_stop_first",
        "short_position_interpretation": SYNTHETIC_SHORT_LABEL,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_full_test_suite(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=Path(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    if result.returncode != 0 or not match or "OK" not in combined:
        raise ProtocolError("The full test suite did not pass during bundle initialization")
    count = int(match.group(1))
    return {
        "status": "passed",
        "tests_run": count,
        "record": f"{count}/{count} passed",
        "command": f"{sys.executable} -m unittest discover -s tests -q",
    }


Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any], str, int], Mapping[str, Any]]


def rank_training_candidates(
    evaluations: Sequence[Mapping[str, Any]], *, minimum_trades: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evaluations:
        row = dict(item)
        row["score"] = selection_score(row)
        row["eligible"] = int(row.get("num_trades", 0)) >= int(minimum_trades)
        rows.append(row)

    def finite(value: Any, *, low: float = -math.inf) -> float:
        number = float(value)
        return number if math.isfinite(number) else low

    rows.sort(
        key=lambda row: (
            not bool(row["eligible"]),
            -finite(row["score"]),
            -finite(row.get("sortino", 0.0)),
            abs(float(row["max_drawdown"])),
            float(row["annualized_turnover_multiple"]),
            str(row["candidate_id"]),
        )
    )
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows


def _split_evaluator_result(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if isinstance(raw.get("metrics"), Mapping):
        metrics = dict(raw["metrics"])
        artifact = raw.get("artifact")
        return metrics, (dict(artifact) if isinstance(artifact, Mapping) else None)
    return dict(raw), None


def run_frozen_walk_forward(
    protocol: Mapping[str, Any],
    *,
    evaluator: Evaluator,
    bundle: ExperimentBundle,
) -> list[dict[str, Any]]:
    candidates = generate_candidates(protocol)
    minimum = int(protocol["walk_forward"]["minimum_training_trades"])
    results: list[dict[str, Any]] = []
    for fold in protocol["walk_forward"]["folds"]:
        fold_id = int(fold["id"])
        training: list[dict[str, Any]] = []
        for candidate in candidates:
            row, _ = _split_evaluator_result(
                evaluator(candidate, fold, "training", fold_id)
            )
            row.update({"candidate_id": candidate["candidate_id"], "scope": "training"})
            training.append(row)
        training = rank_training_candidates(training, minimum_trades=minimum)
        winner = select_training_winner(training, minimum_trades=minimum)
        selected = next(x for x in candidates if x["candidate_id"] == winner["candidate_id"])
        selection_path = f"fold_{fold_id}/selection.json"
        bundle.seal_json(
            selection_path,
            {
                "fold_id": fold_id,
                "training_window": {
                    "start_inclusive": fold["train_start_inclusive"],
                    "end_exclusive": fold["train_end_exclusive"],
                },
                "validation_window": {
                    "start_inclusive": fold["validation_start_inclusive"],
                    "end_exclusive": fold["validation_end_exclusive"],
                },
                "minimum_training_trades": minimum,
                "training_scores": training,
                "selected_candidate_id": winner["candidate_id"],
                "selected_candidate": selected,
            },
            embed_hash=True,
        )
        bundle.verify_json(selection_path)
        validation, validation_artifact = _split_evaluator_result(
            evaluator(selected, fold, "validation", fold_id)
        )
        validation.update({"candidate_id": selected["candidate_id"], "scope": "validation"})
        if validation_artifact is None:
            validation_artifact = dict(validation)
        else:
            validation_artifact["metrics"] = validation
        validation_path = f"fold_{fold_id}/validation.json"
        bundle.seal_json(
            validation_path,
            {"fold_id": fold_id, "result": validation_artifact},
            embed_hash=True,
        )
        results.append({"fold_id": fold_id, "selected_candidate_id": selected["candidate_id"], "validation": validation})
    return results


def persist_final_candidate(
    protocol: Mapping[str, Any],
    *,
    evaluations: Sequence[Mapping[str, Any]],
    bundle: ExperimentBundle,
    code_commit: str,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    winner = select_training_winner(
        evaluations, minimum_trades=int(protocol["walk_forward"]["minimum_training_trades"])
    )
    candidate = next(x for x in generate_candidates(protocol) if x["candidate_id"] == winner["candidate_id"])
    artifact = {
        "candidate_id": candidate["candidate_id"],
        "configuration": candidate["config"],
        "training_period": {
            "start_inclusive": protocol["partitions"]["development"]["start_inclusive"],
            "end_exclusive": protocol["partitions"]["holdout"]["start_inclusive"],
        },
        "objective_value": float(winner["score"]),
        "tie_break_values": {
            "sortino": float(winner.get("sortino", 0.0)),
            "absolute_max_drawdown": abs(float(winner["max_drawdown"])),
            "annualized_turnover_multiple": float(winner["annualized_turnover_multiple"]),
        },
        "protocol_hash": EXPECTED_PROTOCOL_SHA256,
        "dataset_hash": protocol["dataset"]["sha256"],
        "code_commit": code_commit,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "short_position_interpretation": SYNTHETIC_SHORT_LABEL,
    }
    bundle.seal_json("final_fit/final_candidate.json", artifact, embed_hash=True)
    return bundle.verify_json("final_fit/final_candidate.json")


def assert_holdout_ready(
    protocol: Mapping[str, Any],
    *,
    bundle: ExperimentBundle,
    confirm_holdout: bool,
) -> dict[str, Any]:
    if not confirm_holdout:
        raise HoldoutLockedError("Holdout requires the explicit --confirm-holdout action")
    protocol_artifact = bundle.verify_json("protocol.json")
    if protocol_artifact.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise HoldoutLockedError("Bundle protocol hash is not the frozen protocol hash")
    dataset_artifact = bundle.verify_json("dataset_manifest.json")
    if dataset_artifact.get("sha256") != protocol["dataset"]["sha256"]:
        raise HoldoutLockedError("Bundle dataset hash is not the frozen dataset hash")
    provenance = bundle.verify_json("code_provenance.json")
    if (
        not provenance.get("git_commit")
        or not provenance.get("test_suite_record")
        or provenance.get("test_suite_verified") is not True
        or provenance.get("canonical_execution_allowed") is not True
    ):
        raise HoldoutLockedError("Code commit and passing test record are required")
    grid_artifact = bundle.verify_json("candidate_grid.json")
    if grid_artifact != {"candidate_count": 18, "candidates": generate_candidates(protocol)}:
        raise HoldoutLockedError("Sealed candidate grid differs from the frozen 18 candidates")
    development = bundle.verify_json("development/candidate_ranking.json")
    bundle.verify_json("development/baseline.json")
    grid = {item["candidate_id"]: item for item in generate_candidates(protocol)}
    development_rows = development.get("candidate_ranking", [])
    dev_start = _utc(protocol["partitions"]["development"]["start_inclusive"])
    dev_end = _utc(protocol["partitions"]["development"]["end_exclusive"])
    if (
        int(development.get("candidate_count", -1)) != 18
        or len(development_rows) != 18
        or {row.get("candidate_id") for row in development_rows} != set(grid)
        or [row.get("rank") for row in development_rows] != list(range(1, 19))
    ):
        raise HoldoutLockedError("The complete development stage is not sealed")
    for row in development_rows:
        if (
            _utc(row.get("start_inclusive")) != dev_start
            or _utc(row.get("end_exclusive")) != dev_end
            or _utc(row.get("context_start_inclusive")) != dev_start - pd.Timedelta(weeks=1)
            or int(row.get("context_row_count", -1)) != 7 * 24
        ):
            raise HoldoutLockedError("A development score used the wrong interval or context")
    audit = bundle.verify_json("walk_forward_audit.json")
    if audit.get("status") != "passed" or audit.get("holdout_accessed") is not False:
        raise HoldoutLockedError("Walk-forward audit is missing or did not pass before holdout")
    for fold_id in (1, 2, 3):
        fold = protocol["walk_forward"]["folds"][fold_id - 1]
        selection = bundle.verify_json(f"fold_{fold_id}/selection.json")
        validation = bundle.verify_json(f"fold_{fold_id}/validation.json")
        if selection.get("fold_id") != fold_id or validation.get("fold_id") != fold_id:
            raise HoldoutLockedError(f"Fold {fold_id} artifact identity is inconsistent")
        selected_id = selection.get("selected_candidate_id")
        scores = selection.get("training_scores", [])
        if (
            len(scores) != 18
            or {row.get("candidate_id") for row in scores} != set(grid)
            or [row.get("rank") for row in scores] != list(range(1, 19))
        ):
            raise HoldoutLockedError(f"Fold {fold_id} training ranking is incomplete")
        fold_train_start = _utc(fold["train_start_inclusive"])
        fold_train_end = _utc(fold["train_end_exclusive"])
        for row in scores:
            if (
                row.get("scope") != "training"
                or _utc(row.get("start_inclusive")) != fold_train_start
                or _utc(row.get("end_exclusive")) != fold_train_end
                or _utc(row.get("context_start_inclusive")) != fold_train_start - pd.Timedelta(weeks=1)
                or int(row.get("context_row_count", -1)) != 7 * 24
            ):
                raise HoldoutLockedError(f"Fold {fold_id} training score used the wrong scope, interval, or context")
        fold_winner = select_training_winner(
            scores,
            minimum_trades=int(protocol["walk_forward"]["minimum_training_trades"]),
        )
        result = validation.get("result", {})
        validated_id = result.get("metrics", {}).get("candidate_id", result.get("candidate_id"))
        if not selected_id or selected_id != validated_id or selected_id != fold_winner["candidate_id"]:
            raise HoldoutLockedError(f"Fold {fold_id} validation does not match its sealed training winner")
    aggregate = bundle.verify_json("walk_forward_aggregate.json")
    expected_validation_weeks = int(protocol["partitions"]["validation"]["complete_iso_weeks"])
    if int(aggregate.get("metrics", {}).get("complete_weeks", -1)) != expected_validation_weeks:
        raise HoldoutLockedError("Aggregate walk-forward validation coverage is incomplete")
    aggregate_weeks = [_utc(row["week_start_utc"]) for row in aggregate.get("weekly_returns", [])]
    validation_start = _utc(protocol["partitions"]["validation"]["start_inclusive"])
    validation_end = _utc(protocol["partitions"]["validation"]["end_exclusive"])
    expected_aggregate_weeks = list(
        pd.date_range(validation_start, validation_end - pd.Timedelta(weeks=1), freq="7D")
    )
    if aggregate_weeks != expected_aggregate_weeks:
        raise HoldoutLockedError("Aggregate walk-forward weeks are incomplete, duplicated, or unordered")
    candidate = bundle.verify_json("final_fit/final_candidate.json")
    final_scores = bundle.verify_json("final_fit/training_scores.json").get(
        "training_scores", []
    )
    pre_start = _utc(protocol["partitions"]["development"]["start_inclusive"])
    pre_end = _utc(protocol["partitions"]["holdout"]["start_inclusive"])
    if (
        len(final_scores) != 18
        or {row.get("candidate_id") for row in final_scores} != set(grid)
        or [row.get("rank") for row in final_scores] != list(range(1, 19))
    ):
        raise HoldoutLockedError("Final pre-holdout ranking is not the complete frozen grid")
    for row in final_scores:
        if (
            row.get("scope") != "training"
            or _utc(row.get("start_inclusive")) != pre_start
            or _utc(row.get("end_exclusive")) != pre_end
            or _utc(row.get("context_start_inclusive")) != pre_start - pd.Timedelta(weeks=1)
            or int(row.get("context_row_count", -1)) != 7 * 24
        ):
            raise HoldoutLockedError("A final-fit score used the wrong scope, interval, or context")
    selected_from_scores = select_training_winner(
        final_scores,
        minimum_trades=int(protocol["walk_forward"]["minimum_training_trades"]),
    )
    if selected_from_scores["candidate_id"] != candidate.get("candidate_id"):
        raise HoldoutLockedError("Final candidate does not match the sealed pre-holdout winner")
    if not math.isclose(
        float(selected_from_scores["score"]),
        float(candidate.get("objective_value", math.nan)),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise HoldoutLockedError("Final candidate objective differs from sealed training scores")
    grid_candidate = next(
        (
            item
            for item in generate_candidates(protocol)
            if item["candidate_id"] == candidate.get("candidate_id")
        ),
        None,
    )
    if grid_candidate is None or grid_candidate["config"] != candidate.get("configuration"):
        raise HoldoutLockedError("Final candidate is not an exact member of the frozen grid")
    training_period = candidate.get("training_period", {})
    if training_period.get("start_inclusive") != protocol["partitions"]["development"]["start_inclusive"]:
        raise HoldoutLockedError("Final candidate training start differs from the frozen development start")
    if training_period.get("end_exclusive") != protocol["partitions"]["holdout"]["start_inclusive"]:
        raise HoldoutLockedError("Final candidate was not selected strictly before holdout")
    if candidate.get("protocol_hash") != EXPECTED_PROTOCOL_SHA256:
        raise HoldoutLockedError("Final candidate protocol hash differs from the frozen protocol")
    if candidate.get("dataset_hash") != protocol["dataset"]["sha256"]:
        raise HoldoutLockedError("Final candidate dataset hash differs from the frozen dataset")
    if candidate.get("code_commit") != provenance["git_commit"]:
        raise HoldoutLockedError("Final candidate code commit differs from sealed provenance")
    robustness = bundle.verify_json("robustness/results.json")
    required_scenarios = {
        scenario["id"] for scenario in robustness_plan(protocol, grid_candidate)
    }
    observed_scenarios = {
        row.get("scenario", {}).get("id") for row in robustness.get("outputs", [])
    }
    robustness_audit = audit_preholdout_robustness(
        protocol,
        artifact=robustness,
        final_candidate=candidate,
        missing_timestamps=dataset_artifact.get("missing_timestamps", []),
    )
    if (
        robustness.get("selected_candidate_id") != candidate.get("candidate_id")
        or robustness.get("selection_changed") is not False
        or _utc(robustness.get("start_inclusive")) != pre_start
        or _utc(robustness.get("end_exclusive")) != pre_end
        or len(robustness.get("gap_weeks", [])) != 3
        or observed_scenarios != required_scenarios
        or robustness.get("audit") != robustness_audit
    ):
        raise HoldoutLockedError("Pre-holdout robustness is missing or changed selection")
    uncertainty = bundle.verify_json("uncertainty/weekly_bootstrap.json")
    settings = protocol["uncertainty"]
    expected_uncertainty = bootstrap_from_evaluation_artifact(
        aggregate, settings=settings
    )
    expected_uncertainty.update(
        {
            "source_stage": "aggregate_walk_forward_validation",
            "source_artifact": "walk_forward_aggregate.json",
            "confirmatory_null_test": False,
        }
    )
    observed_uncertainty = dict(uncertainty)
    observed_uncertainty.pop("artifact_hash", None)
    if observed_uncertainty != expected_uncertainty:
        raise HoldoutLockedError("Pre-holdout uncertainty does not match the frozen method")
    return candidate


def authorize_holdout(
    protocol: Mapping[str, Any],
    *,
    bundle: ExperimentBundle,
    confirm_holdout: bool,
    global_access_path: str | Path | None = None,
) -> dict[str, Any]:
    candidate = assert_holdout_ready(protocol, bundle=bundle, confirm_holdout=confirm_holdout)
    access_path = Path(global_access_path) if global_access_path is not None else GLOBAL_HOLDOUT_ACCESS_PATH
    local_path = bundle.root / "holdout" / "ACCESS.json"
    local_sidecar = local_path.with_suffix(local_path.suffix + ".sha256")
    global_sidecar = access_path.with_suffix(access_path.suffix + ".sha256")
    if local_path.exists() or local_sidecar.exists():
        raise HoldoutLockedError("This experiment bundle has already accessed the holdout")
    if access_path.exists() or global_sidecar.exists():
        raise HoldoutLockedError("Protocol V1 holdout has already been accessed by another bundle")
    marker = {
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": EXPECTED_PROTOCOL_SHA256,
        "dataset_hash": protocol["dataset"]["sha256"],
        "run_id": bundle.root.name,
        "candidate_id": candidate["candidate_id"],
        "candidate_artifact_hash": candidate["artifact_hash"],
        "allowed_holdout_runs": ["fixed_baseline", "single_final_selected_candidate"],
    }
    digest = payload_sha256(marker)
    global_body = dict(marker, artifact_hash=digest)
    access_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(access_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise HoldoutLockedError("Protocol V1 holdout has already been accessed") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(global_body, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        global_sidecar.write_text(digest + "\n", encoding="ascii")
    except Exception:
        # The access marker deliberately remains in place after a partial failure: once
        # authorization is reserved, retrying would violate the one-use policy.
        raise
    bundle.seal_json("holdout/ACCESS.json", marker, embed_hash=True)
    return bundle.verify_json("holdout/ACCESS.json")


def moving_block_bootstrap(
    weekly_returns: Sequence[float] | pd.Series,
    *,
    weekly_trade_returns: Sequence[Sequence[float]] | None = None,
    block_length_weeks: int = 4,
    replications: int = 10_000,
    seed: int = 20260905,
) -> dict[str, Any]:
    values = np.asarray(weekly_returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ProtocolError("Bootstrap weekly returns must be a finite one-dimensional sequence")
    n = len(values)
    block = int(block_length_weeks)
    reps = int(replications)
    if n < block or block < 1 or reps < 1:
        raise ProtocolError("Bootstrap requires n >= block length and positive replications")
    if weekly_trade_returns is not None and len(weekly_trade_returns) != n:
        raise ProtocolError("Weekly trade-return groups must align with weekly returns")
    starts = np.arange(0, n - block + 1)
    draws_per_rep = int(math.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    means = np.empty(reps, dtype=float)
    annualized = np.empty(reps, dtype=float)
    max_drawdowns = np.empty(reps, dtype=float)
    average_trades = np.full(reps, np.nan, dtype=float)
    for i in range(reps):
        chosen = rng.choice(starts, size=draws_per_rep, replace=True)
        indices = np.concatenate(
            [np.arange(j, j + block, dtype=int) for j in chosen]
        )[:n]
        sample = values[indices]
        means[i] = float(sample.mean())
        compounded = np.cumprod(1.0 + sample)
        annualized[i] = (
            float(compounded[-1] ** (52.0 / n) - 1.0)
            if compounded[-1] > 0 else -1.0
        )
        peaks = np.maximum.accumulate(np.concatenate(([1.0], compounded)))
        curve = np.concatenate(([1.0], compounded))
        max_drawdowns[i] = float(np.min(curve / peaks - 1.0))
        if weekly_trade_returns is not None:
            sampled_trades = [
                float(value)
                for index in indices
                for value in weekly_trade_returns[int(index)]
                if math.isfinite(float(value))
            ]
            if sampled_trades:
                average_trades[i] = float(np.mean(sampled_trades))
    low, median, high = np.percentile(means, [2.5, 50.0, 97.5])
    compounded = np.cumprod(1.0 + values)
    point_annualized = float(compounded[-1] ** (52.0 / n) - 1.0) if compounded[-1] > 0 else -1.0
    point_curve = np.concatenate(([1.0], compounded))
    point_peaks = np.maximum.accumulate(point_curve)
    point_drawdown = float(np.min(point_curve / point_peaks - 1.0))
    all_trade_returns = (
        [float(value) for group in weekly_trade_returns for value in group]
        if weekly_trade_returns is not None else []
    )

    def interval(samples: np.ndarray, point: float | None) -> dict[str, float | None]:
        finite = samples[np.isfinite(samples)]
        if point is None or not len(finite):
            return {"point_estimate": point, "percentile_2_5": None, "percentile_50": None, "percentile_97_5": None}
        p2, p50, p97 = np.percentile(finite, [2.5, 50.0, 97.5])
        return {
            "point_estimate": float(point),
            "percentile_2_5": float(p2),
            "percentile_50": float(p50),
            "percentile_97_5": float(p97),
        }

    mean_interval = interval(means, float(values.mean()))
    return {
        "method": "moving_block_bootstrap",
        "unit": "complete_utc_iso_week",
        "block_length_weeks": block,
        "replications": reps,
        "seed": int(seed),
        "point_estimate": mean_interval["point_estimate"],
        "percentile_2_5": float(low),
        "percentile_50": float(median),
        "percentile_97_5": float(high),
        "reject_null": bool(low > 0.0),
        "statistics": {
            "mean_weekly_net_return": mean_interval,
            "annualized_net_return": interval(annualized, point_annualized),
            "maximum_drawdown": interval(max_drawdowns, point_drawdown),
            "average_trade_net_return": interval(
                average_trades,
                float(np.mean(all_trade_returns)) if all_trade_returns else None,
            ),
        },
    }


def bootstrap_from_evaluation_artifact(
    artifact: Mapping[str, Any], *, settings: Mapping[str, Any]
) -> dict[str, Any]:
    weekly = list(artifact.get("weekly_returns", []))
    if not weekly:
        raise ProtocolError("Evaluation artifact has no complete weekly returns")
    week_starts = [_utc(row["week_start_utc"]) for row in weekly]
    if week_starts != sorted(week_starts) or len(week_starts) != len(set(week_starts)):
        raise ProtocolError("Bootstrap weekly units must be chronological and unique")
    returns = [float(row["net_return"]) for row in weekly]
    grouped_trades: dict[pd.Timestamp, list[float]] = {week: [] for week in week_starts}
    for trade in artifact.get("trades", []):
        value = trade.get("return_pct")
        if value is None:
            continue
        entry = _utc(trade["entry_time"])
        week = (entry - pd.Timedelta(days=entry.weekday())).normalize()
        if week in grouped_trades:
            grouped_trades[week].append(float(value))
    result = moving_block_bootstrap(
        returns,
        weekly_trade_returns=[grouped_trades[week] for week in week_starts],
        block_length_weeks=int(settings["block_length_weeks"]),
        replications=int(settings["replications"]),
        seed=int(settings["random_seed"]),
    )
    result["sample_weeks"] = len(returns)
    return result


def excluded_gap_weeks(missing_timestamps: Iterable[Any]) -> list[str]:
    weeks: set[pd.Timestamp] = set()
    for value in missing_timestamps:
        ts = _utc(value)
        weeks.add((ts - pd.Timedelta(days=ts.weekday())).normalize())
    return [ts.isoformat() for ts in sorted(weeks)]


def exclude_gap_week_rows(frame: pd.DataFrame, missing_timestamps: Iterable[Any]) -> pd.DataFrame:
    require_utc_index(frame)
    excluded = set(pd.Timestamp(x) for x in excluded_gap_weeks(missing_timestamps))
    week_starts = (frame.index - pd.to_timedelta(frame.index.weekday, unit="D")).normalize()
    return frame.loc[~week_starts.isin(excluded)].copy()


def robustness_plan(
    protocol: Mapping[str, Any], selected_candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidate_id = selected_candidate["candidate_id"]
    return [
        {"id": "baseline_costs", "candidate_id": candidate_id, "overrides": protocol["cost_scenarios"]["baseline"]},
        {"id": "costs_1_5x", "candidate_id": candidate_id, "overrides": protocol["cost_scenarios"]["stress_1_5x"]},
        {"id": "costs_2x", "candidate_id": candidate_id, "overrides": protocol["cost_scenarios"]["stress_2x"]},
        {"id": "conservative_stop_first", "candidate_id": candidate_id, "overrides": {"intrabar_policy": "conservative_stop_first"}},
        {"id": "target_first", "candidate_id": candidate_id, "overrides": {"intrabar_policy": "target_first"}},
        {"id": "long_only", "candidate_id": candidate_id, "overrides": {"direction": "long_only"}},
        {"id": "short_only", "candidate_id": candidate_id, "overrides": {"direction": "short_only"}},
        {"id": "calendar_year_subperiods", "candidate_id": candidate_id, "reporting_only": True},
        {"id": "non_overlapping_24_week_subperiods", "candidate_id": candidate_id, "reporting_only": True},
        {"id": "parameter_neighborhood", "candidate_id": candidate_id, "reporting_only": True},
        {"id": "exclude_canonical_gap_weeks", "candidate_id": candidate_id, "reporting_only": True},
    ]


def run_robustness_diagnostics(
    protocol: Mapping[str, Any],
    *,
    selected_candidate: Mapping[str, Any],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    selected_id = selected_candidate["candidate_id"]
    outputs = []
    for scenario in robustness_plan(protocol, selected_candidate):
        result = dict(evaluator(scenario))
        outputs.append({"scenario": scenario, "result": result})
    return {
        "selected_candidate_id": selected_id,
        "selection_changed": False,
        "diagnostics_only": True,
        "outputs": outputs,
    }


def weekly_returns_from_equity(equity: pd.Series) -> pd.Series:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if not isinstance(values.index, pd.DatetimeIndex) or values.index.tz is None:
        raise ProtocolError("Weekly returns require timezone-aware marked equity")
    frame = values.to_frame("equity")
    frame["timestamp"] = frame.index
    grouped = frame.resample("W-MON", label="left", closed="left").agg(
        first=("equity", "first"),
        last=("equity", "last"),
        first_timestamp=("timestamp", "first"),
        last_timestamp=("timestamp", "last"),
    )
    expected_start = grouped.index
    expected_last = expected_start + pd.Timedelta(days=6, hours=23)
    complete = grouped[
        (grouped["first_timestamp"] == expected_start)
        & (grouped["last_timestamp"] == expected_last)
    ]
    returns = complete["last"] / complete["first"] - 1.0
    returns.name = "weekly_net_return"
    return returns


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None or bool(pd.isna(value)):
        return None
    return value


def evaluation_artifact(
    result: Mapping[str, Any], *, include_equity_curve: bool = True
) -> dict[str, Any]:
    equity = result["equity"]
    trades = result["trades"]
    weekly = weekly_returns_from_equity(equity["equity"])
    last = equity.iloc[-1]
    equity_denominator = equity["equity"].replace(0.0, np.nan)
    leverage = (equity["gross_notional"] / equity_denominator).replace(
        [np.inf, -np.inf], np.nan
    )
    artifact = {
        "candidate": _json_value(result["candidate"]),
        "metrics": _json_value(result["metrics"]),
        "accounting": {
            "initial_capital": float(
                result["candidate"]["configuration"]["initial_capital"]
            ),
            "row_count": int(len(equity)),
            "first_timestamp": equity.index[0].isoformat(),
            "last_timestamp": equity.index[-1].isoformat(),
            "first_equity": float(equity["equity"].iloc[0]),
            "final_equity": float(last["equity"]),
            "final_cash": float(last["cash"]),
            "final_unrealized_pnl": float(last["unrealized_pnl"]),
            "final_open_qty": float(last["open_qty"]),
            "total_fees": float(equity["fees_paid"].sum()),
            "total_slippage_cost": float(equity["slippage_cost"].sum()),
            "combined_execution_cost": float(equity["combined_execution_cost"].sum()),
            "total_turnover_notional": float(equity["turnover"].sum()),
            "maximum_leverage_used": (
                float(leverage.max()) if leverage.notna().any() else 0.0
            ),
        },
        "weekly_returns": [
            {"week_start_utc": ts.isoformat(), "net_return": float(value)}
            for ts, value in weekly.items()
        ],
        "trades": [
            {key: _json_value(value) for key, value in row.items()}
            for row in trades.to_dict(orient="records")
        ],
        "short_position_interpretation": SYNTHETIC_SHORT_LABEL,
    }
    if include_equity_curve:
        running_peak = equity["equity"].cummax().replace(0.0, np.nan)
        drawdown = (equity["equity"] - running_peak) / running_peak
        artifact["equity_curve"] = [
            {
                "timestamp": ts.isoformat(),
                "equity": float(row["equity"]),
                "cash": float(row["cash"]),
                "drawdown": float(drawdown.loc[ts]) if pd.notna(drawdown.loc[ts]) else None,
            }
            for ts, row in equity[["equity", "cash"]].iterrows()
        ]
    return artifact


def aggregate_walk_forward_artifact(
    protocol: Mapping[str, Any], *, bundle: ExperimentBundle
) -> dict[str, Any]:
    weekly: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    total_fees = 0.0
    total_slippage = 0.0
    chained_curve: list[dict[str, Any]] = []
    chained_equity = float(protocol["baseline"]["initial_capital"])
    monetary_trade_fields = {
        "pnl", "net_pnl", "gross_pnl", "fees_paid", "entry_slippage_cost",
        "exit_slippage_cost", "total_slippage_cost", "combined_execution_cost",
        "entry_notional", "risk_capital", "max_notional",
    }
    for fold in protocol["walk_forward"]["folds"]:
        fold_id = int(fold["id"])
        selection = bundle.verify_json(f"fold_{fold_id}/selection.json")
        validation = bundle.verify_json(f"fold_{fold_id}/validation.json")["result"]
        fold_curve = list(validation.get("equity_curve", []))
        if not fold_curve:
            raise ProtocolError("A validation fold is missing its marked hourly equity curve")
        fold_initial = float(validation["accounting"]["initial_capital"])
        scale = chained_equity / fold_initial
        weekly.extend(validation.get("weekly_returns", []))
        for trade in validation.get("trades", []):
            scaled_trade = dict(trade)
            for field in monetary_trade_fields:
                if scaled_trade.get(field) is not None:
                    scaled_trade[field] = float(scaled_trade[field]) * scale
            for field in ("qty", "initial_qty"):
                if scaled_trade.get(field) is not None:
                    scaled_trade[field] = float(scaled_trade[field]) * scale
            scaled_trade["source_fold_id"] = fold_id
            scaled_trade["fold_quote_unit_scale"] = float(scale)
            trades.append(scaled_trade)
        total_fees += float(validation["accounting"]["total_fees"]) * scale
        total_slippage += float(validation["accounting"]["total_slippage_cost"]) * scale
        for point in fold_curve:
            chained_curve.append(
                {
                    "timestamp": point["timestamp"],
                    "equity": float(point["equity"]) * scale,
                }
            )
        chained_equity = float(chained_curve[-1]["equity"])
        folds.append(
            {
                "fold_id": fold_id,
                "selected_candidate_id": selection["selected_candidate_id"],
                "validation_metrics": validation["metrics"],
            }
        )
    weekly.sort(key=lambda row: row["week_start_utc"])
    starts = [row["week_start_utc"] for row in weekly]
    expected_weeks = int(protocol["partitions"]["validation"]["complete_iso_weeks"])
    if len(weekly) != expected_weeks or len(starts) != len(set(starts)):
        raise ProtocolError("Aggregate walk-forward weeks are incomplete or duplicated")
    initial = float(protocol["baseline"]["initial_capital"])
    weekly_equity = initial
    weekly_curve = []
    for row in weekly:
        weekly_equity *= 1.0 + float(row["net_return"])
        weekly_curve.append(
            {
                "week_start_utc": row["week_start_utc"],
                "equity": float(weekly_equity),
            }
        )
    if len(chained_curve) == 0:
        raise ProtocolError("Aggregate walk-forward equity curve is empty")
    timestamps = [_utc(row["timestamp"]) for row in chained_curve]
    if len(timestamps) != len(set(timestamps)) or timestamps != sorted(timestamps):
        raise ProtocolError("Aggregate walk-forward hourly timestamps overlap or are unsorted")
    running_peak = initial
    max_drawdown = 0.0
    for point in chained_curve:
        running_peak = max(running_peak, float(point["equity"]))
        point["drawdown"] = float(point["equity"]) / running_peak - 1.0
        max_drawdown = min(max_drawdown, float(point["drawdown"]))
    total_return = chained_equity / initial - 1.0
    if chained_equity <= 0:
        raise ProtocolError("Aggregate walk-forward equity is non-positive and cannot be annualized")
    annualized = (chained_equity / initial) ** (52.0 / len(weekly)) - 1.0
    return {
        "stage": "aggregate_walk_forward_validation",
        "start_inclusive": protocol["partitions"]["validation"]["start_inclusive"],
        "end_exclusive": protocol["partitions"]["validation"]["end_exclusive"],
        "aggregation": "chronological_chain_of_non_overlapping_validation_fold_marked_equity",
        "folds": folds,
        "weekly_returns": weekly,
        "weekly_equity_curve": weekly_curve,
        "equity_curve": chained_curve,
        "trades": trades,
        "metrics": {
            "complete_weeks": len(weekly),
            "trade_count": len(trades),
            "total_return": float(total_return),
            "annualized_net_return": float(annualized),
            "mean_weekly_net_return": float(np.mean([row["net_return"] for row in weekly])),
            "maximum_drawdown": float(max_drawdown),
            "drawdown_frequency": "hourly_marked_equity",
            "total_fees": float(total_fees),
            "total_slippage_cost": float(total_slippage),
            "combined_execution_cost": float(total_fees + total_slippage),
            "cost_units": "chained_portfolio_quote_units",
        },
        "short_position_interpretation": SYNTHETIC_SHORT_LABEL,
    }


def audit_evaluation_artifact(
    artifact: Mapping[str, Any],
    *,
    start_inclusive: Any,
    end_exclusive: Any,
    max_leverage: float,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    expected_week_starts: Sequence[Any] | None = None,
) -> list[str]:
    start, end = _utc(start_inclusive), _utc(end_exclusive)
    metrics = artifact.get("metrics", {})
    accounting = artifact.get("accounting", {})
    trades = list(artifact.get("trades", []))
    weekly = list(artifact.get("weekly_returns", []))
    curve = list(artifact.get("equity_curve", []))
    candidate = artifact.get("candidate", {})
    configuration = candidate.get("configuration", {})
    checks: list[str] = []

    if (
        float(configuration.get("max_leverage", math.nan)) != float(max_leverage)
        or float(configuration.get("fee_bps_per_fill", math.nan)) != float(fee_bps)
        or float(configuration.get("slippage_bps_per_fill", math.nan)) != float(slippage_bps)
        or configuration.get("risk_base") != "current_equity"
        or candidate.get("candidate_id") != metrics.get("candidate_id")
    ):
        raise ProtocolError("Evaluation configuration differs from its declared execution assumptions")
    checks.append("configuration_identity_verified")

    if _utc(metrics.get("start_inclusive")) != start or _utc(metrics.get("end_exclusive")) != end:
        raise ProtocolError("Evaluation metrics do not match the frozen interval")
    if _utc(metrics.get("context_start_inclusive")) != start - pd.Timedelta(weeks=1):
        raise ProtocolError("Evaluation did not use exactly one preceding context week")
    if int(metrics.get("context_row_count", -1)) != 7 * 24:
        raise ProtocolError("Evaluation context is not one complete hourly ISO week")
    checks.append("exact_interval_and_context")

    if _utc(accounting.get("first_timestamp")) != start:
        raise ProtocolError("Accounting begins outside the evaluation interval")
    if _utc(accounting.get("last_timestamp")) != end - pd.Timedelta(hours=1):
        raise ProtocolError("Accounting does not end on the final evaluation bar")
    if int(accounting.get("row_count", -1)) != int(metrics.get("evaluation_row_count", -2)):
        raise ProtocolError("Evaluation and accounting row counts differ")
    if abs(float(accounting.get("final_open_qty", math.nan))) > 1e-9:
        raise ProtocolError("Evaluation ended with an open position")
    if abs(float(accounting.get("final_unrealized_pnl", math.nan))) > 1e-8:
        raise ProtocolError("Evaluation ended with unrealized P&L")
    if not math.isclose(
        float(accounting.get("final_equity", math.nan)),
        float(accounting.get("final_cash", math.nan)),
        rel_tol=1e-10,
        abs_tol=1e-8,
    ):
        raise ProtocolError("Final marked equity and cash are inconsistent")
    checks.append("terminal_accounting_consistent")

    if curve:
        if len(curve) != int(accounting.get("row_count", -1)):
            raise ProtocolError("Persisted equity curve row count is inconsistent")
        if _utc(curve[0]["timestamp"]) != start or _utc(curve[-1]["timestamp"]) != end - pd.Timedelta(hours=1):
            raise ProtocolError("Persisted equity curve boundaries are inconsistent")
        if not math.isclose(
            float(curve[-1]["equity"]),
            float(accounting["final_equity"]),
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ProtocolError("Persisted equity curve differs from final accounting")
        curve_drawdown = min(float(row["drawdown"] or 0.0) for row in curve)
        if not math.isclose(curve_drawdown, float(metrics["max_drawdown"]), rel_tol=1e-9, abs_tol=1e-9):
            raise ProtocolError("Persisted equity curve drawdown differs from metrics")
        curve_series = pd.Series(
            [float(row["equity"]) for row in curve],
            index=pd.DatetimeIndex([_utc(row["timestamp"]) for row in curve]),
        )
        calculated_weekly = weekly_returns_from_equity(curve_series)
        persisted_weekly = {
            _utc(row["week_start_utc"]): float(row["net_return"])
            for row in weekly
        }
        if set(calculated_weekly.index) != set(persisted_weekly):
            raise ProtocolError("Persisted weekly-return units differ from the marked equity curve")
        for week_start, value in calculated_weekly.items():
            if not math.isclose(float(value), persisted_weekly[week_start], rel_tol=1e-10, abs_tol=1e-12):
                raise ProtocolError("Persisted weekly return differs from the marked equity curve")
        checks.append("equity_curve_reconciled")

    fees = float(accounting.get("total_fees", math.nan))
    slippage = float(accounting.get("total_slippage_cost", math.nan))
    combined = float(accounting.get("combined_execution_cost", math.nan))
    if min(fees, slippage, combined) < -1e-10 or not math.isclose(
        combined, fees + slippage, rel_tol=1e-10, abs_tol=1e-8
    ):
        raise ProtocolError("Execution-cost accounting is inconsistent")
    if not math.isclose(float(metrics.get("fees", math.nan)), fees, rel_tol=1e-10, abs_tol=1e-8):
        raise ProtocolError("Metric fees differ from accounting fees")
    if not math.isclose(
        float(metrics.get("total_slippage_cost", math.nan)), slippage, rel_tol=1e-10, abs_tol=1e-8
    ):
        raise ProtocolError("Metric slippage differs from accounting slippage")
    turnover = float(accounting.get("total_turnover_notional", math.nan))
    expected_fees = turnover * float(fee_bps) / 10_000.0
    if not math.isclose(fees, expected_fees, rel_tol=1e-9, abs_tol=1e-8):
        raise ProtocolError("Fees do not reconcile to turnover at the declared fee rate")
    if not math.isclose(float(metrics.get("total_traded_notional", math.nan)), turnover, rel_tol=1e-10, abs_tol=1e-8):
        raise ProtocolError("Metric turnover differs from accounting turnover")
    if not math.isclose(float(metrics.get("combined_execution_cost", math.nan)), combined, rel_tol=1e-10, abs_tol=1e-8):
        raise ProtocolError("Metric combined cost differs from accounting combined cost")
    checks.append("execution_costs_and_turnover_reconciled")

    actual_leverage = float(accounting.get("maximum_leverage_used", math.nan))
    if not math.isfinite(actual_leverage) or actual_leverage < 0:
        raise ProtocolError("Marked leverage is invalid")
    if not math.isclose(float(metrics.get("maximum_leverage_used", math.nan)), actual_leverage, rel_tol=1e-10, abs_tol=1e-10):
        raise ProtocolError("Metric marked leverage differs from accounting")

    if int(metrics.get("num_trades", -1)) != len(trades):
        raise ProtocolError("Trade count differs from the persisted trade records")
    entry_times: list[pd.Timestamp] = []
    week_ids: list[tuple[int, int]] = []
    trade_fee_total = 0.0
    trade_slippage_total = 0.0
    trade_combined_total = 0.0
    trade_net_total = 0.0
    entry_slippage_total = 0.0
    exit_slippage_total = 0.0
    for trade in trades:
        entry = _utc(trade["entry_time"])
        exit_time = _utc(trade["exit_time"])
        if entry < start or entry >= end or exit_time < start or exit_time >= end:
            raise ProtocolError("Trade leaked outside the evaluation interval")
        entry_times.append(entry)
        week = trade.get("week_id")
        if not isinstance(week, list) or len(week) != 2:
            raise ProtocolError("Trade week identifier is invalid")
        week_ids.append((int(week[0]), int(week[1])))
        required_fields = {
            "fees_paid", "total_slippage_cost", "combined_execution_cost",
            "entry_slippage_cost", "exit_slippage_cost", "entry_notional",
            "max_notional", "gross_pnl", "net_pnl",
        }
        if any(trade.get(field) is None for field in required_fields):
            raise ProtocolError("A trade is missing required accounting fields")
        trade_fees = float(trade["fees_paid"])
        trade_slippage = float(trade["total_slippage_cost"])
        trade_combined = float(trade["combined_execution_cost"])
        if min(trade_fees, trade_slippage, trade_combined) < -1e-10 or not math.isclose(
            trade_combined, trade_fees + trade_slippage, rel_tol=1e-10, abs_tol=1e-8
        ):
            raise ProtocolError("A trade has inconsistent execution costs")
        if not math.isclose(
            trade_slippage,
            float(trade["entry_slippage_cost"]) + float(trade["exit_slippage_cost"]),
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ProtocolError("A trade's entry and exit slippage do not reconcile")
        if not math.isclose(
            float(trade["net_pnl"]),
            float(trade["gross_pnl"]) - trade_fees,
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ProtocolError("A trade's gross P&L, fees, and net P&L do not reconcile")
        entry_notional = float(trade["entry_notional"])
        allowed_notional = float(trade["max_notional"])
        if entry_notional < -1e-10 or entry_notional > allowed_notional + 1e-8:
            raise ProtocolError("A trade exceeded its entry-time notional cap")
        if trade.get("reason") == "END_OF_DATA":
            raise ProtocolError("An unexpected terminal exit occurred in a complete Monday-aligned window")
        trade_fee_total += trade_fees
        trade_slippage_total += trade_slippage
        trade_combined_total += trade_combined
        trade_net_total += float(trade["net_pnl"])
        entry_slippage_total += float(trade["entry_slippage_cost"])
        exit_slippage_total += float(trade["exit_slippage_cost"])
    if len(entry_times) != len(set(entry_times)) or len(week_ids) != len(set(week_ids)):
        raise ProtocolError("Duplicate entry timestamps or traded ISO weeks detected")
    initial_capital = float(accounting.get("initial_capital", math.nan))
    final_equity = float(accounting.get("final_equity", math.nan))
    reconciliations = (
        (trade_fee_total, fees, "trade fees"),
        (trade_slippage_total, slippage, "trade slippage"),
        (trade_combined_total, combined, "trade combined costs"),
        (trade_net_total, final_equity - initial_capital, "trade P&L"),
        (entry_slippage_total, float(metrics.get("entry_slippage_cost", math.nan)), "entry slippage"),
        (exit_slippage_total, float(metrics.get("exit_slippage_cost", math.nan)), "exit slippage"),
    )
    for actual, expected, label in reconciliations:
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
            raise ProtocolError(f"Persisted {label} does not reconcile")
    expected_total_return = final_equity / initial_capital - 1.0
    if not math.isclose(expected_total_return, float(metrics.get("total_return", math.nan)), rel_tol=1e-10, abs_tol=1e-10):
        raise ProtocolError("Final equity does not reconcile to total return")
    checks.append("trades_unique_window_bounded_capped_and_reconciled")

    weekly_starts = [_utc(row["week_start_utc"]) for row in weekly]
    expected_starts = (
        [_utc(value) for value in expected_week_starts]
        if expected_week_starts is not None
        else list(pd.date_range(start, end - pd.Timedelta(weeks=1), freq="7D"))
    )
    if weekly_starts != expected_starts:
        raise ProtocolError("Evaluation weekly units are incomplete, duplicated, or unordered")
    checks.append("weekly_units_unique_and_window_bounded")
    return checks


def audit_holdout_results(
    protocol: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
    final_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if artifact.get("stage") != "holdout":
        raise ProtocolError("Canonical holdout container has the wrong stage identity")
    if artifact.get("allowed_configurations") != [
        "fixed_baseline",
        "single_final_selected_candidate",
    ]:
        raise ProtocolError("Canonical holdout container has unexpected configurations")
    baseline = artifact.get("baseline", {})
    selected = artifact.get("selected", {})
    if baseline.get("candidate", {}).get("candidate_id") != "fixed_baseline":
        raise ProtocolError("Canonical holdout baseline identity is invalid")
    if baseline.get("candidate", {}).get("configuration") != dict(protocol["baseline"]):
        raise ProtocolError("Canonical holdout baseline configuration differs from protocol")
    if selected.get("candidate", {}).get("candidate_id") != final_candidate.get("candidate_id"):
        raise ProtocolError("Canonical holdout selected-candidate identity is invalid")
    if selected.get("candidate", {}).get("configuration") != final_candidate.get("configuration"):
        raise ProtocolError("Canonical holdout selected configuration differs from the sealed candidate")
    start, end = partition_bounds(protocol, "holdout")
    fee_bps = float(protocol["baseline"]["fee_bps_per_fill"])
    max_leverage = float(protocol["baseline"]["max_leverage"])
    return {
        "status": "passed",
        "baseline_checks": audit_evaluation_artifact(
            baseline,
            start_inclusive=start,
            end_exclusive=end,
            max_leverage=max_leverage,
            fee_bps=fee_bps,
            slippage_bps=float(protocol["baseline"]["slippage_bps_per_fill"]),
        ),
        "selected_checks": audit_evaluation_artifact(
            selected,
            start_inclusive=start,
            end_exclusive=end,
            max_leverage=max_leverage,
            fee_bps=fee_bps,
            slippage_bps=float(protocol["baseline"]["slippage_bps_per_fill"]),
        ),
    }


def audit_preholdout_robustness(
    protocol: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
    final_candidate: Mapping[str, Any],
    missing_timestamps: Sequence[Any],
) -> dict[str, Any]:
    start, _ = partition_bounds(protocol, "development")
    end, _ = partition_bounds(protocol, "holdout")
    selected = {
        "candidate_id": final_candidate["candidate_id"],
        "config": dict(final_candidate["configuration"]),
    }
    expected_plan = robustness_plan(protocol, selected)
    outputs = list(artifact.get("outputs", []))
    if len(outputs) != len(expected_plan):
        raise ProtocolError("Pre-holdout robustness output count is incomplete")
    observed_ids = [row.get("scenario", {}).get("id") for row in outputs]
    expected_ids = [row["id"] for row in expected_plan]
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise ProtocolError("Pre-holdout robustness scenarios are missing, duplicated, or reordered")
    if [row.get("scenario") for row in outputs] != expected_plan:
        raise ProtocolError("Pre-holdout robustness scenario definitions differ from protocol")

    expected_gap_weeks = excluded_gap_weeks(missing_timestamps)
    if len(expected_gap_weeks) != 3 or artifact.get("gap_weeks") != expected_gap_weeks:
        raise ProtocolError("Pre-holdout robustness does not identify the exact canonical gap weeks")
    all_week_starts = list(pd.date_range(start, end - pd.Timedelta(weeks=1), freq="7D"))
    gap_week_set = {_utc(value) for value in expected_gap_weeks}
    non_gap_week_starts = [week for week in all_week_starts if week not in gap_week_set]
    executable_ids = {
        "baseline_costs",
        "costs_1_5x",
        "costs_2x",
        "conservative_stop_first",
        "target_first",
        "long_only",
        "short_only",
    }
    checks: list[str] = []
    for row in outputs:
        scenario = row["scenario"]
        scenario_id = scenario["id"]
        result = row.get("result", {})
        if scenario_id in executable_ids:
            expected_config = dict(selected["config"])
            expected_config.update(scenario.get("overrides", {}))
            if result.get("candidate", {}).get("configuration") != expected_config:
                raise ProtocolError(f"Robustness scenario {scenario_id} used an unexpected configuration")
            audit_evaluation_artifact(
                result,
                start_inclusive=start,
                end_exclusive=end,
                max_leverage=float(expected_config["max_leverage"]),
                fee_bps=float(expected_config["fee_bps_per_fill"]),
                slippage_bps=float(expected_config["slippage_bps_per_fill"]),
            )
            checks.append(f"{scenario_id}_audited")
        elif scenario_id == "calendar_year_subperiods":
            subperiods = list(result.get("subperiods", []))
            expected_years = list(range(start.year, end.year + 1))
            if [row.get("year") for row in subperiods] != expected_years:
                raise ProtocolError("Calendar-year robustness breakdown is incomplete")
            checks.append("calendar_years_complete")
        elif scenario_id == "non_overlapping_24_week_subperiods":
            subperiods = list(result.get("subperiods", []))
            expected_count = int((end - start) / pd.Timedelta(weeks=24))
            if len(subperiods) != expected_count:
                raise ProtocolError("The 24-week robustness breakdown is incomplete")
            cursor = start
            for number, period in enumerate(subperiods, start=1):
                period_end = cursor + pd.Timedelta(weeks=24)
                if (
                    period.get("period") != number
                    or _utc(period.get("start_inclusive")) != cursor
                    or _utc(period.get("end_exclusive")) != period_end
                    or period.get("complete_weeks") != 24
                ):
                    raise ProtocolError("A 24-week robustness subperiod has invalid boundaries")
                cursor = period_end
            expected_remainder = int((end - cursor) / pd.Timedelta(weeks=1))
            if result.get("remainder_weeks") != expected_remainder:
                raise ProtocolError("The 24-week robustness remainder is incorrect")
            checks.append("non_overlapping_24_week_periods_complete")
        elif scenario_id == "parameter_neighborhood":
            rows = list(result.get("candidate_diagnostics", []))
            grid_ids = {candidate["candidate_id"] for candidate in generate_candidates(protocol)}
            if (
                len(rows) != 18
                or {item.get("candidate_id") for item in rows} != grid_ids
                or result.get("selection_performed") is not False
            ):
                raise ProtocolError("Parameter-neighborhood diagnostics do not cover the frozen grid")
            for item in rows:
                if (
                    _utc(item.get("start_inclusive")) != start
                    or _utc(item.get("end_exclusive")) != end
                    or _utc(item.get("context_start_inclusive")) != start - pd.Timedelta(weeks=1)
                    or int(item.get("context_row_count", -1)) != 7 * 24
                ):
                    raise ProtocolError("A parameter-neighborhood diagnostic used the wrong interval")
            checks.append("frozen_parameter_neighborhood_complete")
        elif scenario_id == "exclude_canonical_gap_weeks":
            if result.get("excluded_weeks") != expected_gap_weeks:
                raise ProtocolError("Gap-week sensitivity excluded the wrong weeks")
            expected_config = dict(selected["config"])
            if result.get("candidate", {}).get("configuration") != expected_config:
                raise ProtocolError("Gap-week sensitivity used the wrong candidate configuration")
            audit_evaluation_artifact(
                result,
                start_inclusive=start,
                end_exclusive=end,
                max_leverage=float(expected_config["max_leverage"]),
                fee_bps=float(expected_config["fee_bps_per_fill"]),
                slippage_bps=float(expected_config["slippage_bps_per_fill"]),
                expected_week_starts=non_gap_week_starts,
            )
            checks.append("canonical_gap_week_exclusion_audited")
        else:
            raise ProtocolError(f"Unexpected robustness scenario: {scenario_id}")
    return {"status": "passed", "checks": checks}


def initialize_experiment_bundle(
    protocol: Mapping[str, Any],
    *,
    run_id: str,
    root: str | Path = EXPERIMENT_ROOT,
    require_clean: bool = True,
    test_suite_record: str,
    test_suite_verified: bool = False,
) -> ExperimentBundle:
    if not run_id or any(part in run_id for part in ("/", "\\", "..")):
        raise ProtocolError("run_id must be a non-empty path-safe identifier")
    if not test_suite_record:
        raise ProtocolError("A code/test version record is required")
    if require_clean and not test_suite_verified:
        raise ProtocolError("Canonical bundle initialization requires an executor-verified test run")
    manifest = verify_canonical_dataset(protocol)
    provenance = collect_code_provenance(require_clean=require_clean)
    provenance["test_suite_record"] = test_suite_record
    provenance["test_suite_verified"] = bool(test_suite_verified)
    provenance["canonical_execution_allowed"] = bool(require_clean and test_suite_verified)
    bundle = ExperimentBundle.create(Path(root) / run_id)
    bundle.seal_json(
        "protocol.json",
        {"protocol_sha256": EXPECTED_PROTOCOL_SHA256, "protocol": dict(protocol)},
    )
    bundle.seal_json("dataset_manifest.json", manifest)
    bundle.seal_json("code_provenance.json", provenance)
    bundle.seal_json(
        "candidate_grid.json",
        {"candidate_count": 18, "candidates": generate_candidates(protocol)},
    )
    return bundle


@dataclass
class ProtocolExecutor:
    protocol: Mapping[str, Any]
    bundle: ExperimentBundle
    ohlc_loader: Callable[[], pd.DataFrame] = load_canonical_ohlcv
    global_holdout_access_path: Path = GLOBAL_HOLDOUT_ACCESS_PATH
    enforce_canonical_holdout_root: bool = True

    def _verify_identity(self) -> None:
        validate_protocol(self.protocol)
        protocol_artifact = self.bundle.verify_json("protocol.json")
        if protocol_artifact["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
            raise ProtocolError("Experiment bundle uses a different protocol")
        if protocol_artifact.get("protocol") != dict(self.protocol):
            raise ProtocolError("Experiment bundle protocol payload differs from the executor protocol")
        dataset_artifact = self.bundle.verify_json("dataset_manifest.json")
        if dataset_artifact["sha256"] != self.protocol["dataset"]["sha256"]:
            raise ProtocolError("Experiment bundle uses a different dataset")
        candidate_grid = self.bundle.verify_json("candidate_grid.json")
        expected_grid = {"candidate_count": 18, "candidates": generate_candidates(self.protocol)}
        if candidate_grid != expected_grid:
            raise ProtocolError("Experiment bundle candidate grid differs from the frozen grid")
        recorded = self.bundle.verify_json("code_provenance.json")
        if recorded.get("canonical_execution_allowed") is not True:
            raise ProtocolError("This bundle was initialized for synthetic development only")
        current = collect_code_provenance(require_clean=True)
        if current["git_commit"] != recorded["git_commit"]:
            raise ProtocolError("Code commit differs from the sealed experiment provenance")

    def _evaluate(
        self,
        ohlc: pd.DataFrame,
        candidate: Mapping[str, Any],
        start: Any,
        end: Any,
    ) -> dict[str, Any]:
        return evaluate_candidate_window(
            ohlc, candidate=candidate, start_inclusive=start, end_exclusive=end
        )

    def development(self) -> dict[str, Any]:
        self._verify_identity()
        ohlc = self.ohlc_loader()
        start, end = partition_bounds(self.protocol, "development")
        candidates = generate_candidates(self.protocol)
        minimum = int(self.protocol["walk_forward"]["minimum_training_trades"])
        rows = [
            self._evaluate(ohlc, candidate, start, end)["metrics"]
            for candidate in candidates
        ]
        ranking = rank_training_candidates(rows, minimum_trades=minimum)
        artifact = {
            "stage": "development_candidate_evaluation",
            "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "candidate_count": len(candidates),
            "minimum_training_trades": minimum,
            "candidate_ranking": ranking,
        }
        self.bundle.seal_json(
            "development/candidate_ranking.json", artifact, embed_hash=True
        )
        baseline = {
            "candidate_id": "fixed_baseline",
            "target_plan_id": "midpoint_half_then_opposite",
            "config": dict(self.protocol["baseline"]),
        }
        baseline_artifact = evaluation_artifact(
            self._evaluate(ohlc, baseline, start, end)
        )
        self.bundle.seal_json(
            "development/baseline.json", baseline_artifact, embed_hash=True
        )
        return artifact

    def walk_forward(self) -> dict[str, Any]:
        self._verify_identity()
        self.bundle.verify_json("development/candidate_ranking.json")
        self.bundle.verify_json("development/baseline.json")
        ohlc = self.ohlc_loader()

        def evaluator(candidate: Mapping[str, Any], fold: Mapping[str, Any], scope: str, _fold_id: int) -> Mapping[str, Any]:
            if scope == "training":
                start, end = fold["train_start_inclusive"], fold["train_end_exclusive"]
            elif scope == "validation":
                start, end = fold["validation_start_inclusive"], fold["validation_end_exclusive"]
            else:
                raise ProtocolError(f"Unknown walk-forward scope: {scope}")
            result = self._evaluate(ohlc, candidate, start, end)
            if scope == "training":
                return result["metrics"]
            return {
                "metrics": result["metrics"],
                "artifact": evaluation_artifact(result),
            }

        folds = run_frozen_walk_forward(
            self.protocol, evaluator=evaluator, bundle=self.bundle
        )
        aggregate = aggregate_walk_forward_artifact(
            self.protocol, bundle=self.bundle
        )
        self.bundle.seal_json(
            "walk_forward_aggregate.json", aggregate, embed_hash=True
        )
        return {"folds": folds, "aggregate": aggregate}

    def final_selection(self) -> dict[str, Any]:
        self._verify_identity()
        audit = self.bundle.verify_json("walk_forward_audit.json")
        if audit.get("status") != "passed":
            raise ProtocolError("Walk-forward audit has not passed")
        ohlc = self.ohlc_loader()
        start, _ = partition_bounds(self.protocol, "development")
        end, _ = partition_bounds(self.protocol, "holdout")
        evaluations = []
        for candidate in generate_candidates(self.protocol):
            row = self._evaluate(ohlc, candidate, start, end)["metrics"]
            row["scope"] = "training"
            evaluations.append(row)
        provenance = self.bundle.verify_json("code_provenance.json")
        ranking = rank_training_candidates(
            evaluations,
            minimum_trades=int(self.protocol["walk_forward"]["minimum_training_trades"]),
        )
        self.bundle.seal_json(
            "final_fit/training_scores.json",
            {"training_scores": ranking},
            embed_hash=True,
        )
        return persist_final_candidate(
            self.protocol,
            evaluations=ranking,
            bundle=self.bundle,
            code_commit=provenance["git_commit"],
        )

    def audit_walk_forward(self) -> dict[str, Any]:
        self._verify_identity()
        global_access_sidecar = self.global_holdout_access_path.with_suffix(
            self.global_holdout_access_path.suffix + ".sha256"
        )
        if self.global_holdout_access_path.exists() or global_access_sidecar.exists() or (
            self.bundle.root / "holdout/ACCESS.json"
        ).exists() or (
            self.bundle.root / "holdout/canonical_results.json"
        ).exists():
            raise ProtocolError("Holdout was accessed before the walk-forward audit")
        development = self.bundle.verify_json("development/candidate_ranking.json")
        baseline = self.bundle.verify_json("development/baseline.json")
        if int(development.get("candidate_count", -1)) != 18:
            raise ProtocolError("Development artifact does not contain exactly 18 candidates")
        if len(development.get("candidate_ranking", [])) != 18:
            raise ProtocolError("Development ranking is incomplete")
        dev_start, dev_end = partition_bounds(self.protocol, "development")
        checks = audit_evaluation_artifact(
            baseline,
            start_inclusive=dev_start,
            end_exclusive=dev_end,
            max_leverage=float(self.protocol["baseline"]["max_leverage"]),
        )
        grid = {row["candidate_id"]: row for row in generate_candidates(self.protocol)}
        development_rows = development["candidate_ranking"]
        if {row.get("candidate_id") for row in development_rows} != set(grid):
            raise ProtocolError("Development candidate IDs differ from the frozen grid")
        if [row.get("rank") for row in development_rows] != list(range(1, 19)):
            raise ProtocolError("Development ranking is not complete and deterministic")
        for row in development_rows:
            if _utc(row.get("start_inclusive")) != dev_start or _utc(row.get("end_exclusive")) != dev_end:
                raise ProtocolError("A development candidate used the wrong interval")
            if _utc(row.get("context_start_inclusive")) != dev_start - pd.Timedelta(weeks=1):
                raise ProtocolError("A development candidate used the wrong context")
            if int(row.get("context_row_count", -1)) != 7 * 24:
                raise ProtocolError("A development candidate lacks complete context")
        fold_results = []
        minimum = int(self.protocol["walk_forward"]["minimum_training_trades"])
        for fold in self.protocol["walk_forward"]["folds"]:
            fold_id = int(fold["id"])
            selection = self.bundle.verify_json(f"fold_{fold_id}/selection.json")
            validation = self.bundle.verify_json(f"fold_{fold_id}/validation.json")
            training_scores = selection.get("training_scores", [])
            if len(training_scores) != 18:
                raise ProtocolError(f"Fold {fold_id} training ranking is incomplete")
            if {row.get("candidate_id") for row in training_scores} != set(grid):
                raise ProtocolError(f"Fold {fold_id} candidate IDs differ from the frozen grid")
            if [row.get("rank") for row in training_scores] != list(range(1, 19)):
                raise ProtocolError(f"Fold {fold_id} training ranks are inconsistent")
            train_start = _utc(fold["train_start_inclusive"])
            train_end = _utc(fold["train_end_exclusive"])
            for row in training_scores:
                if row.get("scope") != "training":
                    raise ProtocolError(f"Fold {fold_id} contains a non-training ranking row")
                if _utc(row.get("start_inclusive")) != train_start or _utc(row.get("end_exclusive")) != train_end:
                    raise ProtocolError(f"Fold {fold_id} training row used the wrong interval")
                if _utc(row.get("context_start_inclusive")) != train_start - pd.Timedelta(weeks=1):
                    raise ProtocolError(f"Fold {fold_id} training row used the wrong context")
                if int(row.get("context_row_count", -1)) != 7 * 24:
                    raise ProtocolError(f"Fold {fold_id} training row lacks complete context")
            winner = select_training_winner(training_scores, minimum_trades=minimum)
            selected_id = selection.get("selected_candidate_id")
            if selected_id != winner["candidate_id"] or selected_id not in grid:
                raise ProtocolError(f"Fold {fold_id} selected candidate is not its training winner")
            if selection.get("selected_candidate") != grid[selected_id]:
                raise ProtocolError(f"Fold {fold_id} selected configuration differs from the frozen grid")
            expected_training = {
                "start_inclusive": fold["train_start_inclusive"],
                "end_exclusive": fold["train_end_exclusive"],
            }
            expected_validation = {
                "start_inclusive": fold["validation_start_inclusive"],
                "end_exclusive": fold["validation_end_exclusive"],
            }
            if selection.get("training_window") != expected_training or selection.get("validation_window") != expected_validation:
                raise ProtocolError(f"Fold {fold_id} persisted boundaries differ from protocol")
            result = validation.get("result", {})
            if result.get("metrics", {}).get("candidate_id") != selected_id:
                raise ProtocolError(f"Fold {fold_id} validation did not use the sealed winner")
            fold_checks = audit_evaluation_artifact(
                result,
                start_inclusive=fold["validation_start_inclusive"],
                end_exclusive=fold["validation_end_exclusive"],
                max_leverage=float(self.protocol["baseline"]["max_leverage"]),
            )
            fold_results.append(
                {
                    "fold_id": fold_id,
                    "selected_candidate_id": selected_id,
                    "checks": fold_checks,
                }
            )
        aggregate = self.bundle.verify_json("walk_forward_aggregate.json")
        aggregate_metrics = aggregate.get("metrics", {})
        expected_validation_weeks = int(
            self.protocol["partitions"]["validation"]["complete_iso_weeks"]
        )
        validation_start, validation_end = partition_bounds(self.protocol, "validation")
        if (
            int(aggregate_metrics.get("complete_weeks", -1)) != expected_validation_weeks
            or _utc(aggregate.get("start_inclusive")) != validation_start
            or _utc(aggregate.get("end_exclusive")) != validation_end
        ):
            raise ProtocolError("Aggregate walk-forward artifact has the wrong weekly coverage")
        aggregate_candidates = [row.get("selected_candidate_id") for row in aggregate.get("folds", [])]
        if aggregate_candidates != [row["selected_candidate_id"] for row in fold_results]:
            raise ProtocolError("Aggregate walk-forward candidates differ from sealed folds")
        aggregate_weekly = list(aggregate.get("weekly_returns", []))
        weekly_starts = [_utc(row["week_start_utc"]) for row in aggregate_weekly]
        expected_starts = list(
            pd.date_range(
                validation_start,
                validation_end - pd.Timedelta(weeks=1),
                freq="7D",
            )
        )
        if weekly_starts != expected_starts:
            raise ProtocolError("Aggregate walk-forward weeks are not the exact chronological validation weeks")
        aggregate_curve = list(aggregate.get("equity_curve", []))
        curve_times = [_utc(row["timestamp"]) for row in aggregate_curve]
        expected_hours = list(
            pd.date_range(
                validation_start,
                validation_end - pd.Timedelta(hours=1),
                freq="1h",
            )
        )
        if curve_times != expected_hours:
            raise ProtocolError("Aggregate walk-forward curve does not cover every validation hour exactly once")
        curve_values = np.asarray([float(row["equity"]) for row in aggregate_curve])
        peaks = np.maximum.accumulate(curve_values)
        recomputed_drawdown = float(np.min(curve_values / peaks - 1.0))
        if not math.isclose(
            recomputed_drawdown,
            float(aggregate_metrics.get("maximum_drawdown", math.nan)),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ProtocolError("Aggregate walk-forward drawdown is not based on hourly marked equity")
        initial = float(self.protocol["baseline"]["initial_capital"])
        final_equity = float(curve_values[-1])
        if not math.isclose(
            final_equity / initial - 1.0,
            float(aggregate_metrics.get("total_return", math.nan)),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ProtocolError("Aggregate walk-forward final equity and total return differ")
        compounded_weekly = initial * float(
            np.prod([1.0 + float(row["net_return"]) for row in aggregate_weekly])
        )
        if not math.isclose(compounded_weekly, final_equity, rel_tol=1e-9, abs_tol=1e-7):
            raise ProtocolError("Aggregate weekly returns and hourly marked equity do not reconcile")
        aggregate_trades = list(aggregate.get("trades", []))
        fee_sum = sum(float(row.get("fees_paid") or 0.0) for row in aggregate_trades)
        slippage_sum = sum(float(row.get("total_slippage_cost") or 0.0) for row in aggregate_trades)
        if not math.isclose(fee_sum, float(aggregate_metrics.get("total_fees", math.nan)), rel_tol=1e-9, abs_tol=1e-8):
            raise ProtocolError("Aggregate walk-forward trade fees do not reconcile")
        if not math.isclose(slippage_sum, float(aggregate_metrics.get("total_slippage_cost", math.nan)), rel_tol=1e-9, abs_tol=1e-8):
            raise ProtocolError("Aggregate walk-forward trade slippage does not reconcile")
        artifact = {
            "status": "passed",
            "holdout_accessed": False,
            "development_checks": checks,
            "folds": fold_results,
            "aggregate_checks": [
                "exact_72_week_and_hourly_coverage",
                "hourly_drawdown_reconciled",
                "weekly_and_hourly_equity_reconciled",
                "scaled_execution_costs_reconciled",
            ],
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.bundle.seal_json("walk_forward_audit.json", artifact, embed_hash=True)
        return artifact

    def holdout(self, *, confirm_holdout: bool) -> dict[str, Any]:
        self._verify_identity()
        if self.enforce_canonical_holdout_root and self.bundle.root.parent.resolve() != EXPERIMENT_ROOT.resolve():
            raise HoldoutLockedError("Production holdout execution requires the canonical experiment root")
        if self.enforce_canonical_holdout_root and self.global_holdout_access_path.resolve() != GLOBAL_HOLDOUT_ACCESS_PATH.resolve():
            raise HoldoutLockedError("Production holdout execution requires the canonical global access registry")
        final_candidate = assert_holdout_ready(
            self.protocol, bundle=self.bundle, confirm_holdout=confirm_holdout
        )
        authorize_holdout(
            self.protocol,
            bundle=self.bundle,
            confirm_holdout=confirm_holdout,
            global_access_path=self.global_holdout_access_path,
        )
        ohlc = self.ohlc_loader()
        start, end = partition_bounds(self.protocol, "holdout")
        baseline = {
            "candidate_id": "fixed_baseline",
            "target_plan_id": "midpoint_half_then_opposite",
            "config": dict(self.protocol["baseline"]),
        }
        selected = {
            "candidate_id": final_candidate["candidate_id"],
            "config": final_candidate["configuration"],
        }
        artifact: dict[str, Any] = {
            "stage": "holdout",
            "allowed_configurations": ["fixed_baseline", "single_final_selected_candidate"],
            "baseline": evaluation_artifact(self._evaluate(ohlc, baseline, start, end)),
            "selected": evaluation_artifact(self._evaluate(ohlc, selected, start, end)),
        }
        artifact["audit"] = audit_holdout_results(
            self.protocol,
            artifact=artifact,
            final_candidate=final_candidate,
        )
        self.bundle.seal_json("holdout/canonical_results.json", artifact, embed_hash=True)
        sealed_results = self.bundle.verify_json("holdout/canonical_results.json")

        settings = self.protocol["uncertainty"]
        uncertainty = bootstrap_from_evaluation_artifact(
            sealed_results["selected"], settings=settings
        )
        uncertainty.update(
            {
                "source_stage": "untouched_holdout_selected_candidate",
                "source_artifact": "holdout/canonical_results.json",
                "candidate_id": final_candidate["candidate_id"],
                "confirmatory_null_test": True,
            }
        )
        self.bundle.seal_json("holdout/uncertainty.json", uncertainty, embed_hash=True)

        holdout_scenarios = [
            {
                "id": "costs_1_5x",
                "overrides": self.protocol["cost_scenarios"]["stress_1_5x"],
            },
            {
                "id": "costs_2x",
                "overrides": self.protocol["cost_scenarios"]["stress_2x"],
            },
            {
                "id": "target_first",
                "overrides": {"intrabar_policy": "target_first"},
            },
        ]
        sensitivity_outputs = []
        for scenario in holdout_scenarios:
            varied = {
                "candidate_id": selected["candidate_id"],
                "config": dict(selected["config"]),
            }
            varied["config"].update(scenario["overrides"])
            result = evaluation_artifact(
                self._evaluate(ohlc, varied, start, end),
                include_equity_curve=False,
            )
            audit_evaluation_artifact(
                result,
                start_inclusive=start,
                end_exclusive=end,
                max_leverage=float(varied["config"]["max_leverage"]),
                fee_bps=float(varied["config"]["fee_bps_per_fill"]),
                slippage_bps=float(varied["config"]["slippage_bps_per_fill"]),
            )
            sensitivity_outputs.append({"scenario": scenario, "result": result})
        holdout_robustness = {
            "stage": "holdout_preregistered_sensitivities",
            "canonical_result_sealed_first": True,
            "selected_candidate_id": final_candidate["candidate_id"],
            "selection_changed": False,
            "outputs": sensitivity_outputs,
        }
        self.bundle.seal_json(
            "holdout/robustness.json", holdout_robustness, embed_hash=True
        )
        return {
            "stage": "holdout_complete",
            "canonical_results": "holdout/canonical_results.json",
            "uncertainty": "holdout/uncertainty.json",
            "robustness": "holdout/robustness.json",
        }

    def robustness(self) -> dict[str, Any]:
        self._verify_identity()
        final_candidate = self.bundle.verify_json("final_fit/final_candidate.json")
        selected = {
            "candidate_id": final_candidate["candidate_id"],
            "config": final_candidate["configuration"],
        }
        ohlc = self.ohlc_loader()
        start, _ = partition_bounds(self.protocol, "development")
        end, _ = partition_bounds(self.protocol, "holdout")
        manifest = self.bundle.verify_json("dataset_manifest.json")

        baseline_result = self._evaluate(ohlc, selected, start, end)

        def evaluate_scenario(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
            if scenario["id"] == "calendar_year_subperiods":
                rows = []
                for year, group in baseline_result["equity"].groupby(baseline_result["equity"].index.year):
                    em = equity_metrics(group["equity"], bars_per_year=365.0 * 24.0)
                    rows.append(
                        {
                            "year": int(year),
                            "start_inclusive": group.index[0].isoformat(),
                            "end_inclusive": group.index[-1].isoformat(),
                            "observed_hours": int(len(group)),
                            **em,
                        }
                    )
                return {
                    "subperiods": rows,
                    "construction": "calendar_slices_of_continuous_preholdout_marked_equity",
                }
            if scenario["id"] == "non_overlapping_24_week_subperiods":
                rows = []
                cursor = start
                period = 1
                while cursor + pd.Timedelta(weeks=24) <= end:
                    period_end = cursor + pd.Timedelta(weeks=24)
                    group = baseline_result["equity"][
                        (baseline_result["equity"].index >= cursor)
                        & (baseline_result["equity"].index < period_end)
                    ]
                    em = equity_metrics(group["equity"], bars_per_year=365.0 * 24.0)
                    rows.append(
                        {
                            "period": period,
                            "start_inclusive": cursor.isoformat(),
                            "end_exclusive": period_end.isoformat(),
                            "complete_weeks": 24,
                            "observed_hours": int(len(group)),
                            **em,
                        }
                    )
                    cursor = period_end
                    period += 1
                return {
                    "subperiods": rows,
                    "remainder_weeks": int((end - cursor) / pd.Timedelta(weeks=1)),
                    "construction": "non_overlapping_slices_of_continuous_preholdout_marked_equity",
                }
            if scenario["id"] == "parameter_neighborhood":
                rows = []
                for candidate in generate_candidates(self.protocol):
                    rows.append(self._evaluate(ohlc, candidate, start, end)["metrics"])
                return {"candidate_diagnostics": rows, "selection_performed": False}
            if scenario["id"] == "exclude_canonical_gap_weeks":
                filtered = exclude_gap_week_rows(ohlc, manifest.get("missing_timestamps", []))
                result = evaluation_artifact(
                    self._evaluate(filtered, selected, start, end),
                    include_equity_curve=False,
                )
                result["excluded_weeks"] = excluded_gap_weeks(
                    manifest.get("missing_timestamps", [])
                )
                return result
            varied = {"candidate_id": selected["candidate_id"], "config": dict(selected["config"])}
            varied["config"].update(scenario.get("overrides", {}))
            return evaluation_artifact(
                self._evaluate(ohlc, varied, start, end),
                include_equity_curve=False,
            )

        artifact = run_robustness_diagnostics(
            self.protocol, selected_candidate=selected, evaluator=evaluate_scenario
        )
        artifact["gap_weeks"] = excluded_gap_weeks(manifest.get("missing_timestamps", []))
        if len(artifact["gap_weeks"]) != 3:
            raise ProtocolError("Canonical gap sensitivity must exclude exactly three ISO weeks")
        artifact["stage"] = "full_preholdout_diagnostic_robustness"
        artifact["evidence_class"] = "diagnostic_in_sample_for_final_selected_candidate"
        artifact["start_inclusive"] = start.isoformat()
        artifact["end_exclusive"] = end.isoformat()
        artifact["parameter_neighborhood"] = [x["candidate_id"] for x in generate_candidates(self.protocol)]
        artifact["calendar_years"] = list(range(start.year, end.year + 1))
        artifact["subperiod_weeks"] = 24
        artifact["audit"] = audit_preholdout_robustness(
            self.protocol,
            artifact=artifact,
            final_candidate=final_candidate,
            missing_timestamps=manifest.get("missing_timestamps", []),
        )
        self.bundle.seal_json("robustness/results.json", artifact, embed_hash=True)
        return artifact

    def uncertainty(self) -> dict[str, Any]:
        self._verify_identity()
        self.bundle.verify_json("robustness/results.json")
        aggregate = self.bundle.verify_json("walk_forward_aggregate.json")
        settings = self.protocol["uncertainty"]
        result = bootstrap_from_evaluation_artifact(
            aggregate,
            settings=settings,
        )
        result.update(
            {
                "source_stage": "aggregate_walk_forward_validation",
                "source_artifact": "walk_forward_aggregate.json",
                "confirmatory_null_test": False,
            }
        )
        self.bundle.seal_json("uncertainty/weekly_bootstrap.json", result, embed_hash=True)
        return result
