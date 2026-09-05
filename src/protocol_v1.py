from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
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
    return {"metrics": metrics, "equity": evaluated, "trades": trades}


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
            "fold_1", "fold_2", "fold_3", "final_fit", "holdout",
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


Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any], str, int], Mapping[str, Any]]


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
            row = dict(evaluator(candidate, fold, "training", fold_id))
            row.update({"candidate_id": candidate["candidate_id"], "scope": "training"})
            row["score"] = selection_score(row)
            training.append(row)
        winner = select_training_winner(training, minimum_trades=minimum)
        selection_path = f"fold_{fold_id}/selection.json"
        bundle.seal_json(
            selection_path,
            {"fold_id": fold_id, "training_scores": training, "selected_candidate_id": winner["candidate_id"]},
        )
        bundle.verify_json(selection_path)
        selected = next(x for x in candidates if x["candidate_id"] == winner["candidate_id"])
        validation = dict(evaluator(selected, fold, "validation", fold_id))
        validation.update({"candidate_id": selected["candidate_id"], "scope": "validation"})
        validation_path = f"fold_{fold_id}/validation.json"
        bundle.seal_json(validation_path, {"fold_id": fold_id, "result": validation})
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
    if not provenance.get("git_commit") or not provenance.get("test_suite_record"):
        raise HoldoutLockedError("Code commit and passing test record are required")
    for fold_id in (1, 2, 3):
        selection = bundle.verify_json(f"fold_{fold_id}/selection.json")
        validation = bundle.verify_json(f"fold_{fold_id}/validation.json")
        if selection.get("fold_id") != fold_id or validation.get("fold_id") != fold_id:
            raise HoldoutLockedError(f"Fold {fold_id} artifact identity is inconsistent")
        selected_id = selection.get("selected_candidate_id")
        validated_id = validation.get("result", {}).get("candidate_id")
        if not selected_id or selected_id != validated_id:
            raise HoldoutLockedError(f"Fold {fold_id} validation does not match its sealed training winner")
    candidate = bundle.verify_json("final_fit/final_candidate.json")
    final_scores = bundle.verify_json("final_fit/training_scores.json").get(
        "training_scores", []
    )
    if not final_scores:
        raise HoldoutLockedError("Final pre-holdout training scores are missing")
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
    return candidate


def authorize_holdout(
    protocol: Mapping[str, Any], *, bundle: ExperimentBundle, confirm_holdout: bool
) -> dict[str, Any]:
    candidate = assert_holdout_ready(protocol, bundle=bundle, confirm_holdout=confirm_holdout)
    marker = {
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": EXPECTED_PROTOCOL_SHA256,
        "dataset_hash": protocol["dataset"]["sha256"],
        "candidate_id": candidate["candidate_id"],
        "candidate_artifact_hash": candidate["artifact_hash"],
        "allowed_holdout_runs": ["fixed_baseline", "single_final_selected_candidate"],
    }
    bundle.seal_json("holdout/ACCESS.json", marker, embed_hash=True)
    return bundle.verify_json("holdout/ACCESS.json")


def moving_block_bootstrap(
    weekly_returns: Sequence[float] | pd.Series,
    *,
    block_length_weeks: int = 4,
    replications: int = 10_000,
    seed: int = 20260905,
) -> dict[str, Any]:
    values = np.asarray(weekly_returns, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    block = int(block_length_weeks)
    reps = int(replications)
    if n < block or block < 1 or reps < 1:
        raise ProtocolError("Bootstrap requires n >= block length and positive replications")
    starts = np.arange(0, n - block + 1)
    draws_per_rep = int(math.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        chosen = rng.choice(starts, size=draws_per_rep, replace=True)
        sample = np.concatenate([values[j : j + block] for j in chosen])[:n]
        means[i] = float(sample.mean())
    low, median, high = np.percentile(means, [2.5, 50.0, 97.5])
    return {
        "method": "moving_block_bootstrap",
        "unit": "complete_utc_iso_week",
        "block_length_weeks": block,
        "replications": reps,
        "seed": int(seed),
        "point_estimate": float(values.mean()),
        "percentile_2_5": float(low),
        "percentile_50": float(median),
        "percentile_97_5": float(high),
        "reject_null": bool(low > 0.0),
    }


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


def evaluation_artifact(result: Mapping[str, Any]) -> dict[str, Any]:
    equity = result["equity"]
    trades = result["trades"]
    weekly = weekly_returns_from_equity(equity["equity"])
    return {
        "metrics": dict(result["metrics"]),
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


def initialize_experiment_bundle(
    protocol: Mapping[str, Any],
    *,
    run_id: str,
    root: str | Path = EXPERIMENT_ROOT,
    require_clean: bool = True,
    test_suite_record: str,
) -> ExperimentBundle:
    if not run_id or any(part in run_id for part in ("/", "\\", "..")):
        raise ProtocolError("run_id must be a non-empty path-safe identifier")
    if not test_suite_record:
        raise ProtocolError("A code/test version record is required")
    manifest = verify_canonical_dataset(protocol)
    provenance = collect_code_provenance(require_clean=require_clean)
    provenance["test_suite_record"] = test_suite_record
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
        recorded = self.bundle.verify_json("code_provenance.json")
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
        rows = []
        for candidate in generate_candidates(self.protocol):
            rows.append(self._evaluate(ohlc, candidate, start, end)["metrics"])
        artifact = {
            "stage": "development_candidate_evaluation",
            "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "candidate_scores": rows,
        }
        self.bundle.seal_json("development.json", artifact)
        return artifact

    def walk_forward(self) -> list[dict[str, Any]]:
        self._verify_identity()
        ohlc = self.ohlc_loader()

        def evaluator(candidate: Mapping[str, Any], fold: Mapping[str, Any], scope: str, _fold_id: int) -> Mapping[str, Any]:
            if scope == "training":
                start, end = fold["train_start_inclusive"], fold["train_end_exclusive"]
            elif scope == "validation":
                start, end = fold["validation_start_inclusive"], fold["validation_end_exclusive"]
            else:
                raise ProtocolError(f"Unknown walk-forward scope: {scope}")
            return self._evaluate(ohlc, candidate, start, end)["metrics"]

        return run_frozen_walk_forward(
            self.protocol, evaluator=evaluator, bundle=self.bundle
        )

    def final_selection(self) -> dict[str, Any]:
        self._verify_identity()
        for fold_id in (1, 2, 3):
            self.bundle.verify_json(f"fold_{fold_id}/selection.json")
            self.bundle.verify_json(f"fold_{fold_id}/validation.json")
        ohlc = self.ohlc_loader()
        start, _ = partition_bounds(self.protocol, "development")
        end, _ = partition_bounds(self.protocol, "holdout")
        evaluations = []
        for candidate in generate_candidates(self.protocol):
            row = self._evaluate(ohlc, candidate, start, end)["metrics"]
            row["scope"] = "training"
            evaluations.append(row)
        provenance = self.bundle.verify_json("code_provenance.json")
        self.bundle.seal_json(
            "final_fit/training_scores.json", {"training_scores": evaluations}
        )
        return persist_final_candidate(
            self.protocol,
            evaluations=evaluations,
            bundle=self.bundle,
            code_commit=provenance["git_commit"],
        )

    def holdout(self, *, confirm_holdout: bool) -> dict[str, Any]:
        self._verify_identity()
        final_candidate = assert_holdout_ready(
            self.protocol, bundle=self.bundle, confirm_holdout=confirm_holdout
        )
        authorize_holdout(
            self.protocol, bundle=self.bundle, confirm_holdout=confirm_holdout
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
        artifact = {
            "stage": "holdout",
            "allowed_configurations": ["fixed_baseline", "single_final_selected_candidate"],
            "baseline": evaluation_artifact(self._evaluate(ohlc, baseline, start, end)),
            "selected": evaluation_artifact(self._evaluate(ohlc, selected, start, end)),
        }
        self.bundle.seal_json("holdout/canonical_results.json", artifact, embed_hash=True)
        return artifact

    def robustness(self) -> dict[str, Any]:
        self._verify_identity()
        final_candidate = self.bundle.verify_json("final_fit/final_candidate.json")
        selected = {
            "candidate_id": final_candidate["candidate_id"],
            "config": final_candidate["configuration"],
        }
        ohlc = self.ohlc_loader()
        start, end = partition_bounds(self.protocol, "validation")
        manifest = self.bundle.verify_json("dataset_manifest.json")

        baseline_result = self._evaluate(ohlc, selected, start, end)

        def evaluate_scenario(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
            if scenario["id"] == "calendar_year_subperiods":
                rows = []
                for year, group in baseline_result["equity"].groupby(baseline_result["equity"].index.year):
                    em = equity_metrics(group["equity"], bars_per_year=365.0 * 24.0)
                    rows.append({"year": int(year), **em})
                return {"subperiods": rows}
            if scenario["id"] == "non_overlapping_24_week_subperiods":
                rows = []
                for fold in self.protocol["walk_forward"]["folds"]:
                    result = self._evaluate(
                        ohlc,
                        selected,
                        fold["validation_start_inclusive"],
                        fold["validation_end_exclusive"],
                    )
                    rows.append(evaluation_artifact(result))
                return {"subperiods": rows}
            if scenario["id"] == "parameter_neighborhood":
                rows = []
                for candidate in generate_candidates(self.protocol):
                    rows.append(self._evaluate(ohlc, candidate, start, end)["metrics"])
                return {"candidate_diagnostics": rows, "selection_performed": False}
            if scenario["id"] == "exclude_canonical_gap_weeks":
                filtered = exclude_gap_week_rows(ohlc, manifest.get("missing_timestamps", []))
                return evaluation_artifact(self._evaluate(filtered, selected, start, end))
            varied = {"candidate_id": selected["candidate_id"], "config": dict(selected["config"])}
            varied["config"].update(scenario.get("overrides", {}))
            return evaluation_artifact(self._evaluate(ohlc, varied, start, end))

        artifact = run_robustness_diagnostics(
            self.protocol, selected_candidate=selected, evaluator=evaluate_scenario
        )
        artifact["gap_weeks"] = excluded_gap_weeks(manifest.get("missing_timestamps", []))
        artifact["parameter_neighborhood"] = [x["candidate_id"] for x in generate_candidates(self.protocol)]
        artifact["calendar_years"] = list(range(start.year, end.year + 1))
        artifact["subperiod_weeks"] = 24
        self.bundle.seal_json("robustness/results.json", artifact, embed_hash=True)
        return artifact

    def uncertainty(self) -> dict[str, Any]:
        self._verify_identity()
        robustness = self.bundle.verify_json("robustness/results.json")
        baseline = next(
            row for row in robustness["outputs"] if row["scenario"]["id"] == "baseline_costs"
        )
        returns = [x["net_return"] for x in baseline["result"]["weekly_returns"]]
        settings = self.protocol["uncertainty"]
        result = moving_block_bootstrap(
            returns,
            block_length_weeks=int(settings["block_length_weeks"]),
            replications=int(settings["replications"]),
            seed=int(settings["random_seed"]),
        )
        self.bundle.seal_json("uncertainty/weekly_bootstrap.json", result, embed_hash=True)
        return result
