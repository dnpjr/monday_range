"""Read-only access to the sealed Monday Range Protocol V1 research bundle.

This module is intentionally presentation-focused.  It verifies semantic SHA-256
sidecars before returning data and never invokes a protocol stage or writes into
``reports/experiments``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .canonical_data import load_canonical_ohlcv
from .protocol_v1 import evaluate_candidate_window, generate_candidates, load_frozen_protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = REPO_ROOT / "reports" / "experiments" / "monday_range_protocol_v1"
RUN_ID = "canonical_20260905_3528492"
RUN_ROOT = PROTOCOL_ROOT / RUN_ID

EXPECTED_PROTOCOL_HASH = "da0d41ce67d445bcb308012bf323d2e573d246a7a1fd3eb57c9ee6bc951d4cf3"
EXPECTED_DATASET_HASH = "4d541711c323ac82e07bade75a522c0a48776c7d2e4eec7f45d8d9c4e00b41ce"
EXPECTED_CODE_COMMIT = "352849299200b749496d588a1d7d1149503750fc"
EXPECTED_CANDIDATE_ID = "mr1_stop125_day2_full_at_midpoint"
HOLDOUT_START = pd.Timestamp("2025-05-19T00:00:00Z")

_REQUIRED_ARTIFACTS = (
    "candidate_grid.json",
    "code_provenance.json",
    "dataset_manifest.json",
    "development/baseline.json",
    "development/candidate_ranking.json",
    "final_fit/final_candidate.json",
    "final_fit/training_scores.json",
    "fold_1/selection.json",
    "fold_1/validation.json",
    "fold_2/selection.json",
    "fold_2/validation.json",
    "fold_3/selection.json",
    "fold_3/validation.json",
    "holdout/ACCESS.json",
    "holdout/canonical_results.json",
    "holdout/robustness.json",
    "holdout/uncertainty.json",
    "protocol.json",
    "robustness/results.json",
    "uncertainty/weekly_bootstrap.json",
    "walk_forward_aggregate.json",
    "walk_forward_audit.json",
)

# Exact semantic hashes freeze the complete public result bundle, not only the
# holdout headline files. A matching edited JSON/sidecar pair must still fail.
EXPECTED_ARTIFACT_HASHES = {
    "../HOLDOUT_ACCESS.json": "5e4bc2df03939220a0f878bdcb8986c554cd8d2596f741b7c589e954d532d68f",
    "candidate_grid.json": "b88b6a568a74d57096c8b9cdc32e67adbffebcc33863a0eabdf8b7e5e4c52383",
    "code_provenance.json": "6714f233bb533367c3d22381f1cfd15a62321c4992f5d51f242eaeb5849c38dd",
    "dataset_manifest.json": "3650497ed97b33d5c372fb32bbe1bda51dbe5acb89733072341f47fbbc476e78",
    "development/baseline.json": "6f88aa633a3b6cee1fe53352226eec31560a6d3b10b9dbf662921fa770540fbc",
    "development/candidate_ranking.json": "fba0fd48295083fda6f5696eccc7f0c86c429891df96f002133077a1b71f7297",
    "final_fit/final_candidate.json": "37e9e7c6d0386791556f5566d257a776ec09c11f1398a17b1d70cd89754097c0",
    "final_fit/training_scores.json": "96b6800078cafadaeacd11f20be638e1c64f2a3174f71ff3983457fdf5af7b6d",
    "fold_1/selection.json": "66572d648bf32090cec0769fd136905f47c68dc1e54498b0a6db58555f4d59d7",
    "fold_1/validation.json": "d296589f2ca1f8b0a6f442b35be753763408af3fdc745a55fe95ed906666a454",
    "fold_2/selection.json": "68ed3a4f811e5b816c3e33316b8d3081cc474ff137e29406f034ff8be1f732eb",
    "fold_2/validation.json": "f2aaa3f0ad63acb45686dfb4553c3c3c20be635646287894635551e7b08edc28",
    "fold_3/selection.json": "360898e449ab52cb37846e16db06217656a380ada08ec6a0e8bdc014670a9017",
    "fold_3/validation.json": "d19d621e1a832feaba5c358da80168cee82e831b317e6b87136f5220571ecad8",
    "holdout/ACCESS.json": "5e4bc2df03939220a0f878bdcb8986c554cd8d2596f741b7c589e954d532d68f",
    "holdout/canonical_results.json": "a2600bce1cd1f34c163ed9ddb570084d686589467fb89d3bbb998e59fa66bc96",
    "holdout/robustness.json": "3ede4b422a0cac0145d9c111891b3c4d1bd3e24c02cb953e8b60989fe32a7833",
    "holdout/uncertainty.json": "354b0405179bcb36a690e197bb4d12b57f7946b9ee89cce341a24f0088013eed",
    "protocol.json": "53b741a6789f619d00ec4c3875b12e60b23ee64d73fd099631e621b2ddf32098",
    "robustness/results.json": "72d9a374958a3be8a90f627133aaadd85353e7cc62b963a92bcad636dc4028db",
    "uncertainty/weekly_bootstrap.json": "3e53b69f121c921bce655833471225e9a95ae54dc5ac7020bea573967c5e784c",
    "walk_forward_aggregate.json": "a191b23b781b5327a79e18655c34141ae7d7b32c4978e2ffdd743e1306e2fe9a",
    "walk_forward_audit.json": "580994a0cf3e7530fcbd199fea3ff08b36f811c3ddf2e12ed1fd139c48c0d168",
}


class ResearchArtifactError(RuntimeError):
    """Raised when canonical presentation artifacts are missing or altered."""


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_hash", None)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_sealed_json(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load one sealed artifact and verify its semantic hash and sidecar."""
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if not target.is_file() or not sidecar.is_file():
        raise ResearchArtifactError(f"Missing sealed artifact or sidecar: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchArtifactError(f"Cannot read sealed artifact: {target}") from exc
    if not isinstance(payload, dict):
        raise ResearchArtifactError(f"Sealed artifact must contain a JSON object: {target}")
    digest = _semantic_hash(payload)
    try:
        expected = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise ResearchArtifactError(f"Cannot read sealed artifact sidecar: {sidecar}") from exc
    embedded = payload.get("artifact_hash")
    if digest != expected or (embedded is not None and embedded != digest):
        raise ResearchArtifactError(f"Sealed artifact hash mismatch: {target}")
    return payload, digest


@dataclass(frozen=True)
class CanonicalResearch:
    """Verified immutable inputs used by the portfolio application."""

    root: Path
    protocol: dict[str, Any]
    dataset_manifest: dict[str, Any]
    provenance: dict[str, Any]
    access: dict[str, Any]
    final_candidate: dict[str, Any]
    development_baseline: dict[str, Any]
    development_ranking: dict[str, Any]
    fold_selections: tuple[dict[str, Any], ...]
    fold_validations: tuple[dict[str, Any], ...]
    walk_forward: dict[str, Any]
    preholdout_robustness: dict[str, Any]
    preholdout_uncertainty: dict[str, Any]
    holdout: dict[str, Any]
    holdout_robustness: dict[str, Any]
    holdout_uncertainty: dict[str, Any]
    artifact_hashes: dict[str, str]

    @property
    def selected(self) -> dict[str, Any]:
        return self.holdout["selected"]

    @property
    def baseline(self) -> dict[str, Any]:
        return self.holdout["baseline"]


def load_canonical_research(
    *, run_root: str | Path = RUN_ROOT, protocol_root: str | Path | None = None
) -> CanonicalResearch:
    """Verify the full sealed bundle, then expose the canonical presentation data."""
    root = Path(run_root)
    outer = Path(protocol_root) if protocol_root is not None else root.parent
    hashes: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for relative in _REQUIRED_ARTIFACTS:
        payloads[relative], hashes[relative] = verify_sealed_json(root / relative)

    global_access, global_digest = verify_sealed_json(outer / "HOLDOUT_ACCESS.json")
    hashes["../HOLDOUT_ACCESS.json"] = global_digest
    local_access = payloads["holdout/ACCESS.json"]
    if global_access != local_access:
        raise ResearchArtifactError("Global and bundle holdout access markers differ")

    protocol_artifact = payloads["protocol.json"]
    final_candidate = payloads["final_fit/final_candidate.json"]
    holdout = payloads["holdout/canonical_results.json"]
    provenance = payloads["code_provenance.json"]
    manifest = payloads["dataset_manifest.json"]

    identity_errors = []
    if root.name != RUN_ID or global_access.get("run_id") != RUN_ID:
        identity_errors.append("run ID")
    if protocol_artifact.get("protocol_sha256") != EXPECTED_PROTOCOL_HASH:
        identity_errors.append("protocol hash")
    if manifest.get("sha256") != EXPECTED_DATASET_HASH:
        identity_errors.append("dataset hash")
    if provenance.get("git_commit") != EXPECTED_CODE_COMMIT:
        identity_errors.append("sealed code commit")
    if final_candidate.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        identity_errors.append("final candidate")
    if holdout.get("selected", {}).get("candidate", {}).get("candidate_id") != EXPECTED_CANDIDATE_ID:
        identity_errors.append("holdout candidate")
    selected_metrics = holdout.get("selected", {}).get("metrics", {})
    if selected_metrics.get("start_inclusive") != "2025-05-19T00:00:00+00:00" or selected_metrics.get("end_exclusive") != "2026-05-18T00:00:00+00:00":
        identity_errors.append("holdout boundaries")
    for relative, expected in EXPECTED_ARTIFACT_HASHES.items():
        if hashes.get(relative) != expected:
            identity_errors.append(relative)
    if identity_errors:
        raise ResearchArtifactError(
            "Canonical research identity check failed: " + ", ".join(identity_errors)
        )

    return CanonicalResearch(
        root=root,
        protocol=protocol_artifact["protocol"],
        dataset_manifest=manifest,
        provenance=provenance,
        access=global_access,
        final_candidate=final_candidate,
        development_baseline=payloads["development/baseline.json"],
        development_ranking=payloads["development/candidate_ranking.json"],
        fold_selections=tuple(payloads[f"fold_{number}/selection.json"] for number in (1, 2, 3)),
        fold_validations=tuple(payloads[f"fold_{number}/validation.json"] for number in (1, 2, 3)),
        walk_forward=payloads["walk_forward_aggregate.json"],
        preholdout_robustness=payloads["robustness/results.json"],
        preholdout_uncertainty=payloads["uncertainty/weekly_bootstrap.json"],
        holdout=holdout,
        holdout_robustness=payloads["holdout/robustness.json"],
        holdout_uncertainty=payloads["holdout/uncertainty.json"],
        artifact_hashes=hashes,
    )


def equity_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result.get("equity_curve", []))
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "cash", "drawdown"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    for column in ("equity", "cash", "drawdown"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def weekly_return_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result.get("weekly_returns", []))
    if frame.empty:
        return pd.DataFrame(columns=["week_start_utc", "net_return"])
    frame["week_start_utc"] = pd.to_datetime(frame["week_start_utc"], utc=True)
    frame["net_return"] = pd.to_numeric(frame["net_return"], errors="coerce")
    return frame.sort_values("week_start_utc").reset_index(drop=True)


def trade_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result.get("trades", []))
    for column in ("entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    config = candidate.get("configuration", candidate.get("config", {}))
    target = "Midpoint"
    if config.get("exit_style") == "partial_tp2":
        target = "Half at midpoint, remainder at opposite boundary"
    elif config.get("single_target_level") == "tp2":
        target = "Opposite boundary"
    return {
        "candidate_id": candidate.get("candidate_id"),
        "stop_offset": float(config.get("stop_range_fraction", 0.0)),
        "latest_entry_day": "Wednesday" if int(config.get("max_entry_day_utc", 2)) == 2 else "Thursday",
        "target": target,
        "direction": config.get("direction", "both"),
        "risk_fraction": float(config.get("risk_fraction", 0.0)),
        "max_leverage": float(config.get("max_leverage", 0.0)),
        "fee_bps": float(config.get("fee_bps_per_fill", 0.0)),
        "slippage_bps": float(config.get("slippage_bps_per_fill", 0.0)),
        "intrabar_policy": config.get("intrabar_policy"),
    }


def walk_forward_table(research: CanonicalResearch) -> pd.DataFrame:
    rows = []
    for selection, validation in zip(research.fold_selections, research.fold_validations):
        candidate = selection["selected_candidate"]
        config = candidate.get("config", candidate.get("configuration", {}))
        metrics = validation["result"]["metrics"]
        rows.append(
            {
                "Fold": int(selection["fold_id"]),
                "Training winner": selection["selected_candidate_id"],
                "Stop": f"{float(config['stop_range_fraction']):.2f}R",
                "Latest entry": "Wed" if int(config["max_entry_day_utc"]) == 2 else "Thu",
                "Target": candidate_summary(candidate)["target"],
                "Validation return": float(metrics["total_return"]),
                "Max drawdown": float(metrics["max_drawdown"]),
                "Trades": int(metrics["num_trades"]),
                "Profit factor": metrics.get("profit_factor"),
            }
        )
    return pd.DataFrame(rows)


def _find_candidate(protocol: Mapping[str, Any], stop_offset: float, latest_entry_day: int, target_plan: str) -> dict[str, Any]:
    for candidate in generate_candidates(protocol):
        if (
            float(candidate["config"]["stop_range_fraction"]) == float(stop_offset)
            and int(candidate["config"]["max_entry_day_utc"]) == int(latest_entry_day)
            and candidate["target_plan_id"] == target_plan
        ):
            return {**candidate, "config": dict(candidate["config"])}
    raise ValueError("The selected settings are outside the frozen 18-candidate grid")


def build_exploratory_candidate(
    *,
    stop_offset: float,
    latest_entry_day: int,
    target_plan: str,
    fee_bps: float,
    slippage_bps: float,
    intrabar_policy: str,
    direction: str,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frozen = dict(protocol) if protocol is not None else load_frozen_protocol()
    candidate = _find_candidate(frozen, stop_offset, latest_entry_day, target_plan)
    candidate["config"].update(
        {
            "fee_bps_per_fill": float(fee_bps),
            "slippage_bps_per_fill": float(slippage_bps),
            "intrabar_policy": str(intrabar_policy),
            "direction": str(direction),
        }
    )
    candidate["candidate_id"] = "explore__" + candidate["candidate_id"]
    return candidate


def run_exploratory_backtest(
    *,
    start_inclusive: Any,
    end_exclusive: Any,
    stop_offset: float,
    latest_entry_day: int,
    target_plan: str,
    fee_bps: float,
    slippage_bps: float,
    intrabar_policy: str,
    direction: str,
    ohlc: pd.DataFrame | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one in-memory exploratory backtest; no path or writer is accepted."""
    start = pd.Timestamp(start_inclusive)
    end = pd.Timestamp(end_exclusive)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    if start.weekday() != 0 or end.weekday() != 0 or start.hour != 0 or end.hour != 0:
        raise ValueError("Explore boundaries must be Monday 00:00 UTC")
    if end <= start:
        raise ValueError("Explore end must be after start")
    if end > HOLDOUT_START:
        raise ValueError("Explore is limited to pre-holdout data")
    candidate = build_exploratory_candidate(
        stop_offset=stop_offset,
        latest_entry_day=latest_entry_day,
        target_plan=target_plan,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        intrabar_policy=intrabar_policy,
        direction=direction,
        protocol=protocol,
    )
    market = load_canonical_ohlcv() if ohlc is None else ohlc
    return evaluate_candidate_window(
        market,
        candidate=candidate,
        start_inclusive=start,
        end_exclusive=end,
    )
