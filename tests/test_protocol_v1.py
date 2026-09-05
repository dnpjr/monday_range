from __future__ import annotations

import copy
import json
import math
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.backtest import backtest_sweep_fade
from run_backtest import build_summary
from src.protocol_v1 import (
    EXPECTED_PROTOCOL_SHA256,
    ArtifactSealedError,
    ExperimentBundle,
    HoldoutLockedError,
    ProtocolError,
    ProtocolExecutor,
    aggregate_walk_forward_artifact,
    audit_evaluation_artifact,
    audit_holdout_results,
    audit_preholdout_robustness,
    assert_holdout_ready,
    authorize_holdout,
    bootstrap_from_evaluation_artifact,
    collect_code_provenance,
    evaluate_candidate_window,
    evaluation_artifact,
    exclude_gap_week_rows,
    excluded_gap_weeks,
    feature_window_with_context,
    generate_candidates,
    initialize_experiment_bundle,
    load_frozen_protocol,
    moving_block_bootstrap,
    payload_sha256,
    persist_final_candidate,
    rank_training_candidates,
    robustness_plan,
    run_frozen_walk_forward,
    run_robustness_diagnostics,
    select_training_winner,
    selection_score,
    validate_protocol,
    verify_canonical_dataset,
    weekly_returns_from_equity,
)


def _protocol() -> dict:
    return load_frozen_protocol()


def _hourly(start: str, end_exclusive: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end_exclusive, utc=True) - pd.Timedelta(hours=1)
    index = pd.date_range(start_ts, end_ts, freq="1h")
    frame = pd.DataFrame(index=index)
    frame["open"] = 100.0
    frame["high"] = 101.0
    frame["low"] = 99.0
    frame["close"] = 100.0
    frame["volume"] = 1.0
    return frame


def _featured(rows: list[dict], times: list[str] | None = None) -> pd.DataFrame:
    if times is None:
        times = [f"2024-01-02 {hour:02d}:00:00+00:00" for hour in range(len(rows))]
    frame = pd.DataFrame(rows, index=pd.to_datetime(times))
    defaults = {
        "mon_high": 110.0,
        "mon_low": 90.0,
        "mon_mid": 100.0,
        "mon_range": 20.0,
        "iso_year": 2024,
        "iso_week": 1,
        "weekday": 1,
        "is_tradeable": True,
    }
    for key, value in defaults.items():
        if key not in frame:
            frame[key] = value
        else:
            frame[key] = frame[key].fillna(value)
    return frame


def _metrics(candidate_id: str, *, score_bias: float = 0.0, trades: int = 30) -> dict:
    return {
        "candidate_id": candidate_id,
        "num_trades": trades,
        "net_cagr": 0.10 + score_bias,
        "max_drawdown": -0.10,
        "annualized_cost_drag_fraction": 0.01,
        "annualized_turnover_multiple": 2.0,
        "sortino": 1.0,
    }


def _ranked_rows(protocol: dict, start: str, end: str) -> list[dict]:
    rows = []
    for position, candidate in enumerate(generate_candidates(protocol)):
        row = _metrics(
            candidate["candidate_id"],
            score_bias=0.02 if position == 0 else -position / 10_000.0,
        )
        row.update(
            {
                "scope": "training",
                "start_inclusive": pd.Timestamp(start).isoformat(),
                "end_exclusive": pd.Timestamp(end).isoformat(),
                "context_start_inclusive": (
                    pd.Timestamp(start) - pd.Timedelta(weeks=1)
                ).isoformat(),
                "context_row_count": 168,
            }
        )
        rows.append(row)
    return rank_training_candidates(rows, minimum_trades=30)


def _zero_evaluation_artifact(
    candidate: dict,
    start: str,
    end: str,
    *,
    excluded_weeks: list[str] | None = None,
) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    excluded = {pd.Timestamp(value) for value in (excluded_weeks or [])}
    week_starts = [
        value
        for value in pd.date_range(start_ts, end_ts - pd.Timedelta(weeks=1), freq="7D")
        if value not in excluded
    ]
    configuration = dict(candidate["config"])
    return {
        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "target_plan_id": candidate.get("target_plan_id"),
            "configuration": configuration,
        },
        "metrics": {
            "candidate_id": candidate["candidate_id"],
            "start_inclusive": start_ts.isoformat(),
            "end_exclusive": end_ts.isoformat(),
            "context_start_inclusive": (start_ts - pd.Timedelta(weeks=1)).isoformat(),
            "context_row_count": 168,
            "evaluation_row_count": 1,
            "num_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "fees": 0.0,
            "entry_slippage_cost": 0.0,
            "exit_slippage_cost": 0.0,
            "total_slippage_cost": 0.0,
            "combined_execution_cost": 0.0,
            "total_traded_notional": 0.0,
            "maximum_leverage_used": 0.0,
        },
        "accounting": {
            "initial_capital": 10_000.0,
            "row_count": 1,
            "first_timestamp": start_ts.isoformat(),
            "last_timestamp": (end_ts - pd.Timedelta(hours=1)).isoformat(),
            "first_equity": 10_000.0,
            "final_equity": 10_000.0,
            "final_cash": 10_000.0,
            "final_unrealized_pnl": 0.0,
            "final_open_qty": 0.0,
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
            "combined_execution_cost": 0.0,
            "total_turnover_notional": 0.0,
            "maximum_leverage_used": 0.0,
        },
        "weekly_returns": [
            {"week_start_utc": value.isoformat(), "net_return": 0.0}
            for value in week_starts
        ],
        "trades": [],
    }


class ProtocolValidationTests(unittest.TestCase):
    def test_frozen_protocol_hash_and_amended_boundary(self) -> None:
        protocol = _protocol()
        self.assertEqual(EXPECTED_PROTOCOL_SHA256, "da0d41ce67d445bcb308012bf323d2e573d246a7a1fd3eb57c9ee6bc951d4cf3")
        development = protocol["partitions"]["development"]
        self.assertEqual(development["start_inclusive"], "2021-05-31T00:00:00Z")
        self.assertEqual(development["complete_iso_weeks"], 135)
        self.assertEqual(protocol["partitions"]["initial_context_only"]["start_inclusive"], "2021-05-24T00:00:00Z")

    def test_protocol_hash_change_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "protocol.json"
            path.write_text(json.dumps(_protocol(), indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(ProtocolError):
                load_frozen_protocol(path)

    def test_partition_gap_or_overlap_fails(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["partitions"]["validation"]["start_inclusive"] = "2023-12-25T00:00:00Z"
        with self.assertRaises(ProtocolError):
            validate_protocol(protocol)

    def test_non_utc_boundary_fails(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["partitions"]["development"]["start_inclusive"] = "2021-05-31T01:00:00+01:00"
        with self.assertRaises(ProtocolError):
            validate_protocol(protocol)

    def test_unrepresentable_accounting_or_strategy_semantics_fail(self) -> None:
        for key, value in (("accounting_version", "legacy"), ("signal_bar", "same_bar")):
            protocol = copy.deepcopy(_protocol())
            protocol["baseline"][key] = value
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                validate_protocol(protocol)

    def test_dataset_hash_mismatch_fails(self) -> None:
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            data_path = root / "data.csv"
            manifest_path = root / "manifest.json"
            data_path.write_text("not-the-canonical-data\n", encoding="utf-8")
            manifest_path.write_text(json.dumps({"sha256": protocol["dataset"]["sha256"]}), encoding="utf-8")
            with self.assertRaises(ProtocolError):
                verify_canonical_dataset(protocol, data_path=data_path, manifest_path=manifest_path)


class ContextAndGridTests(unittest.TestCase):
    def test_fold_one_receives_exact_complete_preceding_week(self) -> None:
        frame = _hourly("2021-05-24", "2021-06-07")
        featured = feature_window_with_context(
            frame,
            start_inclusive="2021-05-31T00:00:00Z",
            end_exclusive="2021-06-07T00:00:00Z",
        )
        self.assertEqual(featured.index.min(), pd.Timestamp("2021-05-24T00:00:00Z"))
        self.assertEqual(int((~featured["protocol_entry_eligible"]).sum()), 7 * 24)
        self.assertEqual(int(featured["protocol_entry_eligible"].sum()), 7 * 24)

    def test_window_without_complete_context_fails(self) -> None:
        frame = _hourly("2021-05-24T01:00:00Z", "2021-06-07")
        with self.assertRaises(ProtocolError):
            feature_window_with_context(
                frame,
                start_inclusive="2021-05-31T00:00:00Z",
                end_exclusive="2021-06-07T00:00:00Z",
            )

    def test_context_can_build_features_but_cannot_trade(self) -> None:
        frame = _hourly("2021-05-24", "2021-06-07")
        frame.loc[pd.Timestamp("2021-05-25T00:00:00Z"), ["low", "close"]] = [98.0, 100.0]
        featured = feature_window_with_context(
            frame,
            start_inclusive="2021-05-31T00:00:00Z",
            end_exclusive="2021-06-07T00:00:00Z",
        )
        output, trades = backtest_sweep_fade(
            featured,
            risk_fraction=0.01,
            max_leverage=1.0,
            entry_start_utc="2021-05-31T00:00:00Z",
            entry_end_exclusive_utc="2021-06-07T00:00:00Z",
        )
        evaluated = output[output.index >= pd.Timestamp("2021-05-31T00:00:00Z")]
        self.assertTrue(trades.empty)
        self.assertTrue((evaluated["equity"] == 10_000.0).all())

    def test_executor_evaluates_current_strategy_on_exact_synthetic_window(self) -> None:
        frame = _hourly("2021-05-24", "2021-06-07")
        candidate = generate_candidates(_protocol())[0]
        result = evaluate_candidate_window(
            frame,
            candidate=candidate,
            start_inclusive="2021-05-31T00:00:00Z",
            end_exclusive="2021-06-07T00:00:00Z",
        )
        self.assertEqual(result["metrics"]["candidate_id"], candidate["candidate_id"])
        self.assertEqual(result["metrics"]["num_trades"], 0)
        self.assertEqual(result["equity"].index.min(), pd.Timestamp("2021-05-31T00:00:00Z"))
        artifact = evaluation_artifact(result)
        self.assertEqual(len(artifact["weekly_returns"]), 1)
        json.dumps(artifact, allow_nan=False)

    def test_candidate_grid_is_exact_unique_and_stable(self) -> None:
        first = generate_candidates(_protocol())
        second = generate_candidates(_protocol())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        self.assertEqual(len({row["candidate_id"] for row in first}), 18)
        self.assertEqual(
            {row["config"]["stop_range_fraction"] for row in first}, {0.75, 1.0, 1.25}
        )
        self.assertEqual({row["config"]["max_entry_day_utc"] for row in first}, {2, 3})
        self.assertEqual(len({row["target_plan_id"] for row in first}), 3)


class SelectionTests(unittest.TestCase):
    def test_frozen_objective_hand_calculation(self) -> None:
        row = {
            "net_cagr": 0.12,
            "max_drawdown": -0.20,
            "annualized_cost_drag_fraction": 0.01,
            "annualized_turnover_multiple": 4.0,
        }
        self.assertAlmostEqual(selection_score(row), 0.12 - 0.10 - 0.01 - 0.002)

    def test_minimum_trade_gate(self) -> None:
        with self.assertRaises(ProtocolError):
            select_training_winner([_metrics("candidate", trades=29)], minimum_trades=30)

    def test_tie_breakers_are_deterministic_in_frozen_order(self) -> None:
        a = _metrics("a")
        b = _metrics("b")
        b["sortino"] = 2.0
        self.assertEqual(select_training_winner([a, b])["candidate_id"], "b")
        b["sortino"] = a["sortino"]
        b["max_drawdown"] = -0.05
        b["net_cagr"] -= 0.025  # preserve the same score after the lower drawdown penalty
        self.assertEqual(select_training_winner([a, b])["candidate_id"], "b")
        c = dict(b, candidate_id="c", annualized_turnover_multiple=1.0)
        c["net_cagr"] -= 0.0005  # preserve score after the lower turnover penalty
        self.assertEqual(select_training_winner([b, c])["candidate_id"], "c")
        d = dict(c, candidate_id="d")
        self.assertEqual(select_training_winner([d, c])["candidate_id"], "c")

    def test_walk_forward_selects_training_winner_only(self) -> None:
        protocol = _protocol()
        ids = [x["candidate_id"] for x in generate_candidates(protocol)]
        training_winner = ids[0]
        validation_calls: list[str] = []

        def evaluator(candidate, _fold, scope, _fold_id):
            if scope == "training":
                return _metrics(candidate["candidate_id"], score_bias=0.10 if candidate["candidate_id"] == training_winner else 0.0)
            validation_calls.append(candidate["candidate_id"])
            # A different candidate would be best in validation, but it is never evaluated.
            return _metrics(candidate["candidate_id"], score_bias=-0.50)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = ExperimentBundle.create(pathlib.Path(tmp) / "bundle")
            results = run_frozen_walk_forward(protocol, evaluator=evaluator, bundle=bundle)
        self.assertEqual([x["selected_candidate_id"] for x in results], [training_winner] * 3)
        self.assertEqual(validation_calls, [training_winner] * 3)


class ArtifactAndHoldoutTests(unittest.TestCase):
    def test_bundle_initialization_records_provenance_and_layout(self) -> None:
        protocol = _protocol()
        manifest = {
            "dataset_version": protocol["dataset"]["version"],
            "sha256": protocol["dataset"]["sha256"],
        }
        provenance = {"git_commit": "abc", "working_tree": "clean"}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.protocol_v1.verify_canonical_dataset", return_value=manifest
        ), patch("src.protocol_v1.collect_code_provenance", return_value=provenance):
            bundle = initialize_experiment_bundle(
                protocol,
                run_id="synthetic-run",
                root=tmp,
                test_suite_record="synthetic 1/1 passed",
                test_suite_verified=True,
            )
            self.assertTrue((bundle.root / "charts").is_dir())
            self.assertEqual(bundle.verify_json("candidate_grid.json")["candidate_count"], 18)
            self.assertEqual(bundle.verify_json("code_provenance.json")["test_suite_record"], "synthetic 1/1 passed")

    def test_sealed_artifact_cannot_be_overwritten_or_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = ExperimentBundle.create(pathlib.Path(tmp) / "bundle")
            bundle.seal_json("fold_1/result.json", {"value": 1})
            with self.assertRaises(ArtifactSealedError):
                bundle.seal_json("fold_1/result.json", {"value": 2})
            (bundle.root / "fold_1/result.json").write_text('{"value": 3}\n', encoding="utf-8")
            with self.assertRaises(ProtocolError):
                bundle.verify_json("fold_1/result.json")

    def test_final_candidate_contains_verifiable_artifact_hash(self) -> None:
        protocol = _protocol()
        candidate_id = generate_candidates(protocol)[0]["candidate_id"]
        with tempfile.TemporaryDirectory() as tmp:
            bundle = ExperimentBundle.create(pathlib.Path(tmp) / "bundle")
            artifact = persist_final_candidate(
                protocol,
                evaluations=[_metrics(candidate_id)],
                bundle=bundle,
                code_commit="abc123",
                timestamp_utc="2026-09-05T00:00:00+00:00",
            )
            self.assertEqual(artifact["artifact_hash"], payload_sha256({k: v for k, v in artifact.items() if k != "artifact_hash"}))

    def test_canonical_initialization_rejects_unverified_test_text(self) -> None:
        with self.assertRaises(ProtocolError):
            initialize_experiment_bundle(
                _protocol(),
                run_id="must-fail-before-io",
                root="unused",
                test_suite_record="claimed passed",
                test_suite_verified=False,
            )

    def _ready_bundle(self, root: pathlib.Path, protocol: dict) -> ExperimentBundle:
        bundle = ExperimentBundle.create(root)
        bundle.seal_json("protocol.json", {"protocol_sha256": EXPECTED_PROTOCOL_SHA256, "protocol": protocol})
        missing_timestamps = [
            "2021-08-13T02:00:00+00:00",
            "2021-08-13T03:00:00+00:00",
            "2021-08-13T04:00:00+00:00",
            "2021-08-13T05:00:00+00:00",
            "2021-09-29T07:00:00+00:00",
            "2021-09-29T08:00:00+00:00",
            "2023-03-24T13:00:00+00:00",
        ]
        bundle.seal_json(
            "dataset_manifest.json",
            {
                "sha256": protocol["dataset"]["sha256"],
                "missing_timestamps": missing_timestamps,
            },
        )
        bundle.seal_json(
            "code_provenance.json",
            {
                "git_commit": "abc",
                "test_suite_record": "synthetic passed",
                "test_suite_verified": True,
                "canonical_execution_allowed": True,
            },
        )
        grid = generate_candidates(protocol)
        bundle.seal_json("candidate_grid.json", {"candidate_count": 18, "candidates": grid})
        candidate = grid[0]
        dev = protocol["partitions"]["development"]
        development_rows = _ranked_rows(
            protocol, dev["start_inclusive"], dev["end_exclusive"]
        )
        bundle.seal_json(
            "development/candidate_ranking.json",
            {
                "candidate_count": 18,
                "candidate_ranking": development_rows,
            },
        )
        bundle.seal_json("development/baseline.json", {"synthetic": True})
        fold_ids = []
        for fold in protocol["walk_forward"]["folds"]:
            fold_id = int(fold["id"])
            scores = _ranked_rows(
                protocol,
                fold["train_start_inclusive"],
                fold["train_end_exclusive"],
            )
            selected_id = scores[0]["candidate_id"]
            fold_ids.append(selected_id)
            bundle.seal_json(
                f"fold_{fold_id}/selection.json",
                {
                    "fold_id": fold_id,
                    "training_scores": scores,
                    "selected_candidate_id": selected_id,
                },
            )
            bundle.seal_json(
                f"fold_{fold_id}/validation.json",
                {
                    "fold_id": fold_id,
                    "result": {"metrics": {"candidate_id": selected_id}},
                },
            )
        validation = protocol["partitions"]["validation"]
        week_starts = pd.date_range(
            validation["start_inclusive"],
            pd.Timestamp(validation["end_exclusive"]) - pd.Timedelta(weeks=1),
            freq="7D",
        )
        aggregate_artifact = {
            "metrics": {"complete_weeks": 72},
            "weekly_returns": [
                {"week_start_utc": ts.isoformat(), "net_return": 0.0}
                for ts in week_starts
            ],
            "trades": [],
            "folds": [
                {"fold_id": i + 1, "selected_candidate_id": selected_id}
                for i, selected_id in enumerate(fold_ids)
            ],
        }
        bundle.seal_json("walk_forward_aggregate.json", aggregate_artifact)
        bundle.seal_json(
            "walk_forward_audit.json", {"status": "passed", "holdout_accessed": False}
        )
        holdout_start = protocol["partitions"]["holdout"]["start_inclusive"]
        final_rows = _ranked_rows(protocol, dev["start_inclusive"], holdout_start)
        bundle.seal_json(
            "final_fit/training_scores.json", {"training_scores": final_rows}
        )
        final_candidate = persist_final_candidate(
            protocol,
            evaluations=final_rows,
            bundle=bundle,
            code_commit="abc",
            timestamp_utc="2026-09-05T00:00:00+00:00",
        )
        gap_weeks = [
            "2021-08-09T00:00:00+00:00",
            "2021-09-27T00:00:00+00:00",
            "2023-03-20T00:00:00+00:00",
        ]
        selected = next(row for row in grid if row["candidate_id"] == final_candidate["candidate_id"])
        robustness_outputs = []
        pre_start = dev["start_inclusive"]
        pre_end = holdout_start
        for scenario in robustness_plan(protocol, selected):
            scenario_id = scenario["id"]
            if scenario_id == "calendar_year_subperiods":
                result = {
                    "subperiods": [
                        {"year": year}
                        for year in range(pd.Timestamp(pre_start).year, pd.Timestamp(pre_end).year + 1)
                    ],
                    "construction": "calendar_slices_of_continuous_preholdout_marked_equity",
                }
            elif scenario_id == "non_overlapping_24_week_subperiods":
                cursor = pd.Timestamp(pre_start)
                subperiods = []
                while cursor + pd.Timedelta(weeks=24) <= pd.Timestamp(pre_end):
                    period_end = cursor + pd.Timedelta(weeks=24)
                    subperiods.append(
                        {
                            "period": len(subperiods) + 1,
                            "start_inclusive": cursor.isoformat(),
                            "end_exclusive": period_end.isoformat(),
                            "complete_weeks": 24,
                        }
                    )
                    cursor = period_end
                result = {
                    "subperiods": subperiods,
                    "remainder_weeks": int(
                        (pd.Timestamp(pre_end) - cursor) / pd.Timedelta(weeks=1)
                    ),
                    "construction": "non_overlapping_slices_of_continuous_preholdout_marked_equity",
                }
            elif scenario_id == "parameter_neighborhood":
                result = {
                    "candidate_diagnostics": [
                        {
                            **_metrics(candidate_row["candidate_id"]),
                            "start_inclusive": pd.Timestamp(pre_start).isoformat(),
                            "end_exclusive": pd.Timestamp(pre_end).isoformat(),
                            "context_start_inclusive": (
                                pd.Timestamp(pre_start) - pd.Timedelta(weeks=1)
                            ).isoformat(),
                            "context_row_count": 168,
                        }
                        for candidate_row in grid
                    ],
                    "selection_performed": False,
                }
            else:
                varied = {"candidate_id": selected["candidate_id"], "config": dict(selected["config"])}
                varied["config"].update(scenario.get("overrides", {}))
                result = _zero_evaluation_artifact(
                    varied,
                    pre_start,
                    pre_end,
                    excluded_weeks=(gap_weeks if scenario_id == "exclude_canonical_gap_weeks" else None),
                )
                if scenario_id == "exclude_canonical_gap_weeks":
                    result["excluded_weeks"] = gap_weeks
            robustness_outputs.append({"scenario": scenario, "result": result})
        robustness_artifact = {
            "selected_candidate_id": final_candidate["candidate_id"],
            "selection_changed": False,
            "start_inclusive": pd.Timestamp(dev["start_inclusive"]).isoformat(),
            "end_exclusive": pd.Timestamp(holdout_start).isoformat(),
            "gap_weeks": gap_weeks,
            "outputs": robustness_outputs,
        }
        robustness_artifact["audit"] = audit_preholdout_robustness(
            protocol,
            artifact=robustness_artifact,
            final_candidate=final_candidate,
            missing_timestamps=missing_timestamps,
        )
        bundle.seal_json("robustness/results.json", robustness_artifact)
        settings = protocol["uncertainty"]
        uncertainty_artifact = bootstrap_from_evaluation_artifact(
            aggregate_artifact, settings=settings
        )
        uncertainty_artifact.update(
            {
                "source_stage": "aggregate_walk_forward_validation",
                "source_artifact": "walk_forward_aggregate.json",
                "confirmatory_null_test": False,
            }
        )
        bundle.seal_json(
            "uncertainty/weekly_bootstrap.json",
            uncertainty_artifact,
        )
        return bundle

    def test_holdout_requires_explicit_confirmation(self) -> None:
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._ready_bundle(pathlib.Path(tmp) / "bundle", protocol)
            with self.assertRaises(HoldoutLockedError):
                assert_holdout_ready(protocol, bundle=bundle, confirm_holdout=False)
            self.assertEqual(assert_holdout_ready(protocol, bundle=bundle, confirm_holdout=True)["candidate_id"], generate_candidates(protocol)[0]["candidate_id"])

    def test_holdout_fails_before_prerequisites_are_sealed(self) -> None:
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = ExperimentBundle.create(pathlib.Path(tmp) / "bundle")
            with self.assertRaises(ProtocolError):
                assert_holdout_ready(protocol, bundle=bundle, confirm_holdout=True)

    def test_holdout_authorization_marker_is_one_time_and_hashable(self) -> None:
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bundle = self._ready_bundle(root / "run-one", protocol)
            global_marker = root / "HOLDOUT_ACCESS.json"
            marker = authorize_holdout(
                protocol,
                bundle=bundle,
                confirm_holdout=True,
                global_access_path=global_marker,
            )
            self.assertEqual(marker["protocol_hash"], EXPECTED_PROTOCOL_SHA256)
            self.assertTrue(global_marker.exists())
            with self.assertRaises(HoldoutLockedError):
                authorize_holdout(
                    protocol,
                    bundle=bundle,
                    confirm_holdout=True,
                    global_access_path=global_marker,
                )

    def test_global_holdout_marker_blocks_a_second_run_id(self) -> None:
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            first = self._ready_bundle(root / "run-one", protocol)
            second = self._ready_bundle(root / "run-two", protocol)
            global_marker = root / "HOLDOUT_ACCESS.json"
            authorize_holdout(
                protocol,
                bundle=first,
                confirm_holdout=True,
                global_access_path=global_marker,
            )
            with self.assertRaises(HoldoutLockedError):
                authorize_holdout(
                    protocol,
                    bundle=second,
                    confirm_holdout=True,
                    global_access_path=global_marker,
                )
            self.assertFalse((second.root / "holdout/ACCESS.json").exists())

    def test_incomplete_final_grid_cannot_unlock_holdout(self) -> None:
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._ready_bundle(pathlib.Path(tmp) / "bundle", protocol)
            scores_path = bundle.root / "final_fit/training_scores.json"
            sidecar = scores_path.with_suffix(".json.sha256")
            scores_path.unlink()
            sidecar.unlink()
            bundle.seal_json(
                "final_fit/training_scores.json",
                {"training_scores": _ranked_rows(
                    protocol,
                    protocol["partitions"]["development"]["start_inclusive"],
                    protocol["partitions"]["holdout"]["start_inclusive"],
                )[:-1]},
            )
            with self.assertRaises(HoldoutLockedError):
                assert_holdout_ready(protocol, bundle=bundle, confirm_holdout=True)

    def test_empty_robustness_result_cannot_unlock_holdout(self) -> None:
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._ready_bundle(pathlib.Path(tmp) / "bundle", protocol)
            path = bundle.root / "robustness/results.json"
            sidecar = path.with_suffix(".json.sha256")
            payload = bundle.verify_json("robustness/results.json")
            payload.pop("artifact_hash", None)
            payload["outputs"][0]["result"] = {}
            path.unlink()
            sidecar.unlink()
            bundle.seal_json("robustness/results.json", payload)
            with self.assertRaises(ProtocolError):
                assert_holdout_ready(protocol, bundle=bundle, confirm_holdout=True)

    def test_incomplete_uncertainty_result_cannot_unlock_holdout(self) -> None:
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._ready_bundle(pathlib.Path(tmp) / "bundle", protocol)
            path = bundle.root / "uncertainty/weekly_bootstrap.json"
            sidecar = path.with_suffix(".json.sha256")
            payload = bundle.verify_json("uncertainty/weekly_bootstrap.json")
            payload.pop("artifact_hash", None)
            payload["statistics"]["mean_weekly_net_return"].pop("percentile_2_5")
            path.unlink()
            sidecar.unlink()
            bundle.seal_json("uncertainty/weekly_bootstrap.json", payload)
            with self.assertRaises(HoldoutLockedError):
                assert_holdout_ready(protocol, bundle=bundle, confirm_holdout=True)

    def test_production_holdout_rejects_noncanonical_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = ExperimentBundle.create(pathlib.Path(tmp) / "custom-run")
            executor = ProtocolExecutor(protocol=_protocol(), bundle=bundle)
            with patch.object(ProtocolExecutor, "_verify_identity", return_value=None):
                with self.assertRaises(HoldoutLockedError):
                    executor.holdout(confirm_holdout=True)

    def test_evaluation_artifact_audit_reconciles_equity_weeks_and_costs(self) -> None:
        protocol = _protocol()
        start = "2021-05-31T00:00:00Z"
        end = "2021-06-07T00:00:00Z"
        result = evaluate_candidate_window(
            _hourly("2021-05-24T00:00:00Z", end),
            candidate=generate_candidates(protocol)[0],
            start_inclusive=start,
            end_exclusive=end,
        )
        artifact = evaluation_artifact(result)
        checks = audit_evaluation_artifact(
            artifact,
            start_inclusive=start,
            end_exclusive=end,
            max_leverage=1.0,
        )
        self.assertIn("equity_curve_reconciled", checks)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = ExperimentBundle.create(pathlib.Path(tmp) / "bundle")
            bundle.seal_json("development/baseline.json", artifact, embed_hash=True)
            self.assertEqual(
                bundle.verify_json("development/baseline.json")["metrics"]["candidate_id"],
                result["metrics"]["candidate_id"],
            )
        broken = copy.deepcopy(artifact)
        broken["accounting"]["final_cash"] += 1.0
        with self.assertRaises(ProtocolError):
            audit_evaluation_artifact(
                broken,
                start_inclusive=start,
                end_exclusive=end,
                max_leverage=1.0,
            )

    def test_holdout_container_audits_both_frozen_configurations(self) -> None:
        protocol = _protocol()
        start = protocol["partitions"]["holdout"]["start_inclusive"]
        end = protocol["partitions"]["holdout"]["end_exclusive"]
        context_start = (
            pd.Timestamp(start) - pd.Timedelta(weeks=1)
        ).isoformat()
        frame = _hourly(context_start, end)
        selected = generate_candidates(protocol)[0]
        baseline = {
            "candidate_id": "fixed_baseline",
            "target_plan_id": "midpoint_half_then_opposite",
            "config": dict(protocol["baseline"]),
        }
        final_candidate = {
            "candidate_id": selected["candidate_id"],
            "configuration": selected["config"],
        }
        artifact = {
            "stage": "holdout",
            "allowed_configurations": [
                "fixed_baseline",
                "single_final_selected_candidate",
            ],
            "baseline": evaluation_artifact(
                evaluate_candidate_window(
                    frame,
                    candidate=baseline,
                    start_inclusive=start,
                    end_exclusive=end,
                )
            ),
            "selected": evaluation_artifact(
                evaluate_candidate_window(
                    frame,
                    candidate=selected,
                    start_inclusive=start,
                    end_exclusive=end,
                )
            ),
        }
        audit = audit_holdout_results(
            protocol, artifact=artifact, final_candidate=final_candidate
        )
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(len(artifact["selected"]["weekly_returns"]), 52)


class CostUncertaintyAndRobustnessTests(unittest.TestCase):
    def test_slippage_is_reported_without_double_counting_partial_terminal_trade(self) -> None:
        data = _featured(
            [
                {"open": 95.0, "high": 96.0, "low": 89.0, "close": 91.0},
                {"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.0},
            ]
        )
        output, trades = backtest_sweep_fade(
            data,
            risk_per_trade=100.0,
            max_leverage=10.0,
            tp1_close_fraction=0.5,
            fee_bps=10.0,
            slippage_bps=5.0,
        )
        trade = trades.iloc[0]
        self.assertGreater(float(trade["entry_slippage_cost"]), 0.0)
        self.assertGreater(float(trade["exit_slippage_cost"]), 0.0)
        self.assertAlmostEqual(float(trade["total_slippage_cost"]), float(trade["entry_slippage_cost"] + trade["exit_slippage_cost"]))
        self.assertAlmostEqual(float(trade["combined_execution_cost"]), float(trade["fees_paid"] + trade["total_slippage_cost"]))
        self.assertAlmostEqual(float(trade["net_pnl"]), float(trade["gross_pnl"] - trade["fees_paid"]))
        self.assertAlmostEqual(float(output["slippage_cost"].sum()), float(trade["total_slippage_cost"]))
        summary = build_summary(
            strategy="current_monday_range",
            signal_count=1,
            exit_style="partial_tp2",
            single_target_level="tp2",
            symbol="SYNTHETIC",
            interval="1h",
            initial_cash=10_000.0,
            df_out=output,
            trades=trades,
            start_used=None,
            end_used=None,
        )
        self.assertAlmostEqual(summary["total_slippage_cost"], float(trade["total_slippage_cost"]))
        self.assertAlmostEqual(summary["combined_execution_cost"], summary["total_fees_paid"] + summary["total_slippage_cost"])

    def test_short_slippage_cost_is_positive_and_metadata_is_explicit(self) -> None:
        data = _featured(
            [
                {"open": 105.0, "high": 111.0, "low": 104.0, "close": 109.0},
                {"open": 100.0, "high": 105.0, "low": 99.0, "close": 100.0},
            ]
        )
        output, trades = backtest_sweep_fade(
            data, risk_per_trade=100.0, max_leverage=10.0, slippage_bps=5.0
        )
        self.assertEqual(trades.iloc[0]["side"], "SHORT")
        self.assertGreater(float(trades.iloc[0]["total_slippage_cost"]), 0.0)
        self.assertEqual(output.attrs["short_position_interpretation"], "synthetic_research_position_on_binance_spot_price_series")

    def test_bootstrap_is_deterministic(self) -> None:
        returns = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.005, 0.01]
        trade_groups = [[value / 2.0] for value in returns]
        first = moving_block_bootstrap(
            returns,
            weekly_trade_returns=trade_groups,
            replications=200,
            seed=123,
        )
        second = moving_block_bootstrap(
            returns,
            weekly_trade_returns=trade_groups,
            replications=200,
            seed=123,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["block_length_weeks"], 4)
        self.assertAlmostEqual(first["point_estimate"], sum(returns) / len(returns))
        self.assertEqual(
            set(first["statistics"]),
            {
                "mean_weekly_net_return",
                "annualized_net_return",
                "maximum_drawdown",
                "average_trade_net_return",
            },
        )
        self.assertAlmostEqual(
            first["statistics"]["average_trade_net_return"]["point_estimate"],
            sum(returns) / len(returns) / 2.0,
        )

    def test_bootstrap_rejects_nonfinite_or_unordered_weekly_units(self) -> None:
        with self.assertRaises(ProtocolError):
            moving_block_bootstrap([0.0, 0.1, math.nan, 0.2])
        artifact = {
            "weekly_returns": [
                {"week_start_utc": "2024-01-08T00:00:00Z", "net_return": 0.01},
                {"week_start_utc": "2024-01-01T00:00:00Z", "net_return": 0.02},
                {"week_start_utc": "2024-01-15T00:00:00Z", "net_return": 0.00},
                {"week_start_utc": "2024-01-22T00:00:00Z", "net_return": -0.01},
            ],
            "trades": [],
        }
        settings = {"block_length_weeks": 4, "replications": 10, "random_seed": 1}
        with self.assertRaises(ProtocolError):
            bootstrap_from_evaluation_artifact(artifact, settings=settings)

    def test_genuine_intraweek_gap_does_not_remove_primary_week(self) -> None:
        frame = _hourly("2021-08-09", "2021-08-16")
        frame = frame.drop(pd.Timestamp("2021-08-13T02:00:00Z"))
        equity = pd.Series(range(10_000, 10_000 + len(frame)), index=frame.index, dtype=float)
        weekly = weekly_returns_from_equity(equity)
        self.assertEqual(len(weekly), 1)

    def test_gap_week_exclusion_removes_whole_iso_weeks(self) -> None:
        frame = _hourly("2021-08-09", "2021-08-23")
        gaps = ["2021-08-13T02:00:00Z", "2021-08-13T05:00:00Z"]
        result = exclude_gap_week_rows(frame, gaps)
        self.assertEqual(excluded_gap_weeks(gaps), ["2021-08-09T00:00:00+00:00"])
        self.assertEqual(result.index.min(), pd.Timestamp("2021-08-16T00:00:00Z"))

    def test_robustness_is_diagnostic_and_cannot_change_selection(self) -> None:
        protocol = _protocol()
        selected = generate_candidates(protocol)[0]
        seen = []

        def evaluator(scenario):
            seen.append(scenario["candidate_id"])
            return {"arbitrary_performance": len(seen)}

        result = run_robustness_diagnostics(
            protocol, selected_candidate=selected, evaluator=evaluator
        )
        self.assertFalse(result["selection_changed"])
        self.assertEqual(result["selected_candidate_id"], selected["candidate_id"])
        self.assertTrue(all(candidate_id == selected["candidate_id"] for candidate_id in seen))
        self.assertEqual(len(robustness_plan(protocol, selected)), 11)

    def test_dirty_tree_guard(self) -> None:
        clean_commit = SimpleNamespace(stdout="abc\n")
        dirty_status = SimpleNamespace(stdout=" M file.py\n")
        with patch("src.protocol_v1.subprocess.run", side_effect=[clean_commit, dirty_status]):
            with self.assertRaises(ProtocolError):
                collect_code_provenance(repo_root=".", require_clean=True)


if __name__ == "__main__":
    unittest.main()
