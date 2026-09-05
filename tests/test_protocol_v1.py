from __future__ import annotations

import copy
import json
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
    assert_holdout_ready,
    authorize_holdout,
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

    def _ready_bundle(self, root: pathlib.Path, protocol: dict) -> ExperimentBundle:
        bundle = ExperimentBundle.create(root)
        bundle.seal_json("protocol.json", {"protocol_sha256": EXPECTED_PROTOCOL_SHA256, "protocol": protocol})
        bundle.seal_json("dataset_manifest.json", {"sha256": protocol["dataset"]["sha256"]})
        bundle.seal_json("code_provenance.json", {"git_commit": "abc", "test_suite_record": "synthetic passed"})
        candidate = generate_candidates(protocol)[0]
        final_metrics = _metrics(candidate["candidate_id"])
        for fold_id in (1, 2, 3):
            bundle.seal_json(
                f"fold_{fold_id}/selection.json",
                {"fold_id": fold_id, "selected_candidate_id": candidate["candidate_id"]},
            )
            bundle.seal_json(
                f"fold_{fold_id}/validation.json",
                {"fold_id": fold_id, "result": {"candidate_id": candidate["candidate_id"]}},
            )
        bundle.seal_json(
            "final_fit/training_scores.json", {"training_scores": [final_metrics]}
        )
        bundle.seal_json(
            "final_fit/final_candidate.json",
            {
                "candidate_id": candidate["candidate_id"],
                "configuration": candidate["config"],
                "training_period": {
                    "start_inclusive": protocol["partitions"]["development"]["start_inclusive"],
                    "end_exclusive": protocol["partitions"]["holdout"]["start_inclusive"],
                },
                "protocol_hash": EXPECTED_PROTOCOL_SHA256,
                "dataset_hash": protocol["dataset"]["sha256"],
                "code_commit": "abc",
                "objective_value": selection_score(final_metrics),
            },
            embed_hash=True,
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
            bundle = self._ready_bundle(pathlib.Path(tmp) / "bundle", protocol)
            marker = authorize_holdout(protocol, bundle=bundle, confirm_holdout=True)
            self.assertEqual(marker["protocol_hash"], EXPECTED_PROTOCOL_SHA256)
            with self.assertRaises(ArtifactSealedError):
                authorize_holdout(protocol, bundle=bundle, confirm_holdout=True)


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
        first = moving_block_bootstrap(returns, replications=200, seed=123)
        second = moving_block_bootstrap(returns, replications=200, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["block_length_weeks"], 4)
        self.assertAlmostEqual(first["point_estimate"], sum(returns) / len(returns))

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
