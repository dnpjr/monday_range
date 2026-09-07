import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.portfolio_results import (
    EXPECTED_ARTIFACT_HASHES,
    EXPECTED_CANDIDATE_ID,
    EXPECTED_DATASET_HASH,
    EXPECTED_PROTOCOL_HASH,
    RUN_ID,
    ResearchArtifactError,
    build_exploratory_candidate,
    equity_frame,
    load_canonical_research,
    run_exploratory_backtest,
    verify_sealed_json,
    walk_forward_table,
    weekly_return_frame,
)
from src.canonical_data import DATA_FILE, sha256_file


class PortfolioResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.research = load_canonical_research()

    def test_loads_expected_canonical_identity_and_outcome(self):
        research = self.research
        self.assertEqual(research.root.name, RUN_ID)
        self.assertEqual(research.access["candidate_id"], EXPECTED_CANDIDATE_ID)
        self.assertEqual(research.access["protocol_hash"], EXPECTED_PROTOCOL_HASH)
        self.assertEqual(research.access["dataset_hash"], EXPECTED_DATASET_HASH)
        metrics = research.selected["metrics"]
        self.assertAlmostEqual(metrics["total_return"], 0.015247909821012318)
        self.assertFalse(research.holdout_uncertainty["reject_null"])
        self.assertEqual(sha256_file(DATA_FILE), EXPECTED_DATASET_HASH)

    def test_verifies_every_required_bundle_artifact(self):
        # 22 bundle JSON artifacts plus the protocol-wide access marker.
        self.assertEqual(self.research.artifact_hashes, EXPECTED_ARTIFACT_HASHES)

    def test_tampered_artifact_is_rejected(self):
        source = self.research.root / "final_fit" / "final_candidate.json"
        source_sidecar = source.with_suffix(".json.sha256")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "artifact.json"
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            target.with_suffix(".json.sha256").write_text(
                source_sidecar.read_text(encoding="ascii"), encoding="ascii"
            )
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["candidate_id"] = "tampered"
            target.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ResearchArtifactError):
                verify_sealed_json(target)

    def test_missing_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "artifact.json"
            target.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ResearchArtifactError):
                verify_sealed_json(target)

    def test_resigned_noncanonical_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol_root = Path(tmp) / "monday_range_protocol_v1"
            run_root = protocol_root / RUN_ID
            shutil.copytree(self.research.root, run_root)
            for suffix in ("", ".sha256"):
                shutil.copy2(
                    self.research.root.parent / f"HOLDOUT_ACCESS.json{suffix}",
                    protocol_root / f"HOLDOUT_ACCESS.json{suffix}",
                )
            target = run_root / "development" / "baseline.json"
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["metrics"]["total_return"] = 99.0
            payload_without_hash = dict(payload)
            payload_without_hash.pop("artifact_hash", None)
            digest = hashlib.sha256(
                json.dumps(
                    payload_without_hash,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            payload["artifact_hash"] = digest
            target.write_text(json.dumps(payload), encoding="utf-8")
            target.with_suffix(".json.sha256").write_text(digest + "\n", encoding="ascii")
            with self.assertRaises(ResearchArtifactError):
                load_canonical_research(run_root=run_root, protocol_root=protocol_root)

    def test_malformed_sidecar_is_rejected_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "artifact.json"
            target.write_text("{}\n", encoding="utf-8")
            target.with_suffix(".json.sha256").write_bytes(b"\xff")
            with self.assertRaises(ResearchArtifactError):
                verify_sealed_json(target)

    def test_loader_is_independent_of_current_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                research = load_canonical_research()
            finally:
                os.chdir(original)
        self.assertEqual(research.root.name, RUN_ID)

    def test_result_frames_are_sorted_and_complete(self):
        curve = equity_frame(self.research.selected)
        weekly = weekly_return_frame(self.research.selected)
        self.assertEqual(len(curve), 8736)
        self.assertEqual(len(weekly), 52)
        self.assertTrue(curve["timestamp"].is_monotonic_increasing)
        self.assertTrue(weekly["week_start_utc"].is_monotonic_increasing)
        self.assertEqual(str(curve["timestamp"].dt.tz), "UTC")

    def test_walk_forward_table_uses_three_sealed_folds(self):
        table = walk_forward_table(self.research)
        self.assertEqual(table["Fold"].tolist(), [1, 2, 3])
        self.assertEqual(table["Trades"].sum(), 58)
        self.assertEqual(table.iloc[1]["Training winner"], EXPECTED_CANDIDATE_ID)

    def test_explore_candidate_defaults_can_match_final_candidate(self):
        candidate = build_exploratory_candidate(
            stop_offset=1.25,
            latest_entry_day=2,
            target_plan="full_at_midpoint",
            fee_bps=10.0,
            slippage_bps=5.0,
            intrabar_policy="conservative_stop_first",
            direction="both",
            protocol=self.research.protocol,
        )
        self.assertEqual(candidate["candidate_id"], "explore__" + EXPECTED_CANDIDATE_ID)
        self.assertEqual(candidate["config"]["risk_fraction"], 0.01)
        self.assertEqual(candidate["config"]["max_leverage"], 1.0)

    def test_explore_is_in_memory_and_has_no_output_path(self):
        dummy = pd.DataFrame(index=pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"))
        expected = {"metrics": {"total_return": 0.0}}
        with patch("src.portfolio_results.evaluate_candidate_window", return_value=expected) as evaluator:
            result = run_exploratory_backtest(
                start_inclusive="2024-01-01T00:00:00Z",
                end_exclusive="2024-01-08T00:00:00Z",
                stop_offset=1.25,
                latest_entry_day=2,
                target_plan="full_at_midpoint",
                fee_bps=10.0,
                slippage_bps=5.0,
                intrabar_policy="conservative_stop_first",
                direction="both",
                ohlc=dummy,
                protocol=self.research.protocol,
            )
        self.assertIs(result, expected)
        self.assertNotIn("output_path", evaluator.call_args.kwargs)
        self.assertNotIn("bundle", evaluator.call_args.kwargs)

    def test_explore_cannot_overlap_consumed_holdout(self):
        with patch("src.portfolio_results.evaluate_candidate_window") as evaluator:
            with self.assertRaisesRegex(ValueError, "pre-holdout"):
                run_exploratory_backtest(
                    start_inclusive="2025-05-12T00:00:00Z",
                    end_exclusive="2025-05-26T00:00:00Z",
                    stop_offset=1.25,
                    latest_entry_day=2,
                    target_plan="full_at_midpoint",
                    fee_bps=10.0,
                    slippage_bps=5.0,
                    intrabar_policy="conservative_stop_first",
                    direction="both",
                    ohlc=pd.DataFrame(),
                    protocol=self.research.protocol,
                )
        evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
