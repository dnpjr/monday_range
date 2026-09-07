#!/usr/bin/env python3
"""Verify frozen inputs and presentation artifacts without running research."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonical_data import DATA_FILE, sha256_file
from src.portfolio_results import (
    EXPECTED_CANDIDATE_ID,
    EXPECTED_DATASET_HASH,
    EXPECTED_PROTOCOL_HASH,
    load_canonical_research,
)
from src.protocol_v1 import load_frozen_protocol, verify_canonical_dataset


def main() -> None:
    protocol = load_frozen_protocol()
    verify_canonical_dataset(protocol)
    research = load_canonical_research()
    if sha256_file(DATA_FILE) != EXPECTED_DATASET_HASH:
        raise RuntimeError("Canonical CSV hash differs from the frozen identity")
    if research.access["protocol_hash"] != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("Presentation bundle protocol identity differs")
    if research.final_candidate["candidate_id"] != EXPECTED_CANDIDATE_ID:
        raise RuntimeError("Presentation bundle candidate identity differs")
    if research.holdout_uncertainty["reject_null"] is not False:
        raise RuntimeError("Canonical statistical conclusion differs")
    print(
        "Verified Protocol V1 release: "
        f"{research.root.name}, {len(research.artifact_hashes)} sealed artifacts, "
        f"candidate {EXPECTED_CANDIDATE_ID}."
    )


if __name__ == "__main__":
    main()
