from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.protocol_v1 import (
    EXPERIMENT_ROOT,
    EXPECTED_PROTOCOL_SHA256,
    ExperimentBundle,
    ProtocolExecutor,
    initialize_experiment_bundle,
    load_frozen_protocol,
    verify_canonical_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute frozen Monday Range Research Protocol V1 stages")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validate", "init", "development", "walk-forward", "final-selection", "robustness", "uncertainty", "holdout"),
    )
    parser.add_argument("--run-id", help="Immutable experiment bundle identifier")
    parser.add_argument("--test-suite-record", help="Passing test version recorded at bundle initialization")
    parser.add_argument("--confirm-holdout", action="store_true", help="Explicitly authorize the one-time holdout stage")
    parser.add_argument("--development-mode", action="store_true", help="Allow bundle initialization from a dirty tree for synthetic development only")
    parser.add_argument("--experiment-root", default=str(EXPERIMENT_ROOT))
    args = parser.parse_args()

    protocol = load_frozen_protocol()
    manifest = verify_canonical_dataset(protocol)
    if args.stage == "validate":
        print(json.dumps({
            "status": "valid",
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "dataset_version": manifest["dataset_version"],
            "dataset_sha256": manifest["sha256"],
        }, indent=2))
        return

    if not args.run_id:
        parser.error("--run-id is required for every stage except validate")
    bundle_path = Path(args.experiment_root) / args.run_id
    if args.stage == "init":
        if not args.test_suite_record:
            parser.error("--test-suite-record is required for init")
        bundle = initialize_experiment_bundle(
            protocol,
            run_id=args.run_id,
            root=args.experiment_root,
            require_clean=not args.development_mode,
            test_suite_record=args.test_suite_record,
        )
        print(bundle.root)
        return

    bundle = ExperimentBundle(bundle_path)
    executor = ProtocolExecutor(protocol=protocol, bundle=bundle)
    actions = {
        "development": executor.development,
        "walk-forward": executor.walk_forward,
        "final-selection": executor.final_selection,
        "robustness": executor.robustness,
        "uncertainty": executor.uncertainty,
        "holdout": lambda: executor.holdout(confirm_holdout=args.confirm_holdout),
    }
    result = actions[args.stage]()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
