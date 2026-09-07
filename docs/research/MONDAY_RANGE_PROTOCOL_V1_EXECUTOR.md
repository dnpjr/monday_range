# Monday Range Protocol V1 Executor

`run_protocol_v1.py` is the dedicated entrypoint for the frozen Monday Range research workflow. It validates the amended Protocol V1 file against the embedded SHA-256 before doing anything else. It also verifies the canonical dataset against both the protocol and its manifest.

The executor has independent stages:

```text
validate
init
development
walk-forward
audit
final-selection
robustness
uncertainty
holdout
```

There is no `all` stage. `walk-forward` requires the sealed development stage, `final-selection` requires a passed sealed audit, and `holdout` requires `--confirm-holdout` plus every sealed pre-holdout prerequisite.

## Window and leakage rules

Every evaluation interval uses `[start, end)` UTC boundaries. Boundaries must be Monday 00:00 UTC, partitions must be adjacent and non-overlapping, and the three 24-week validation folds must match the protocol exactly.

Feature construction receives the complete preceding ISO week. The executor builds Monday and rolling features over that context plus the evaluation window, but passes explicit entry boundaries into the backtest engine. Context bars cannot open positions. Metrics and weekly returns are then calculated only from rows inside the evaluation interval. Fold 1 therefore uses 2021-05-24 through 2021-05-31 as context and begins accounting on 2021-05-31.

## Candidates and selection

Candidate generation follows protocol-array order and produces stable IDs from:

- stop offsets `0.75R`, `1.00R`, and `1.25R`;
- latest entry day Wednesday or Thursday;
- three frozen target plans.

The Cartesian product contains exactly 18 unique candidates. A training candidate needs 30 completed trades. Validation rows are excluded from ranking.

The training score uses fractional units:

```text
net CAGR
- 0.50 * absolute maximum drawdown
- annualized execution-cost drag
- 0.0005 * annualized turnover multiple
```

Execution-cost drag is `(fees + separately measured slippage cost) / average marked equity / elapsed years`. Turnover is `total traded notional / average marked equity / elapsed years`. Ties resolve by higher Sortino, smaller absolute drawdown, lower turnover, then candidate ID.

## Execution-cost reporting

Slippage remains embedded in executed prices and therefore already affects P&L. The engine additionally records the economic difference between the unslipped reference price and executed price for each quantity. This is reported as entry, exit, and total slippage cost without deducting it again. Combined execution cost is fees plus measured slippage.

Short trades are labelled `synthetic_research_position_on_binance_spot_price_series` in engine output, provenance, final-candidate metadata, and reports. No borrowing or funding model is introduced.

## Artifact layout and seals

Each run uses a new directory under:

```text
reports/experiments/monday_range_protocol_v1/<run_id>/
├── protocol.json
├── dataset_manifest.json
├── code_provenance.json
├── candidate_grid.json
├── development/{candidate_ranking.json,baseline.json}
├── fold_1/{selection.json,validation.json}
├── fold_2/{selection.json,validation.json}
├── fold_3/{selection.json,validation.json}
├── walk_forward_aggregate.json
├── walk_forward_audit.json
├── final_fit/{training_scores.json,final_candidate.json}
├── robustness/results.json
├── uncertainty/weekly_bootstrap.json
├── charts/
└── holdout/{ACCESS.json,canonical_results.json,uncertainty.json,robustness.json}
```

The experiment root also contains one atomic `HOLDOUT_ACCESS.json` registry. It applies across run IDs, so creating another bundle cannot reopen Protocol V1's holdout.

Every JSON artifact has a `.sha256` sidecar. Critical artifacts also embed their semantic payload hash. Existing artifacts cannot be overwritten; a rerun requires a new run ID. Every later stage verifies the files it consumes.

Code provenance records the Git commit, clean/dirty state, test-suite record, Python, NumPy and pandas versions, accounting version, marking convention, gap policy, intrabar policy, and synthetic-short interpretation. Production stages refuse a dirty working tree. Unit tests may initialize development bundles without this restriction.

## Holdout lock

Holdout authorization fails unless all of the following verify:

1. frozen protocol hash;
2. canonical dataset hash;
3. clean recorded code commit and an executor-verified full test run;
4. complete development ranking and separately sealed frozen baseline;
5. all three exact 18-candidate training rankings and their mechanical winners;
6. matching sealed validation results and the passed execution/accounting audit;
7. the stitched 72-week walk-forward artifact and its hashes;
8. final candidate selected over exactly the amended pre-holdout interval;
9. complete pre-holdout robustness and uncertainty artifacts;
10. final candidate protocol, dataset, code and artifact hashes;
11. explicit `--confirm-holdout` action.

Authorization atomically writes the global access registry and the bundle's immutable `holdout/ACCESS.json` before any holdout data is loaded. The global marker remains consumed if execution subsequently fails. Production holdout execution is restricted to the canonical experiment root.

## Uncertainty and robustness

Weekly uncertainty uses a four-week moving-block bootstrap with 10,000 replications and seed `20260905`. Blocks are sampled from complete Monday-to-Monday units. The pre-holdout estimate uses the sealed 72-week aggregate walk-forward validation series. A distinct confirmatory estimate is created from the selected candidate's 52-week holdout series after the canonical holdout result is sealed. Both report point estimates and the 2.5th, 50th and 97.5th percentiles for mean weekly return, annualized return, maximum drawdown, and average trade return. The frozen null rule is `lower bound > 0`.

Pre-holdout robustness uses the full `[2021-05-31, 2025-05-19)` interval and keeps the selected candidate fixed. It produces baseline, 1.5× and 2× costs; conservative and target-first intrabar results; long-only and short-only decompositions; calendar-year and consecutive non-overlapping 24-week breakdowns; all frozen-grid neighbors as diagnostics; and a separate result excluding the three documented gap weeks. It is explicitly diagnostic and in-sample for the final selected candidate. After the canonical holdout result is sealed, only the preregistered cost stresses and target-first sensitivity are run on holdout. Robustness never replaces the selected candidate.

Training evaluations retain metrics only. Full marked-equity artifacts are kept for the frozen baseline, the three validation winners, the stitched walk-forward series, and the two canonical holdout configurations. Repeated robustness artifacts omit hourly curves to keep the sealed bundle compact while retaining metrics, weekly returns, trades, and accounting summaries.
