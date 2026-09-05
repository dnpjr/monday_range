# Monday Range Protocol V1 Executor

`run_protocol_v1.py` is the dedicated entrypoint for the frozen Monday Range research workflow. It validates the amended Protocol V1 file against the embedded SHA-256 before doing anything else. It also verifies the canonical dataset against both the protocol and its manifest.

The executor has independent stages:

```text
validate
init
development
walk-forward
final-selection
robustness
uncertainty
holdout
```

There is no `all` stage. `holdout` additionally requires `--confirm-holdout` and all sealed prerequisites.

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
├── development.json
├── fold_1/{selection.json,validation.json}
├── fold_2/{selection.json,validation.json}
├── fold_3/{selection.json,validation.json}
├── final_fit/{training_scores.json,final_candidate.json}
├── robustness/results.json
├── uncertainty/weekly_bootstrap.json
├── charts/
└── holdout/{ACCESS.json,canonical_results.json}
```

Every JSON artifact has a `.sha256` sidecar. Critical artifacts also embed their semantic payload hash. Existing artifacts cannot be overwritten; a rerun requires a new run ID. Every later stage verifies the files it consumes.

Code provenance records the Git commit, clean/dirty state, test-suite record, Python, NumPy and pandas versions, accounting version, marking convention, gap policy, intrabar policy, and synthetic-short interpretation. Production stages refuse a dirty working tree. Unit tests may initialize development bundles without this restriction.

## Holdout lock

Holdout authorization fails unless all of the following verify:

1. frozen protocol hash;
2. canonical dataset hash;
3. code commit and passing test record;
4. all three sealed training selections;
5. matching sealed validation results;
6. final candidate selected over exactly the amended pre-holdout interval;
7. final candidate protocol, dataset, code and artifact hashes;
8. explicit `--confirm-holdout` action.

Authorization writes the immutable `holdout/ACCESS.json` marker before evaluation. A failed or repeated run cannot overwrite it and must use a new experiment bundle.

## Uncertainty and robustness

Weekly uncertainty uses a four-week moving-block bootstrap with 10,000 replications and seed `20260905`. Blocks are sampled from complete Monday-to-Monday units. Genuine missing Binance hours inside a week do not remove that week from the primary series. The report contains the point estimate, 2.5th, 50th and 97.5th percentiles, and the frozen `lower bound > 0` null rule.

Robustness keeps the selected candidate fixed. It produces baseline, 1.5× and 2× costs; conservative and target-first intrabar results; long-only and short-only decompositions; calendar-year and non-overlapping 24-week breakdowns; all frozen-grid neighbors as diagnostics; and a separate result excluding the three documented gap weeks. These outputs never call the selection function or replace the final candidate.

The implementation and test task must use synthetic fixtures. Running `development`, `walk-forward`, `final-selection`, `robustness`, `uncertainty`, or `holdout` against the canonical loader is a research execution and belongs to the separately authorized experiment task.
