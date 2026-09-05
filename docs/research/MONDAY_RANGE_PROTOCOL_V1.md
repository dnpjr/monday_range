# Monday Range Research Protocol V1

Status: **frozen before corrected strategy results are viewed**.

This document preregisters the first canonical evaluation of the Monday Range strategy. It uses canonical BTCUSDT data and the corrected engine, but it does not contain or rely on strategy performance. The machine-readable source of truth is `configs/research/monday_range_protocol_v1.json`.

## Research question and hypotheses

The primary hypothesis is that, after BTCUSDT trades beyond the completed UTC Monday range and a completed 1h candle closes back inside, entering the fade at the next hourly open produces positive expected weekly net portfolio return after the frozen transaction-cost model.

The null hypothesis is that the expected weekly net portfolio return is less than or equal to zero after costs.

The primary unit of observation is one complete UTC ISO week. A long signal occurs when a completed candle's low is below the Monday low and its close is above the Monday low. A short signal occurs when its high is above the Monday high and its close is below the Monday high. If both conditions occur on the same candle, the implementation's fixed LONG precedence applies. A trade is the first eligible signal in the week that produces a non-zero position; entry occurs at the following bar's open after adverse slippage. At most one trade is allowed per ISO week.

The first evaluation uses BTCUSDT, Binance spot klines, 1h bars, and UTC Monday boundaries. Binance spot is the price source. The engine simulates collateralized long and short exposure rather than literal spot inventory; short-side deployment claims therefore require a later venue and borrow-cost specification.

## Dataset and frozen partitions

The dataset is `btcusdt_binance_spot_1h_v1`, SHA-256 `4d541711c323ac82e07bade75a522c0a48776c7d2e4eec7f45d8d9c4e00b41ce`.

All boundaries are UTC, Monday 00:00, and interpreted as `[start, end)`:

| Partition | Start inclusive | End exclusive | Complete ISO weeks | Purpose |
|---|---:|---:|---:|---|
| Development | 2021-05-24 00:00 | 2024-01-01 00:00 | 136 | Exploration and training |
| Validation | 2024-01-01 00:00 | 2025-05-19 00:00 | 72 | Chronological selection-method validation |
| Final holdout | 2025-05-19 00:00 | 2026-05-18 00:00 | 52 | One final confirmatory use |

The partial opening interval from 2021-05-21 15:00 through 2021-05-24 00:00 and the partial final week from 2026-05-18 00:00 through the dataset end are excluded. The holdout boundary was chosen from coverage and calendar structure without viewing corrected outcomes.

The seven canonical exchange gaps remain unfilled. Any affected week remains in the primary evaluation because the next available open is the first observable executable price. Results excluding the three affected ISO weeks will be a labelled data-quality sensitivity, never the canonical result.

## Prior exploration and fixed degrees of freedom

Legacy configuration files show that earlier work exposed or tried risk fractions, fee/slippage levels, 1h and 4h intervals, target distances, partial-exit fractions, stop offsets, Friday hours, entry cutoffs, range filters, SMA filtering, direction, breakeven behavior, and single-target exits. Examples include stop offsets from 0.5 to 3.0 Monday ranges, target fractions from 0.35 to 1.0, and fee/slippage settings from zero through non-zero scenarios. These values are inventory only; no legacy performance value was consulted.

The following remain fixed for the first evaluation because they define the hypothesis or its accounting: BTCUSDT 1h; `current_monday_range`; completed-previous-bar signal; next-open entry; both directions; one trade per week; marked equity; current-equity risk; 1x maximum leverage; conservative stop-first ordering; gap-aware fills; Friday liquidation; no SMA; no range filter; and the canonical data gaps.

## Frozen baseline strategy

The baseline uses the current strategy implementation and older simple defaults:

- Eligible entry bars: Tuesday through Thursday UTC, as enforced by `current_monday_range`; no additional hourly cutoff.
- Direction: both LONG and SHORT.
- Stop: `swept_boundary_offset`, one full Monday-range width beyond the swept boundary.
- TP1: Monday midpoint, 0.5 range from the swept boundary; close 50%.
- TP2: opposite Monday boundary, 1.0 range from the swept boundary; close the remainder.
- After TP1: move stop to entry price; fees are not added to the breakeven level.
- Friday exit: the bar containing 23:00 UTC; use its close with normal costs.
- Terminal exit: `end_of_data` with normal costs.
- Risk: 1% of current marked equity per trade.
- Maximum gross leverage: 1.0x current equity.
- Fees: 10 bps per fill.
- Slippage: 5 bps per fill, adverse on every entry and exit; spread is included in this allowance.
- Intrabar policy: `conservative_stop_first`.
- Initial capital: 10,000 quote units; this is a scale choice rather than a tuned parameter.

Hypothesis-defined fields are the signal, Tuesday-through-Thursday eligibility, both directions, one-trade-per-week rule, Monday-relative stop/targets, and Friday closure. Execution assumptions are next-open/gap fills, fees, slippage, intrabar ordering, current-equity sizing, leverage, terminal closure, and bar-close marking. Only the three fields below may vary during selection.

## Parameter search space

The grid has exactly 18 candidates: 3 stop offsets × 2 latest entry days × 3 target plans.

| Parameter | Allowed values | Role and reason |
|---|---|---|
| Stop offset beyond swept boundary | 0.75R, 1.00R, 1.25R | Structural tolerance around the baseline; bounded and symmetric |
| Latest entry day | Wednesday (`2`), Thursday (`3`) | Structural horizon test within the existing Tue-Thu rule |
| Target plan | 50% at midpoint then remainder at opposite boundary; full exit at midpoint; full exit at opposite boundary | Three interpretable expressions of the mean-reversion hypothesis |

Fees, slippage, risk fraction, risk base, leverage, intrabar policy, gap behavior, accounting, direction, Friday cutoff, SMA, and range filters are not optimized. No candidate may be added after corrected results are viewed.

## Walk-forward method

Use expanding training windows with three non-overlapping 24-week validation windows and a 24-week step:

| Fold | Training `[start, end)` | Validation `[start, end)` |
|---|---|---|
| 1 | 2021-05-24 → 2024-01-01 | 2024-01-01 → 2024-06-17 |
| 2 | 2021-05-24 → 2024-06-17 | 2024-06-17 → 2024-12-02 |
| 3 | 2021-05-24 → 2024-12-02 | 2024-12-02 → 2025-05-19 |

Each fold ranks the same 18 candidates on training data only and sends exactly the top candidate into the next validation window. A candidate needs at least 30 completed training trades. If no candidate qualifies or any validation fold cannot complete, stop without opening the holdout.

After the three fold artifacts are sealed, perform one final fit on all pre-holdout data, 2021-05-24 through 2025-05-19, using the same grid and objective. That produces exactly one selected candidate for the holdout. Earlier validation periods may enter later expanding training windows; no future validation or holdout bar may enter an earlier training calculation.

Features must be computed with canonical history preceding each fold. Supply at least one complete prior ISO week, reset cash and positions at the fold start, and permit signals and entries only within the fold. Boundaries are Monday-aligned, so a trade or Monday range is never split across folds.

## Selection objective

Candidates below 30 completed training trades are ineligible. For all others, use fractions rather than percentage points:

```text
score = net_CAGR
        - 0.50 × abs(max_drawdown)
        - annualized_cost_drag_fraction
        - 0.0005 × annualized_turnover_multiple
```

`annualized_cost_drag_fraction = (fees + explicit slippage cost) / average marked equity / elapsed years`.

`annualized_turnover_multiple = total traded notional / average marked equity / elapsed years`.

The score rewards compounding while penalizing drawdown, modeled cost drag, and turnover. Ties resolve by higher Sortino, smaller absolute drawdown, lower turnover, then stable candidate ID. The existing `sweep_score` is not used because its weights, eight-trade threshold, and trade-count turnover proxy do not match this protocol.

## Transaction costs and execution sensitivities

The baseline assumes 10 bps fee and 5 bps adverse slippage per fill. Entries and protective exits are marketable, so maker rebates are not assumed; targets receive the same conservative treatment. Spread is included in slippage. Perpetual funding is irrelevant because the source is spot. The simulated short leg does not yet model margin borrow, so the first report must label its execution interpretation and show long-only and short-only decompositions.

Pre-registered cost sensitivities are:

| Scenario | Fee per fill | Slippage per fill |
|---|---:|---:|
| Baseline | 10 bps | 5 bps |
| 1.5× | 15 bps | 7.5 bps |
| 2× | 20 bps | 10 bps |

The canonical intrabar result is conservative stop-first. `target_first` is an optimistic sensitivity and must be reported separately. The gap convention remains fixed: use the bar open when it opens through a trigger, then apply adverse slippage. Optimistic and conservative results are never averaged.

## Metrics fixed before evaluation

Primary evidence consists of mean weekly net portfolio return with its 95% block-bootstrap interval, net CAGR, and maximum drawdown. Evidence against the null requires the lower 95% bootstrap bound for mean weekly net portfolio return to exceed zero; all other outcomes are reported without redefining success.

Secondary portfolio metrics are Sortino, Sharpe, profit factor, total return, annualized volatility, gross and net exposure, annualized turnover, total fees, explicit slippage cost, maximum leverage used, and the fraction of entries capped by leverage.

Trade statistics are count, win rate, mean and median net trade return, mean and median P&L, mean and median holding time, exit-reason distribution, and long/short decomposition.

Required robustness outputs are the three validation folds; calendar-year and 24-week subperiods; parameter-neighborhood stability within the frozen grid; long-only and short-only decompositions; 1.5× and 2× cost stress; `target_first` sensitivity; and exclusion of the three weeks containing canonical gaps. These are diagnostics, not alternative winners.

Optional follow-up work, after the first experiment is sealed, may examine deterministic 4h bars and ETHUSDT as an external market. Neither belongs to V1 selection or holdout testing.

## Statistical uncertainty

Build weekly returns from marked equity over complete Monday-to-Monday UTC weeks. Use a moving-block bootstrap of four consecutive ISO weeks, 10,000 replications, and seed `20260905`. Report 2.5th, 50th, and 97.5th percentiles for mean weekly net return, annualized net return, maximum drawdown, and average trade net return. Candle-level or independently shuffled trade resampling is prohibited because it discards temporal dependence.

## Holdout and multiple-testing discipline

Legacy exploratory results are not confirmatory. The holdout stays inaccessible until the protocol executor and its tests are committed, all walk-forward artifacts are sealed, and the final candidate ID is written to immutable pre-holdout metadata.

The holdout bundle may run only the preregistered fixed baseline and the one selected candidate. The selected candidate is the primary configuration. Pre-registered cost and intrabar sensitivities may be produced only after the canonical holdout result is sealed and cannot alter it.

After first holdout access, no parameter, objective, partition, cost, execution, or reporting rule may change. If methodology changes, the current holdout becomes development data and a new untouched future period is required. Statistical intervals will be described as uncertainty estimates rather than proof of independence or universal profitability.

## Reproducible experiment bundle

Each run must create a versioned directory containing protocol version and config SHA-256, dataset version/hash, code commit, exact partitions, baseline, grid, objective, costs, folds, chosen candidate, all OOS metrics, weekly returns, trades, equity, robustness and bootstrap outputs, charts, and UTC run timestamp. Pre-holdout and holdout bundles must be distinct and immutable.

## Existing-code readiness

| Area | Status | Required action before evaluation |
|---|---|---|
| Canonical loading, hashing, accounting, fills, and strategy signal timing | Ready | Use the canonical loader and current corrected engine |
| Baseline parameters and 18 candidate values | Ready | Existing engine fields can express them |
| Exact `[start, end)` partition filtering | Small fix required | Add protocol parsing/validation and timestamp-based partition guards |
| Holdout isolation | Small fix required | Default-deny holdout access and write an irreversible access marker into the bundle |
| Fold context | Substantial fix required | Compute features with preceding history, reset at fold start, and gate entries to the fold; current `_window_features` slices before feature construction |
| Walk-forward scheme | Substantial fix required | Replace hard-coded calendar-year folds; current runner supports only `sweep_retest`, while V1 uses `current_monday_range` and 24-week expanding folds |
| Selection objective | Substantial fix required | Implement the frozen formula, minimum-trade gate, deterministic tie-breaks, and explicit slippage-cost accounting |
| Final pre-holdout fit | Substantial fix required | Add the single frozen-grid fit and persist the selected candidate before unlocking holdout |
| Weekly uncertainty and required robustness | Substantial fix required | Persist weekly equity/trade data and implement block bootstrap, subperiod, decomposition, and sensitivity outputs |
| Experiment provenance | Small fix required | Add protocol version/config hash, chosen candidate, fold artifacts, and charts to the existing dataset/code metadata |
| Short-side execution interpretation | Small fix required | Label the synthetic short leg and either add a fixed borrow-cost model before deployment claims or keep conclusions at signal-research level |

The implementation task must not run the strategy. It should add a protocol executor, guards, metrics, bundle persistence, and synthetic tests so the next task can execute the preregistered development/validation workflow without touching the holdout.
