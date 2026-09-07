# Monday Range Protocol V1 — Final Research Report

**Run ID:** `canonical_20260905_3528492`

**Research state:** Final; holdout opened once and sealed

**Protocol:** `monday_range_protocol_v1`
**Report date:** 2026-09-05

> **Finding.** The selected strategy returned **+1.52%** on the untouched 52-week holdout with a **−2.02%** maximum drawdown. Its estimated mean weekly net return was **+0.0296%**, but the preregistered 95% four-week moving-block bootstrap interval was **−0.0563% to +0.0916%**. Because the interval includes zero, Protocol V1 **does not reject the null hypothesis**. The result also turned slightly negative at 1.5× transaction costs. This study does not provide sufficient evidence for a robust positive expected return.

## 1. Executive summary

This study tested whether BTC tends to mean-revert after moving beyond the completed UTC Monday range and then closing back inside it. The research process was fixed before corrected canonical performance was viewed: one immutable hourly dataset, a realistic mark-to-market backtest, an 18-candidate grid, an expanding-window walk-forward design, a mechanical selection objective, predefined robustness checks, and a final 52-week holdout that could be opened only once.

The results were mixed and, taken together, weak. The initial development winner returned **+6.12%** in sample. The three subsequent validation folds all lost money, producing an aggregate walk-forward return of **−7.16%**. When the same 18 candidates were ranked mechanically over all pre-holdout data, the winning candidate itself had returned **−4.18%**. That candidate then returned **+1.52%** on the untouched holdout, while the frozen baseline returned **−0.62%**.

The positive holdout point estimate is not strong evidence of an edge. The confidence interval around mean weekly return crosses zero, the holdout contains only 52 weekly observational units and 43 trades, and a modest increase in modeled execution costs removes the profit. The correct conclusion is therefore neither that the strategy works nor that it is disproven in every possible setting. Under the frozen Protocol V1 definitions, the evidence was insufficient to establish a reliable positive expected return after costs.

## 2. Hypothesis

The **primary hypothesis** was:

> After BTCUSDT sweeps beyond the completed UTC Monday range and a completed hourly candle closes back inside, entering the fade at the following hourly open produces positive expected weekly net portfolio returns after the frozen transaction-cost model.

The **null hypothesis** was that expected weekly net portfolio return is less than or equal to zero after costs.

The unit of observation was one complete UTC ISO week. At most one position-producing signal was allowed per week. The preregistered decision rule rejected the null only when the lower bound of the 95% four-week moving-block bootstrap interval for mean weekly net return was greater than zero.

## 3. Dataset

The study used one canonical market dataset:

| Field | Canonical value |
|---|---|
| Symbol | BTCUSDT |
| Venue | Binance spot |
| Source | Binance public spot kline endpoint |
| Base interval | 1 hour |
| Timezone | UTC |
| First candle | 2021-05-21 15:00 UTC |
| Last candle | 2026-05-20 14:00 UTC |
| Rows | 43,793 |
| Dataset version | `btcusdt_binance_spot_1h_v1` |
| SHA-256 | `4d541711c323ac82e07bade75a522c0a48776c7d2e4eec7f45d8d9c4e00b41ce` |

The file is chronologically sorted, has unique timestamps, contains completed candles only, and passed numeric and OHLCV validity checks. No missing hours were interpolated or forward-filled.

Seven hours are absent from Binance's authoritative response and remain explicit gaps:

- 2021-08-13 02:00–05:00 UTC;
- 2021-09-29 07:00–08:00 UTC;
- 2023-03-24 13:00 UTC.

Higher-timeframe views are derived from the canonical 1-hour data with UTC-left-anchored, left-closed bars. Open, high, low, close, and volume aggregate as first, maximum, minimum, last, and sum. A higher-timeframe period is dropped if any expected hourly component is absent.

## 4. Strategy

Each UTC ISO week follows the same sequence:

1. Use the completed Monday session to define the week's high and low.
2. From Tuesday through the permitted final entry day, observe whether price trades beyond either boundary.
3. Require a completed hourly candle to close back inside the Monday range.
4. Enter the fade at the next hourly bar's open, after adverse slippage.
5. Exit according to the candidate's frozen stop and target plan, or on the Friday 23:00 UTC bar if the position remains open.

For a long signal, the completed signal candle trades below the Monday low and closes back above it. A short signal is the mirror image at the Monday high. Signal formation uses only the completed prior bar; the following bar supplies the entry price. This timing prevents same-bar look-ahead.

The frozen grid contained exactly 18 candidates:

- stop offsets of 0.75, 1.00, or 1.25 Monday-range widths beyond the swept boundary;
- latest entry day of Wednesday or Thursday;
- one of three target plans: half at the midpoint then the remainder at the opposite boundary, full exit at the midpoint, or full exit at the opposite boundary.

All other strategy and execution settings stayed fixed during selection. A candidate required at least 30 completed training trades.

## 5. Execution and accounting assumptions

The portfolio started with 10,000 quote units. Position risk was 1% of current marked equity, capped at 1.0× gross notional leverage. Open positions were marked to each hourly close, so total equity, drawdown, volatility, Sharpe, Sortino, and return metrics reflect unrealized as well as realized P&L.

The cost and fill model used:

- fees of 10 basis points per fill;
- adverse slippage of 5 basis points per fill, including spread;
- gap-aware execution: when a bar opens through a stop or target, the opening price is used before adverse slippage;
- conservative stop-first precedence when an OHLC bar touches both stop and target;
- normal exit costs on Friday and end-of-data exits;
- one open position and at most one position-producing signal per ISO week.

Cash, open quantity, realized P&L, unrealized P&L, marked position value, and total equity are tracked through time. Every sealed evaluation was audited for interval boundaries, feature-context isolation, terminal accounting, equity reconciliation, execution costs, turnover, leverage constraints, unique weeks, and duplicate trades.

Because the source is Binance **spot** price data, short trades are explicitly interpreted as `synthetic_research_position_on_binance_spot_price_series`. They are research constructs, not evidence that the same positions could have been executed on Binance spot under identical financing and operational assumptions.

## 6. Research design

All intervals are half-open, `[start, end)`, in UTC.

| Evidence stage | Interval | Weeks | Purpose |
|---|---:|---:|---|
| Context only | 2021-05-24 → 2021-05-31 | 1 | Build Fold 1 features; no trades or P&L |
| Development | 2021-05-31 → 2024-01-01 | 135 | Initial candidate evaluation and first training window |
| Walk-forward validation | 2024-01-01 → 2025-05-19 | 72 | Three non-overlapping 24-week validation folds |
| Final holdout | 2025-05-19 → 2026-05-18 | 52 | One-time untouched final evaluation |

The walk-forward procedure used expanding training windows. For each fold, all 18 candidates were scored on training data only. The training winner was sealed before it was evaluated on that fold's validation data. Validation results could not affect selection. After all three folds and their audits were sealed, the same grid was ranked once over all pre-holdout data to choose the final candidate.

The frozen score, in fractional units, was:

```text
score = net CAGR
        − 0.50 × |maximum drawdown|
        − annualized cost drag
        − 0.0005 × annualized turnover
```

Ties were resolved by higher Sortino, smaller absolute drawdown, lower turnover, then lexicographically smaller candidate ID.

The development-start amendment from 2021-05-24 to 2021-05-31 was made before corrected canonical performance was generated or viewed. It supplied the complete preceding ISO week required for feature context while leaving validation, holdout, candidate, cost, and decision rules unchanged. The pre-amendment protocol hash is retained in the protocol history.

## 7. Development results

Development results are **in sample** and are included only to document the selection process. They are not out-of-sample evidence.

All 18 candidates met the 30-trade eligibility requirement. The development winner was `mr1_stop100_day3_full_at_opposite_boundary`: a 1.00-range stop, entries through Thursday, and a full exit at the opposite boundary.

| Development configuration | Return | CAGR | Max drawdown | Trades | Profit factor | Frozen score |
|---|---:|---:|---:|---:|---:|---:|
| Development winner | +6.12% | +2.32% | −5.84% | 117 | 1.167 | −0.04892 |
| Frozen baseline | +0.56% | +0.22% | −7.82% | 117 | 1.024 | −0.07969 |
| Eventual final candidate | +0.48% | +0.18% | −7.04% | 107 | 1.027 | −0.06464 |

Even the best development score was negative because the frozen objective explicitly penalized drawdown, cost drag, and turnover. The eventual final candidate ranked fifth in this initial window; it emerged later from the mechanically expanded pre-holdout fit.

## 8. Walk-forward validation

The walk-forward evidence was consistently weaker than the early development result.

| Fold | Training winner | Selected parameters | Validation return | Max drawdown | Trades | Profit factor |
|---:|---|---|---:|---:|---:|---:|
| 1 | `mr1_stop100_day3_full_at_opposite_boundary` | 1.00R; through Thu; opposite boundary | −0.87% | −4.18% | 21 | 0.876 |
| 2 | `mr1_stop125_day2_full_at_midpoint` | 1.25R; through Wed; midpoint | −6.18% | −7.04% | 20 | 0.249 |
| 3 | `mr1_stop125_day2_full_at_midpoint` | 1.25R; through Wed; midpoint | −0.17% | −1.23% | 17 | 0.946 |

The selected parameters stabilized in Folds 2 and 3, but parameter stability did not translate into positive validation performance. All three validation returns were negative.

The 72 non-overlapping validation weeks were chained chronologically, without joining them to the development curve:

| Aggregate walk-forward metric | Result |
|---|---:|
| Total return | **−7.16%** |
| Annualized net return | −5.22% |
| Mean weekly net return | −0.1016% |
| Maximum drawdown | −10.59% |
| Trades | 58 |
| Combined execution cost | 341.75 quote units |

The associated four-week moving-block interval for mean weekly return was **−0.2279% to −0.0025%**. This walk-forward bootstrap was diagnostic; the protocol's confirmatory decision was reserved for the untouched holdout.

## 9. Final candidate

After the walk-forward artifacts passed their integrity audit, all 18 candidates were ranked over the complete pre-holdout interval, 2021-05-31 through 2025-05-19. The mechanical winner was:

**`mr1_stop125_day2_full_at_midpoint`**

- strategy: `current_monday_range`;
- stop: 1.25 Monday-range widths beyond the swept boundary;
- eligible entry days: Tuesday and Wednesday;
- target: full exit at the Monday midpoint;
- direction: long and synthetic short;
- risk: 1% of current marked equity;
- maximum leverage: 1.0×;
- fees: 10 bps per fill;
- adverse slippage: 5 bps per fill;
- intrabar policy: conservative stop-first;
- Friday exit: 23:00 UTC.

The final pre-holdout fit was itself negative: **−4.18%** total return, **−1.07%** CAGR, **−10.94%** maximum drawdown, 164 trades, and a 0.865 profit factor. Its frozen score was **−0.096641**. It was selected because it ranked first under the preregistered objective, not because its historical return was positive.

## 10. Untouched holdout

The holdout covered exactly 52 complete ISO weeks from 2025-05-19 through 2026-05-18. It was opened once after protocol, dataset, code, test, artifact, and candidate checks passed. The final candidate was not changed after access.

| Holdout metric | Final candidate |
|---|---:|
| Net total return | **+1.5248%** |
| CAGR | +1.5292% |
| Mean weekly net return | +0.02963% |
| Maximum drawdown | −2.0218% |
| Sharpe | 0.495 |
| Sortino | 0.194 |
| Profit factor | 1.303 |
| Trades | 43 |
| Win rate | 74.42% |
| Mean trade return | +0.01038% |
| Median trade return | +0.63007% |
| Mean P&L per trade | +3.55 quote units |
| Median P&L per trade | +16.91 quote units |
| Time exposed | 15.36% |
| Average gross exposure | 3.14% |
| Annualized turnover | 20.92× |
| Fees | 209.66 quote units |
| Explicit slippage cost | 104.83 quote units |
| Combined execution cost | 314.49 quote units |
| Maximum marked leverage | 0.534× |
| Entries capped by leverage | 0 of 43 |
| Terminal exits | 0 |

The final equity was 10,152.48 quote units, for net profit of 152.48. Combined modeled execution costs were more than twice that net profit, which makes the result economically sensitive to the cost assumptions.

Directionally, 24 long trades contributed **+89.82**, while 19 synthetic short trades contributed **+62.66** quote units. Exit reasons were 33 midpoint targets, 7 Friday exits, and 3 stops. No end-of-data exit was needed.

The preregistered frozen baseline returned **−0.6203%**, with a **−2.7035%** maximum drawdown and a 0.913 profit factor. The selected candidate outperformed that baseline on the holdout, but this comparison does not change the statistical decision rule.

## 11. Robustness

Robustness checks were diagnostics only. They did not trigger reselection or change the final candidate.

### Holdout cost and intrabar sensitivity

| Holdout scenario | Return | Max drawdown | Profit factor | Combined cost |
|---|---:|---:|---:|---:|
| Canonical costs | **+1.5248%** | −2.0218% | 1.303 | 314.49 |
| 1.5× costs | **−0.0580%** | −2.2934% | 0.989 | 464.90 |
| 2× costs | **−1.5955%** | −2.9944% | 0.709 | 611.01 |
| Target-first | +1.5248% | −2.0218% | 1.303 | 314.49 |

The holdout profit disappeared under the 1.5× cost scenario and deteriorated further at 2× costs. Target-first produced the same outcome as conservative stop-first, indicating that same-bar stop/target ordering did not affect this holdout.

### Pre-holdout diagnostics

On the full pre-holdout interval, the final candidate returned **−4.18%** at canonical costs, **−8.46%** at 1.5× costs, and **−12.50%** at 2× costs. Long-only and short-only variants were both negative. The calendar breakdown was positive in 2021 and 2022, then negative in 2023, 2024, and the pre-holdout portion of 2025. The eight complete 24-week subperiods likewise shifted from three positive early periods to five negative later periods.

All 18 frozen-grid candidates had negative returns over the full pre-holdout diagnostic interval. This is a stronger warning about temporal instability than the stability of the Fold 2 and Fold 3 parameter choice is a reassurance.

Removing the three weeks containing the seven documented Binance gaps changed the selected candidate's pre-holdout return from **−4.18%** to **−4.71%** and reduced its trade count from 164 to 161. The missing hours therefore do not explain the weak pre-holdout result. Target-first and conservative stop-first were identical in this interval as well.

## 12. Bootstrap uncertainty

The confirmatory holdout analysis resampled complete weekly returns with four-week moving blocks, 10,000 replications, and seed `20260905`.

| Statistic | Point estimate | 2.5th percentile | Median | 97.5th percentile |
|---|---:|---:|---:|---:|
| Mean weekly net return | +0.02963% | **−0.05634%** | +0.02421% | **+0.09157%** |
| Annualized net return | +1.5248% | −2.9368% | +1.2375% | +4.8567% |
| Average trade net return | +0.01038% | −0.56660% | −0.01028% | +0.42905% |
| Maximum drawdown | −1.8201% | −4.2997% | −1.8102% | −0.8333% |

The maximum-drawdown point estimate in this bootstrap table is calculated from the weekly resampled series, while the canonical portfolio drawdown of **−2.0218%** uses the full hourly marked-equity path.

The lower bound for mean weekly net return is below zero. Therefore, the frozen decision rule is not satisfied and **Protocol V1 does not reject the null hypothesis**.

## 13. Conclusion

The selected Monday Range strategy produced a modest positive return on the untouched holdout, but the preregistered bootstrap criterion was not satisfied and performance deteriorated under moderately higher transaction costs. The walk-forward period was negative, the final pre-holdout fit was negative, and the positive holdout profit was small relative to modeled execution costs.

Protocol V1 therefore does not provide sufficient evidence for a robust positive expected weekly return. The study's useful result is methodological: it turns an intuitive trading idea into a reproducible, falsifiable experiment and preserves an honest negative-to-inconclusive evidence trail without post-holdout tuning.

## 14. Limitations

- **OHLC ambiguity.** Hourly bars cannot reveal the path taken within a candle. Conservative stop-first is the canonical assumption; target-first sensitivity happened to be identical, but this does not eliminate all intrabar uncertainty.
- **Synthetic shorts.** Short positions are evaluated on Binance spot prices without modeling an actual short venue, borrowing, financing, liquidation mechanics, or venue-specific constraints.
- **Simplified execution.** Fixed fees and slippage do not capture order-book depth, latency, variable spread, market impact, rejects, or implementation outages.
- **One asset and one venue.** Results from BTCUSDT on Binance spot do not establish external validity across assets, venues, or market structures.
- **Limited observations.** The confirmatory holdout contains 52 weekly units and 43 trades. Serial dependence is addressed approximately with moving blocks, and the resulting uncertainty interval remains wide.
- **Historical strategy development.** Earlier exploratory work on the Monday Range idea creates some broader data-snooping risk even though Protocol V1's corrected candidate grid and holdout rules were frozen before canonical results were viewed.
- **Regime dependence.** Early subperiods were positive and later pre-holdout subperiods were negative. A positive final year does not resolve that instability.
- **Known exchange gaps.** Seven genuine missing hours are preserved rather than invented. The predefined exclusion check suggests they did not create the main result.
- **Finite candidate grid.** The final choice is the best candidate under one frozen grid and objective, not proof of a globally optimal parameter set.
- **Research, not deployment evidence.** No live execution study, capacity analysis, operational risk assessment, or forward paper-trading validation is part of Protocol V1.

## 15. Reproducibility and provenance

Canonical inputs and identity:

| Item | Value |
|---|---|
| Experiment run | `canonical_20260905_3528492` |
| Protocol hash | `da0d41ce67d445bcb308012bf323d2e573d246a7a1fd3eb57c9ee6bc951d4cf3` |
| Previous protocol hash | `5266a47fdf69e742ec455eb87986a423b3e5ae451757472558cac12d4e04d5ed` |
| Dataset hash | `4d541711c323ac82e07bade75a522c0a48776c7d2e4eec7f45d8d9c4e00b41ce` |
| Code commit | `352849299200b749496d588a1d7d1149503750fc` |
| Accounting version | `marked_equity_v1` |
| Python | 3.11.5 |
| pandas / NumPy | 2.0.3 / 1.24.3 |
| Recorded test suite | 190/190 passed |

Primary files:

- canonical dataset: `data/canonical/btcusdt_binance_spot_1h_v1.csv`;
- data manifest: `data/canonical/btcusdt_binance_spot_1h_v1.manifest.json`;
- frozen protocol: `configs/research/monday_range_protocol_v1.json`;
- human-readable protocol: `docs/research/MONDAY_RANGE_PROTOCOL_V1.md`;
- executor: `run_protocol_v1.py` and `src/protocol_v1.py`;
- sealed bundle: `reports/experiments/monday_range_protocol_v1/canonical_20260905_3528492/`.

Key sealed artifact hashes:

| Artifact | SHA-256 / semantic hash |
|---|---|
| Final candidate | `37e9e7c6d0386791556f5566d257a776ec09c11f1398a17b1d70cd89754097c0` |
| Holdout access marker | `5e4bc2df03939220a0f878bdcb8986c554cd8d2596f741b7c589e954d532d68f` |
| Holdout canonical results | `a2600bce1cd1f34c163ed9ddb570084d686589467fb89d3bbb998e59fa66bc96` |
| Holdout robustness | `3ede4b422a0cac0145d9c111891b3c4d1bd3e24c02cb953e8b60989fe32a7833` |
| Holdout uncertainty | `354b0405179bcb36a690e197bb4d12b57f7946b9ee89cce341a24f0088013eed` |

Each sealed JSON artifact has a `.sha256` sidecar. The walk-forward audit passed before holdout access, and the holdout artifact audit reconciled configuration identity, interval and context boundaries, marked equity, terminal accounting, costs, turnover, leverage, trades, and weekly units. The holdout was opened exactly once. No post-holdout candidate, protocol, dataset, accounting, execution, bootstrap, or robustness definition was changed.

Historical performance outputs created before the corrected engine and canonical data pipeline remain legacy artifacts. They are not evidence for or against the Protocol V1 hypothesis.
