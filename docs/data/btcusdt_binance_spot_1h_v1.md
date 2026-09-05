# BTCUSDT Binance spot 1h canonical dataset v1

## Policy and coverage

- Dataset version: `btcusdt_binance_spot_1h_v1`
- Authoritative file: `data/canonical/btcusdt_binance_spot_1h_v1.csv`
- Source: Binance spot public kline endpoint, `https://api.binance.com/api/v3/klines`
- Symbol / interval / timezone: `BTCUSDT` / `1h` / `UTC`
- Frozen window: `2021-05-21T15:00:00+00:00` through `2026-05-20T14:00:00+00:00` (end exclusive `2026-05-20T15:00:00+00:00`)
- Boundary rationale: preserve the established Monday cache timestamp window. The final 14:00 candle was re-fetched after completion; no newer candle was added.
- Rows: `43793`
- SHA-256: `4d541711c323ac82e07bade75a522c0a48776c7d2e4eec7f45d8d9c4e00b41ce`
- Retrieval timestamp: `2026-09-05T16:32:18.496470+00:00`

Only completed candles are admitted. Timestamps are sorted and unique. No missing candle is forward-filled or interpolated.

## Data quality

- Duplicate source timestamps removed: `0`
- Incomplete candles removed from the frozen source response: `0`
- Missing expected hours: `7`
- OHLCV validation passed: `True`
- Invalid numeric rows: `0`
- Invalid OHLC/volume rows: `0`

The validation requires high >= open/close/low, low <= open/close, high >= low, non-negative volume, and valid numeric values.

## Gap investigation

The official Binance spot API was queried around every gap and returned the surrounding candles but no candle for any of these seven timestamps:

- `2021-08-13T02:00:00+00:00, 2021-08-13T03:00:00+00:00, 2021-08-13T04:00:00+00:00, 2021-08-13T05:00:00+00:00`: Binance spot API omits these hours while returning 01:00 and 06:00. Scheduled Binance spot trading system upgrade beginning 02:00 UTC. Preserve the four-hour gap; do not synthesize candles.
- `2021-09-29T07:00:00+00:00, 2021-09-29T08:00:00+00:00`: Binance spot API omits these hours while returning 06:00 and 09:00. Scheduled Binance spot trading system upgrade beginning 07:00 UTC. Preserve the two-hour gap; do not synthesize candles.
- `2023-03-24T13:00:00+00:00`: Binance spot API omits 13:00 while returning a shortened zero-volume 12:00 candle and 14:00. Binance suspended spot trading after a matching-engine issue. Preserve the one-hour gap and the authoritative 12:00 candle; do not synthesize data.

The gaps are venue-history facts in the canonical data. They remain explicit.

## Differences from legacy caches

### Monday 1h cache

- Rows / coverage: `43793` / `2021-05-21T15:00:00+00:00` through `2026-05-20T14:00:00+00:00`
- Legacy file SHA-256: `9700a97dd4db2c1efe97395d5b3973cbca7c3f3e034bba33f58e840d86d947ed`
- Shared rows: `43793`
- Canonical-only / legacy-only rows: `0` / `0`
- Shared rows with different OHLCV: `1`

The one differing row is `2026-05-20 14:00 UTC`. The Monday cache captured it while forming; canonical v1 uses Binance's settled candle. The old file remains a legacy cache and was not overwritten.

### Crypto research lab 1h cache

- Rows / coverage: `43793` / `2021-05-22T15:00:00+00:00` through `2026-05-21T14:00:00+00:00`
- Legacy file SHA-256: `8321bc7a19e16c0e66349d60faa9bdef37ded649998443f35536c10fc3c15fb1`
- Shared rows: `43769`
- Canonical-only / legacy-only rows: `24` / `24`
- Shared rows with different OHLCV: `0`

The lab starts 24 hours later and ends 24 hours later. Its overlap agrees with canonical v1. The lab repository was read only throughout this work.

The separately downloaded Monday 4h file (SHA-256 `f7f09ff6b7acf69d1a2d2b7d1a667f7a256b81f819285c4a726b3d2c6ad00149`) and lab 1d file (SHA-256 `14775dac3c52ba4025583ff8b45135fcb68d762ecf24372ff7d4f4e566723592`) are legacy caches. Their final saved periods were incomplete. They are superseded for reproducible evaluation by deterministic views derived from canonical 1h data, but have not been deleted.

## Deterministic resampling

All bins use UTC `origin=start_day`, left labels, and left-closed intervals. Open is first, high is maximum, low is minimum, close is last, and volume is summed. A bin is kept only when every expected 1h timestamp is present. This drops initial/final partial bins and every 4h/daily period intersecting a genuine hourly gap.

- 4h: `10944` rows, `2021-05-21T16:00:00+00:00` through `2026-05-20T08:00:00+00:00`, deterministic frame SHA-256 `dbbb7703128974b78fdac0ea825bd9a16f2d2b834011e41c2539879dc481424f`
- 1d: `1821` rows, `2021-05-22T00:00:00+00:00` through `2026-05-19T00:00:00+00:00`, deterministic frame SHA-256 `91d4378dd3463e3e1504281e2e12199030d957710f342c4fd266cc985a5b8f83`

Derived views are generated on demand and are not separate authoritative datasets.

## Data flow and provenance

- `run_backtest.py`, `run_parameter_sweep.py`, and `run_walk_forward.py` default to canonical v1.
- Higher-timeframe evaluation is derived from canonical 1h through `src.canonical_data.load_canonical_ohlcv`.
- Diagnostics use the dataset version recorded by the originating run.
- The dashboard Data Manager and paper cycle remain live/runtime paths. Their downloaded evaluation candles exclude nominally incomplete bars; the paper cycle also applies its existing close-time guard.
- The dashboard's standard backtest and walk-forward actions inherit the canonical runner defaults. Its standalone Research Lab and Range Sweep exploratory pages still read selected live/legacy caches; their outputs must not be treated as canonical evaluation until those callers explicitly select a canonical dataset.
- Historical `data/binance/*.csv`, saved backtests, sweeps, walk-forward outputs, and paper state remain legacy/runtime artifacts.

Future run configuration records dataset version/hash, code commit, strategy configuration, accounting version, risk base, leverage, fees, slippage, intrabar policy, timeframe, requested date range, and UTC run timestamp.

## Research interpretation

Old performance artifacts are retained but classified as legacy because they lack an immutable dataset identity and may include independently downloaded higher timeframes or a forming final candle. No performance search or strategy evaluation was run while producing this dataset.
