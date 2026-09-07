# Legacy material

This directory preserves superseded project material for provenance. It is excluded from the maintained application, package, and test suite.

- `pre_correction_results/` contains performance output produced before marked-equity accounting, current-equity sizing, leverage limits, gap-aware fills, explicit intrabar policy, and terminal closure. It is not valid Protocol V1 evidence.
- `scripts/` contains the Yahoo-era entry points, a duplicate Binance downloader, and the generic pre-Protocol parameter-sweep and walk-forward runners.
- `src/` contains the former 12-page dashboard backend, Yahoo loader, and superseded range-sweep research surface.
- `tests/` preserves tests tied only to those retired modules.
- `planning/` contains the early bot roadmap, which no longer describes the research application.

The maintained research path is documented in the repository root README. Canonical results live only in the sealed `reports/experiments/monday_range_protocol_v1/canonical_20260905_3528492/` bundle.
