# Deploy the research dashboard

The Streamlit app is designed to run from a clean checkout. It reads the tracked canonical dataset and sealed Protocol V1 result bundle. It does not need credentials, a database, a live Binance connection, or a writable research directory.

## Local production-style start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m unittest discover -s tests -q
streamlit run dashboard.py
```

The verification command checks the frozen protocol, canonical dataset SHA-256, one-time holdout marker, final candidate, and every sealed JSON sidecar. It does not run a backtest or reopen the holdout.

## Streamlit Community Cloud

1. Push the validated release branch to the public GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the entry point to `dashboard.py` and use the repository root as the working directory.
4. Deploy without secrets. The repository's `requirements.txt`, `runtime.txt`, `.streamlit/config.toml`, canonical data, and sealed results provide everything the app needs.
5. Confirm the Overview page loads and compare its run ID and hashes with the final report.

The app has no canonical artifact writer. Explore evaluates a single configuration in memory, is limited to dates before the consumed holdout, and cannot write into `reports/experiments/`.

## Release check

GitHub Actions repeats artifact verification, the full unit suite, and a headless Streamlit health check on Python 3.11. Do not deploy a revision whose checks fail.
