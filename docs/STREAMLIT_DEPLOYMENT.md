# Streamlit deployment and release verification

The supported deployment is the thin Community Cloud entrypoint at
`streamlit_app/streamlit_app.py` on `main`. The root `streamlit_app.py` is a
deprecated compatibility shim that delegates to the same implementation.

## Reproducible local launch

From the repository root, run:

```bash
uv sync --locked --dev
uv run --with streamlit streamlit run streamlit_app/streamlit_app.py
```

The dashboard uses deterministic synthetic data and does not need PostgreSQL,
secrets, live prices, brokerage accounts, or customer portfolios.

## Community Cloud

In the Streamlit workspace, inspect the existing application's **Settings**
and confirm its actual repository, branch and entrypoint. Do not change the
privacy setting without authorization. To deploy this dashboard in a new app,
choose repository `BStofftTX/quant-ai-research-platform`, branch `main`, and
`streamlit_app/streamlit_app.py` as the main file. Use
`streamlit_app/requirements.txt`; leave secrets empty.

A GitHub pull request or merge does **not** deploy this code to an existing
Streamlit application unless that app is already configured to use this
repository, branch and entrypoint. Record the deployed commit SHA and check
Cloud logs after release.

## Smoke checks

1. Load without credentials if the site is intentionally public.
2. Confirm a synthetic-data informational banner appears.
3. Confirm the Security analysis, Portfolio risk, and Walk-forward models tabs
   render from deterministic synthetic data.
4. Check on narrow/mobile layout and inspect Streamlit logs for tracebacks.

No brokerage integration, account-specific financial advice, authentication
bypass, or secrets are needed for this research demonstration.
