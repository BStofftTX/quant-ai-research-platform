# Streamlit deployment and release verification

This branch provides a **new reference dashboard**, not an assertion about the
source of the existing quant-ai-research.streamlit.app deployment.

## Reproducible local launch

From the repository root, run:

```bash
uv sync --locked --dev
uv run --with streamlit streamlit run streamlit_app.py
```

The dashboard uses deterministic synthetic data by default, accepts optional
Date/Close CSV uploads, and does not need PostgreSQL or secrets.

## Community Cloud

In the Streamlit workspace, inspect the existing application's **Settings**
and confirm its actual repository, branch and entrypoint. Do not change the
privacy setting without authorization. To deploy this dashboard in a new app,
choose repository `BStofftTX/quant-ai-research-platform`, the merged branch,
and `streamlit_app.py` as the main file. `requirements.txt` installs the
package and Streamlit.

A GitHub pull request or merge does **not** deploy this code to an existing
Streamlit application unless that app is already configured to use this
repository, branch and entrypoint. Record the deployed commit SHA and check
Cloud logs after release.

## Smoke checks

1. Load without credentials if the site is intentionally public.
2. Confirm a synthetic-data informational banner appears.
3. Change moving average windows and transaction costs; charts and net-return
   figures must recalculate.
4. Confirm net returns do not exceed gross returns with nonnegative costs.
5. Upload a CSV with Date/Close columns; confirm uploaded data is reflected.
6. Upload an invalid CSV; confirm a readable validation message.
7. Download returns and check the Date, gross_return, turnover,
   execution_cost, net_return columns.
8. Check on narrow/mobile layout and inspect Streamlit logs for tracebacks.

No brokerage integration, account-specific financial advice, authentication
bypass, or secrets are needed for this research demonstration.
