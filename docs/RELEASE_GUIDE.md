# Quant-AI Streamlit release guide

## Scope and safety boundary

This guide prepares a **research demonstration** only. It uses deterministic
synthetic OHLC data (seed `42`), does not access a database or brokerage, and
does not require secrets. It is not investment advice or a production trading
system.

## Private review walkthrough

1. Start the app with `uv run streamlit run app.py` for the full local profile,
   or use the Cloud-profile command below.
2. **Security analysis:** select a synthetic asset and review the seeded price
   chart, return, volatility, Sharpe ratio, beta, and tail-risk table.
3. **Portfolio risk:** compare equal-weight and constrained optimized synthetic
   portfolios; verify the weights, metrics, and cumulative-return chart.
4. **Walk-forward models:** review chronological logistic-regression splits.
   The hosted profile intentionally omits PyTorch and identifies that the
   logistic-regression path remains available.

Private review screenshots are stored in `docs/screenshots/`. They must be
reviewed before use on MacroStofft.com or any other public surface.

## Exact Community Cloud configuration

- Repository: `BStofftTX/quant-ai-research-platform`
- Branch: `feature/macrostofft-deployment-demo`
- Entrypoint: `streamlit_app/streamlit_app.py`
- Python: `3.12`
- Dependencies: `streamlit_app/requirements.txt`
- Secrets: leave empty
- External system packages: none

Community Cloud searches the entrypoint directory before the repository root
for dependency files. This layout keeps the root `uv.lock` and optional
PyTorch, PostgreSQL, and yfinance dependencies out of the hosted demo.

## Local Cloud-profile verification

```bash
uv venv --python 3.12 .venv-cloud
.venv-cloud/bin/pip install -r streamlit_app/requirements.txt
PYTHONPATH=.:src .venv-cloud/bin/python -c "import app; print('offline-demo-import-ok')"
PYTHONPATH=.:src .venv-cloud/bin/streamlit run streamlit_app/streamlit_app.py
```

Run the full project checks separately:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pip-audit
uv run pip-audit -r streamlit_app/requirements.txt
```

## Rollback and troubleshooting

- **Bad app revision:** create a normal revert commit on the deployment branch
  to restore the previously reviewed state, then let Community Cloud redeploy.
  Do not force-push.
- **Dependency/build failure:** compare the entrypoint, Python version, and
  `streamlit_app/requirements.txt`; do not add the root `uv.lock` or optional
  research dependencies as a workaround.
- **PyTorch unavailable:** expected for this profile. Confirm the logistic
  regression table renders and retain the disclosure.
- **Resource warning or slow startup:** do not claim production readiness;
  record observed limits and obtain approval before moving to paid or private
  hosting.
- **Python-version change:** Community Cloud requires app deletion and
  redeployment to change Python, so record the existing app settings first.

## Known limitations

- Community Cloud resource limits are shared and can change; it is not a
  capacity guarantee or SLA.
- The full local development lock reported 42 transitive dependency findings
  during the October 26, 2026 release check. The isolated Cloud requirements
  audit was clean and does not use that lock; refresh the full local profile
  in a separate, reviewed dependency-maintenance change.
- A hosted app receives a shareable `streamlit.app` URL. Publishing one,
  changing repository visibility, or connecting private data needs approval.
- The demo does not include fees, slippage, tax, liquidity, market impact, or
  execution constraints needed for production financial use.
- PyTorch comparisons remain a full-local-profile capability pending a separate
  resource review.
