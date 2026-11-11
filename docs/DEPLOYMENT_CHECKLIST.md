# Demonstration Deployment Checklist

## Recommended first host

Use **Streamlit Community Cloud** for the first portfolio demonstration. It is
GitHub-based and the app must remain self-contained and contain no secrets.
Resource limits are shared and may change, so this is appropriate for a small
research demonstration rather than a production client service.

## Community Cloud entrypoint

- Repository branch: `feature/macrostofft-deployment-demo`
- Entrypoint: `streamlit_app/streamlit_app.py`
- Python version: 3.12
- Dependency file: `streamlit_app/requirements.txt`

The entrypoint directory is intentional. Community Cloud resolves dependency
files from the entrypoint directory before the repository root; this prevents
the root `uv.lock` development environment from installing optional PyTorch and
research-only dependencies for the public demonstration.

## Before deployment

- [ ] Human review of the feature-branch diff and sample output.
- [ ] Confirm the repository and app are safe to make public.
- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run ruff check .` and `uv run ruff format --check .`.
- [ ] Run `uv run pip-audit` for the full development environment and
  `uv run pip-audit -r streamlit_app/requirements.txt` for the Cloud profile.
- [ ] Test `uv run streamlit run app.py` locally with `uv sync --extra demo --extra pytorch --dev`.
- [ ] Test `streamlit run streamlit_app/streamlit_app.py` locally using the slim Cloud dependency profile.
- [ ] Confirm the demo uses only seeded synthetic data and no PostgreSQL connection.

## Configuration and dependencies

- Python 3.12+
- `streamlit` demo extra
- `torch` optional extra for the PyTorch comparison
- Standard project dependencies for pandas, NumPy, scikit-learn, and SciPy
- No database URL, API key, brokerage credential, or private data required

The Cloud profile pins only Streamlit, NumPy, pandas, SciPy, and scikit-learn.
It intentionally omits `torch`, `psycopg`, and `yfinance`. The PyTorch panel
degrades to the logistic-regression result when PyTorch is absent.

## Secrets and security

- [ ] Do not add `.streamlit/secrets.toml` to Git.
- [ ] Do not place credentials in source, screenshots, sample data, or logs.
- [ ] If a future private data source is added, use the host's secret manager and
  least-privilege read-only credentials.
- [ ] Keep the public demo separate from any production or client data path.

## Cost and hosting decision

- **First choice:** Streamlit Community Cloud, subject to Bruce's deployment
  approval and the provider's current terms and shared resource limits.
- **Fallback:** a small container service or a local/private host if the app
  later needs private data, persistent storage, scheduled jobs, or stricter
  access control. Re-evaluate cost and security at that point.

## Approval gates

Public deployment, repository visibility changes, external communications,
private-data access, paid hosting, and production credentials require Bruce's
explicit approval. This checklist does not authorize any of those actions.
