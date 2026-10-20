# Demonstration Deployment Checklist

## Recommended first host

Use **Streamlit Community Cloud** for the first portfolio demonstration. The
official Streamlit documentation describes it as a free deployment option for
non-commercial, personal, and educational apps, with GitHub-based deployment.
The app must remain self-contained and contain no secrets.

## Before deployment

- [ ] Human review of the feature-branch diff and sample output.
- [ ] Confirm the repository and app are safe to make public.
- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run ruff check .` and `uv run ruff format --check .`.
- [ ] Run `uv run pip-audit` after the lockfile is refreshed.
- [ ] Test `uv run streamlit run app.py` locally with `uv sync --extra demo --extra pytorch --dev`.
- [ ] Confirm the demo uses only seeded synthetic data and no PostgreSQL connection.

## Configuration and dependencies

- Python 3.12+
- `streamlit` demo extra
- `torch` optional extra for the PyTorch comparison
- Standard project dependencies for pandas, NumPy, scikit-learn, and SciPy
- No database URL, API key, brokerage credential, or private data required

## Secrets and security

- [ ] Do not add `.streamlit/secrets.toml` to Git.
- [ ] Do not place credentials in source, screenshots, sample data, or logs.
- [ ] If a future private data source is added, use the host's secret manager and
  least-privilege read-only credentials.
- [ ] Keep the public demo separate from any production or client data path.

## Cost and hosting decision

- **First choice:** Streamlit Community Cloud; no infrastructure charge is
  expected for this demonstration, subject to the provider's current terms and
  resource limits.
- **Fallback:** a small container service or a local/private host if the app
  later needs private data, persistent storage, scheduled jobs, or stricter
  access control. Re-evaluate cost and security at that point.

## Approval gates

Public deployment, repository visibility changes, external communications,
private-data access, paid hosting, and production credentials require Bruce's
explicit approval. This checklist does not authorize any of those actions.
