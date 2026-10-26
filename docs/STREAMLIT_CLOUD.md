# Streamlit Community Cloud readiness

## Deployment target

Deploy the `feature/macrostofft-deployment-demo` branch with:

- **Entrypoint:** `streamlit_app/streamlit_app.py`
- **Python:** 3.12
- **Dependencies:** `streamlit_app/requirements.txt`
- **Secrets:** none

This layout is compatible with Streamlit Community Cloud's dependency lookup:
the entrypoint directory is considered before repository-root dependency files.
The local `uv.lock` remains available for full development and testing, while
the hosted demo receives only its required runtime packages.

The requirements file also pins Pillow at a security-fixed version. Re-run the
requirements audit immediately before approving a deployment because Community
Cloud builds a fresh environment from this file.

## Resource profile

The default demonstration generates four 520-business-day synthetic series and
runs CPU-only analytics. It needs no database, background jobs, persistent
storage, API calls, or external Linux packages.

Expected hosted profile:

- CPU: short interactive bursts only; no scheduled computation.
- Memory: modest analytics workload; select the standard Community Cloud
  allocation and observe initial launch before committing to another host.
- Storage: source code and normal Python wheels only; no dataset download or
  model checkpoint storage.

These are deployment-planning estimates, not a capacity guarantee. Measure the
first staged deployment before making any performance claim.

## Optional PyTorch comparison

The repository's full local development environment supports the optional
PyTorch comparison. It is intentionally omitted from the default Community
Cloud profile because the local PyTorch installation alone is approximately
553 MB. When it is absent, the UI keeps the logistic-regression comparison
available and explains the omission.

Enable PyTorch only after a staged resource review demonstrates that the host
supports it comfortably. Do not add model packages, credentials, or private
data to the public demonstration without explicit approval.

## Approval gate before deployment

1. Review the deployed-branch diff and the rendered synthetic-data disclosures.
2. Confirm repository visibility and that only approved public material exists.
3. In Streamlit Community Cloud, select the feature branch and the entrypoint
   above; retain Python 3.12.
4. Leave the secrets field empty.
5. Verify the deployed page, disclosures, and optional-PyTorch fallback.
6. Obtain approval before sharing the URL or merging any code.
