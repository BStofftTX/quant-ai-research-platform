# Streamlit Community Cloud configuration backup — 2026-11-14

This record was captured from the authenticated Streamlit Community Cloud
dashboard before changing the existing public demonstration or renaming the
repository. It is a configuration record only; no application secret value is
recorded here.

## Existing application observed

- **Application URL:** `https://quant-ai-research.streamlit.app/`
- **Repository:** `BStofftTX/quant-ai-research-platform`
- **Source branch:** `feature/macrostofft-deployment-demo` (obsolete; already
  merged into `main`)
- **Main module:** `streamlit_app/streamlit_app.py`
- **Python version:** 3.14
- **Cloud account:** `bstoffttx`

The root-level `streamlit_app.py` is a deprecated compatibility shim. The
canonical Cloud entrypoint is already the thin
`streamlit_app/streamlit_app.py` wrapper, which delegates to `app.py`.

## Required target configuration

- **Repository branch:** `main`
- **Main module:** `streamlit_app/streamlit_app.py`
- **Python version:** 3.12
- **Dependency profile:** `streamlit_app/requirements.txt`
- **Secrets:** none

The application uses deterministic synthetic data. It must not receive market
data, brokerage, customer-portfolio, or production database credentials.

## Safeguards and rollback

1. Preserve the existing custom URL during the branch and Python transition.
2. Do not change the sharing/privacy setting without separate authorization.
3. Do not add or display secrets. The Secrets tab was not read or modified
   during this backup.
4. If the rebuilt application fails, restore the previous source branch
   `feature/macrostofft-deployment-demo` and Python 3.14, then investigate the
   build logs before retrying.
5. Record the deployed commit SHA, Cloud build result, and smoke-test results
   before repository rename, tagging, or a public URL change.

## Follow-on release sequence

After this backup is merged: update the existing app to the target
configuration, verify a successful build and the retained current URL, tag
`v0.1.0-demo`, create the approved GitHub Release, then rename the repository.
The optional `marketscope-analytics.streamlit.app` URL remains approval-gated
and must not replace the current URL without Bruce Stofft's explicit approval.
