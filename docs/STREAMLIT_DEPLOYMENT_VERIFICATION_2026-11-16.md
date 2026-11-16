# MarketScope Analytics — deployment verification (2026-11-16)

## Deployed application

- **Live URL:** `https://marketscopebymacrostofft.streamlit.app/`
- **Repository:** `BStofftTX/quant-ai-research-platform`
- **Verified source SHA:** `027859bdbea098cf3175613ee2de176d45430cf5`
- **Cloud source branch:** `feature/macrostofft-deployment-demo`
  - This compatibility branch is an exact pointer to the verified `main` SHA above.
- **Entrypoint:** `streamlit_app/streamlit_app.py`
- **Runtime:** Python 3.12.16
- **Dependency profile:** `streamlit_app/requirements.txt`
- **Secrets:** none required or configured for the demonstration.

## Build evidence

Streamlit Community Cloud rebuilt the app successfully on 2026-11-16. Its log
records successful installation from `streamlit_app/requirements.txt`, then
starts Uvicorn on port 8501. The Cloud environment initially resolved pyarrow
26.0.1, detected Streamlit's known issue with that build, and automatically
replaced it with pyarrow 24.0.0 before startup. No runtime exception was shown.

The slim profile deliberately excludes PostgreSQL, live-market providers,
development tooling, and PyTorch. It contains Streamlit, NumPy, pandas,
scikit-learn, SciPy, and Pillow only.

## Browser verification

Verified without a Streamlit sign-in in a clean browser profile:

- Desktop access reached the application and rendered all three tabs:
  **Security analysis**, **Portfolio risk**, and **Walk-forward models**.
- The security selector, threshold slider, tables, charts, CSV/PNG controls,
  and tab controls rendered without browser exceptions.
- The same app rendered at an iPhone-sized viewport with the expected collapsed
  sidebar control and no observed horizontal-overflow failure.

The Cloud dashboard's Sharing setting is **This app is public and searchable**.
Command-line `HEAD` requests may redirect to Streamlit's authentication endpoint;
browser verification is the authoritative public-access proof for this release.

## Demonstration limitations

MarketScope is an offline research demonstration using deterministic synthetic
OHLC data (NumPy seed 42). It does not use live prices, brokerage accounts,
customer portfolios, production databases, credentials, or trading execution.
Outputs do not establish predictive performance or provide investment advice.

The app does not model all real-market constraints, including fees, spreads,
slippage, taxes, liquidity, borrowing, market impact, corporate actions,
survivorship effects, or production operations.

## Local and GitHub gates

- Python 3.12: 67 tests passed; 4 optional tests skipped.
- Ruff check and formatting checks passed.
- Full and slim dependency audits reported no known vulnerabilities.
- GitHub CI and CodeQL passed for `027859b`.

## Rollback

If the current deployment fails, restore Python 3.14 in the existing app's
General settings. The compatibility branch remains an exact pointer to the
verified `main` revision, so this rollback does not alter the public URL or
source content. Any future source-branch transition must be tested as a
controlled redeploy while retaining the existing custom URL.
