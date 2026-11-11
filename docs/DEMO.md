# Offline Streamlit demonstration

This demonstration is intentionally self-contained. It generates reproducible synthetic OHLC data locally with NumPy (seed `42`) and does not use PostgreSQL, network access, market-data credentials, or external deployment.

## Run

From the repository root:

```bash
streamlit run app.py
```

Install Streamlit if it is not already present:

```bash
python -m pip install streamlit
```

The application still imports safely when Streamlit is missing and prints a short installation message when executed directly with Python.

## Demonstrated paths

- **Security analysis:** synthetic security prices versus a synthetic benchmark, including return, volatility, drawdown, Sharpe, VaR/ES, beta, and alpha.
- **Portfolio risk:** equal-weight and long-only optimized portfolios using the existing portfolio analytics APIs.
- **Walk-forward models:** chronological logistic-regression evaluation and, when installed, a comparable optional PyTorch classifier.

PyTorch is optional. If it is unavailable or fails to initialize, the UI explains that limitation and keeps the logistic-regression results available. Results are for research/education only and are not investment advice.
