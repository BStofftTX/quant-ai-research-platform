"""Public-facing, read-only Quant AI Research demonstration.

No brokerage integration, remote shell access, or database secrets are needed.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

from quant_ai_research_platform.backtest import moving_average_signals
from quant_ai_research_platform.execution_costs import cost_adjusted_returns

st.set_page_config(page_title="Quant AI Research", page_icon="📊", layout="wide")
st.title("Quant AI Research Platform")
st.caption("Research demonstration · not investment advice · no order execution")


@st.cache_data
def demo_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=400)
    daily_returns = rng.normal(0.0003, 0.012, len(dates))
    close = 100 * np.cumprod(1 + daily_returns)
    return pd.DataFrame({"Close": close}, index=dates)


with st.sidebar:
    st.header("Research inputs")
    uploaded = st.file_uploader(
        "Optional CSV with Date and Close columns",
        type=["csv"],
        help="Uploaded data is analyzed within this session. Do not upload confidential financial data.",
    )
    short_window = st.number_input("Short moving average", 2, 100, 20)
    long_window = st.number_input("Long moving average", 3, 250, 50)
    fee_bps = st.number_input("Fee per trade (bps)", 0.0, 100.0, 1.0, step=0.5)
    spread_bps = st.number_input("Full bid-ask spread (bps)", 0.0, 100.0, 2.0, step=0.5)
    slippage_bps = st.number_input(
        "Slippage per trade (bps)", 0.0, 100.0, 1.0, step=0.5
    )

if short_window >= long_window:
    st.error("The short moving average must be smaller than the long moving average.")
    st.stop()

if uploaded is None:
    data = demo_prices()
    st.info("Using seeded synthetic demonstration prices, not live market data.")
else:
    try:
        raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        if not {"Date", "Close"}.issubset(raw.columns):
            raise ValueError("CSV must include Date and Close columns.")
        dates = pd.to_datetime(raw["Date"], errors="raise")
        closes = pd.to_numeric(raw["Close"], errors="raise")
        data = pd.DataFrame({"Close": closes.to_numpy()}, index=dates)
        data = data.sort_index()
        if data.index.has_duplicates or data["Close"].isna().any():
            raise ValueError("Dates must be unique and closing prices nonmissing.")
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        st.error(f"Cannot use uploaded CSV: {exc}")
        st.stop()

if len(data) < long_window + 2:
    st.error(
        "There are not enough observations for the selected moving-average windows."
    )
    st.stop()

try:
    signals = moving_average_signals(data, int(short_window), int(long_window))
    report = cost_adjusted_returns(
        data["Close"], signals, fee_bps, spread_bps, slippage_bps
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

gross_equity = (1 + report["gross_return"]).cumprod()
net_equity = (1 + report["net_return"]).cumprod()
benchmark = data["Close"] / data["Close"].iloc[0]

a, b, c = st.columns(3)
a.metric("Gross strategy return", f"{(gross_equity.iloc[-1] - 1):.2%}")
b.metric("Net strategy return", f"{(net_equity.iloc[-1] - 1):.2%}")
c.metric("Buy-and-hold return", f"{(benchmark.iloc[-1] - 1):.2%}")

st.subheader("Equity comparison")
st.line_chart(
    pd.DataFrame(
        {
            "Gross strategy": gross_equity,
            "Net strategy": net_equity,
            "Buy and hold": benchmark,
        }
    )
)
st.subheader("Prices")
st.line_chart(data["Close"])
st.download_button(
    "Download calculated returns",
    report.to_csv(index_label="Date"),
    file_name="quant_research_returns.csv",
    mime="text/csv",
)
st.caption(
    "Simplified next-bar execution model: fees, half-spread and slippage are charged "
    "on position changes. Excludes market impact, taxes, financing and market constraints. "
    "Historical and synthetic demonstrations cannot establish predictive performance."
)
