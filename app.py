"""Minimal, offline Streamlit demonstration for the quant research library.

The demo intentionally creates all market data locally.  It does not connect to
PostgreSQL, download prices, or require credentials.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from quant_ai_research_platform.market_analysis import (
    analyze_portfolio,
    calculate_benchmark_metrics,
    calculate_statistics,
    create_security_profile,
    optimize_portfolio,
)
from quant_ai_research_platform.modeling import run_walk_forward_backtest

ASSETS = ("ALPHA", "BETA", "GAMMA")
BENCHMARK = "BENCHMARK"
SEED = 42


def make_synthetic_ohlc(
    periods: int = 520,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Return deterministic synthetic OHLC frames for the demo."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=periods)
    benchmark_returns = rng.normal(0.00025, 0.009, periods)
    frames: dict[str, pd.DataFrame] = {}

    for position, asset in enumerate((BENCHMARK, *ASSETS)):
        idiosyncratic = rng.normal(0.0001 * (position + 1), 0.010, periods)
        returns = 0.35 * benchmark_returns + idiosyncratic
        close = 100.0 * np.exp(np.cumsum(returns))
        open_price = close * (1.0 + rng.normal(0.0, 0.002, periods))
        high = np.maximum(open_price, close) * (
            1.0 + rng.uniform(0.001, 0.012, periods)
        )
        low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.012, periods))
        frames[asset] = pd.DataFrame(
            {"Open": open_price, "High": high, "Low": low, "Close": close},
            index=dates,
        )
    return frames


def _optional_torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _metric_rows(metrics: dict[str, Any], names: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [name.replace("_", " ").title() for name in names],
            "Value": [metrics[name] for name in names],
        }
    )


def run_demo() -> None:
    """Render the Streamlit application, or explain how to install it."""
    try:
        import streamlit as st
    except ImportError:
        print(
            "Streamlit is not installed. Install it with `pip install streamlit` and run `streamlit run app.py`."
        )
        return

    st.set_page_config(
        page_title="Quant AI Research Demo",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.title("Quant AI Research Platform")
    st.subheader("Offline research demonstration")
    st.info(
        "This application uses deterministic synthetic OHLC data (seed 42). "
        "It does not access live prices, a database, credentials, or a brokerage account."
    )
    data = make_synthetic_ohlc()
    st.sidebar.header("Demo controls")
    selected = st.sidebar.selectbox("Synthetic security", ASSETS)
    threshold = st.sidebar.slider("Signal threshold", 0.50, 0.80, 0.60, 0.05)
    st.sidebar.divider()
    st.sidebar.caption(
        "Research/education only. Results are a reproducible pipeline demonstration, not live-market performance or investment advice."
    )

    with st.expander("Data and model disclosures", expanded=False):
        st.markdown(
            "- **Data:** locally generated synthetic OHLC series; no market, customer, or personal data.\n"
            "- **Reproducibility:** fixed NumPy seed `42`; changing a control does not retrieve external data.\n"
            "- **Models:** chronological walk-forward evaluation; the PyTorch comparison is optional.\n"
            "- **Limits:** this is not investment advice, a trading system, or evidence of predictive performance."
        )

    tabs = st.tabs(["Security analysis", "Portfolio risk", "Walk-forward models"])

    with tabs[0]:
        security = data[selected]
        benchmark = data[BENCHMARK]
        stats = calculate_statistics(security)
        comparison = calculate_benchmark_metrics(security, benchmark)
        profile = create_security_profile(stats, comparison)
        st.subheader(f"{selected} versus synthetic {BENCHMARK}")
        st.caption("Price paths are illustrative outputs from the seeded generator.")
        st.line_chart(
            pd.DataFrame({selected: security["Close"], BENCHMARK: benchmark["Close"]})
        )
        cards = st.columns(4)
        cards[0].metric("Total return", f"{profile['performance']['total_return']:.2%}")
        cards[1].metric(
            "Annualized volatility", f"{profile['risk']['annualized_volatility']:.2%}"
        )
        cards[2].metric(
            "Sharpe ratio", f"{profile['risk_adjusted']['sharpe_ratio']:.2f}"
        )
        cards[3].metric("Beta", f"{profile['market_behavior']['beta']:.2f}")
        st.dataframe(
            _metric_rows(
                stats,
                (
                    "cagr",
                    "max_drawdown",
                    "sortino_ratio",
                    "historical_var_95",
                    "historical_es_95",
                ),
            ),
            hide_index=True,
            use_container_width=True,
        )

    with tabs[1]:
        returns = pd.DataFrame(
            {
                asset: frame["Close"].pct_change()
                for asset, frame in data.items()
                if asset != BENCHMARK
            }
        ).dropna()
        equal = analyze_portfolio(returns)
        optimized = optimize_portfolio(returns)
        st.subheader("Synthetic portfolio risk")
        st.caption(
            "Weights and risk statistics are calculated from the generated series only; no portfolio is traded or managed."
        )
        left, right = st.columns(2)
        for container, title, result in (
            (left, "Equal-weight portfolio", equal),
            (right, "Long-only optimized portfolio", optimized),
        ):
            container.markdown(f"**{title}**")
            container.dataframe(
                pd.DataFrame({"Weight": result["weights"]}).style.format("{:.1%}"),
                use_container_width=True,
            )
            container.dataframe(
                _metric_rows(
                    result["statistics"],
                    (
                        "total_return",
                        "annualized_return",
                        "annualized_volatility",
                        "sharpe_ratio",
                        "max_drawdown",
                    ),
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.line_chart(
            pd.DataFrame(
                {
                    "Equal weight": (1 + equal["returns"]).cumprod(),
                    "Optimized": (1 + optimized["returns"]).cumprod(),
                }
            )
        )

    with tabs[2]:
        st.subheader("Walk-forward model comparison")
        st.write(
            "Each split trains only on prior observations. Both paths use the same synthetic security and signal threshold."
        )
        initial_train_size, test_size = 260, 40
        logistic = run_walk_forward_backtest(
            data[selected], initial_train_size, test_size, threshold, "logistic"
        )
        st.dataframe(logistic, hide_index=True, use_container_width=True)
        if _optional_torch_available():
            try:
                pytorch = run_walk_forward_backtest(
                    data[selected], initial_train_size, test_size, threshold, "pytorch"
                )
                st.dataframe(
                    pytorch.assign(model="PyTorch (optional)"),
                    hide_index=True,
                    use_container_width=True,
                )
                st.line_chart(
                    pd.DataFrame(
                        {
                            "Logistic excess return": logistic[
                                "excess_return"
                            ].reset_index(drop=True),
                            "PyTorch excess return": pytorch[
                                "excess_return"
                            ].reset_index(drop=True),
                        }
                    )
                )
            except (ImportError, RuntimeError, ValueError) as error:
                st.warning(
                    f"PyTorch comparison unavailable: {error}. Logistic regression remains available."
                )
        else:
            st.info(
                "The slim Cloud profile omits optional PyTorch to keep startup resources modest. "
                "Logistic regression remains fully available."
            )
        st.caption(
            "Walk-forward results are illustrative, use synthetic data, and are not investment advice."
        )


if __name__ == "__main__":
    run_demo()
