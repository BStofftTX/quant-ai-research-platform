from datetime import date

import pandas as pd

from quant_ai_research_platform.market_analysis import (
    calculate_benchmark_metrics,
    calculate_factor_regression,
    find_abnormal_returns,
    calculate_statistics,
)


def test_calculate_statistics():
    data = pd.DataFrame(
        {"Close": [100.0, 110.0, 121.0]}
    )

    stats = calculate_statistics(data)

    assert stats["starting_price"] == 100.0
    assert stats["ending_price"] == 121.0
    assert round(stats["total_return"], 2) == 0.21


def test_benchmark_metrics_align_dates():
    stock = pd.DataFrame(
        {"Close": [100.0, 108.0, 121.0, 127.0]},
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
        ),
    )

    benchmark = pd.DataFrame(
        {"Close": [200.0, 210.0, 218.0]},
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06"]
        ),
    )

    result = calculate_benchmark_metrics(stock, benchmark)

    assert result["observations"] == 3
    assert round(result["r_squared"], 6) == round(result["correlation"] ** 2, 6)
    assert result["start_date"] == date(2026, 1, 2)
    assert result["end_date"] == date(2026, 1, 6)


def test_calculate_factor_regression():
    factor = pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0, 104.0, 103.0]},
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        ),
    )
    stock = factor.copy()
    stock["Close"] = 2 * factor["Close"]

    result = calculate_factor_regression(stock, factor)

    assert result["observations"] == 4
    assert round(result["beta"], 6) == 1.0
    assert round(result["daily_alpha"], 6) == 0.0
    assert round(result["r_squared"], 6) == 1.0
    assert abs(result["residuals"]).max() < 1e-12


def test_find_abnormal_returns():
    residuals = pd.Series(
        [0.01, -0.03, 0.02, -0.005],
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
        ),
    )

    result = find_abnormal_returns(residuals, top_n=2)

    assert list(result.index) == list(
        pd.to_datetime(["2026-01-05", "2026-01-06"])
    )
    assert result.iloc[0]["residual"] == -0.03
    assert result.iloc[1]["residual"] == 0.02
    assert result.iloc[0]["absolute_residual"] == 0.03
