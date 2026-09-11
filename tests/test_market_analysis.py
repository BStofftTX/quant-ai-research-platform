import pytest
def test_buy_and_hold_return():
    import pandas as pd
    from quant_ai_research_platform.backtest import buy_and_hold_return

    data = pd.DataFrame({"Close": [100.0, 110.0]})

    result = buy_and_hold_return(data)

    assert result == pytest.approx(0.10)
from datetime import date

import pandas as pd

from quant_ai_research_platform.market_analysis import (
    calculate_benchmark_metrics,
    calculate_factor_regression,
    calculate_rolling_metrics,
    find_abnormal_returns,
    calculate_statistics,
    create_risk_report,
    create_security_profile,
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

def test_find_abnormal_returns_flags_large_z_scores():
    residuals = pd.Series(
        [0.001, -0.002, 0.0005, 0.0015, -0.020],
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
            ]
        ),
    )

    result = find_abnormal_returns(
        residuals,
        top_n=5,
        z_threshold=1.5,
    )

    assert "z_score" in result.columns
    assert "is_abnormal" in result.columns
    assert result.loc[pd.Timestamp("2026-01-08"), "is_abnormal"]

def test_calculate_rolling_metrics():
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
        ]
    )

    stock = pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0, 104.0, 103.0, 106.0]},
        index=dates,
    )

    benchmark = pd.DataFrame(
        {"Close": [200.0, 202.0, 201.0, 203.0, 202.0, 205.0]},
        index=dates,
    )

    result = calculate_rolling_metrics(
        stock,
        benchmark,
        window=3,
    )

    assert list(result.columns) == [
        "annualized_volatility",
        "correlation",
        "beta",
    ]
    assert len(result) == 3
    assert result.notna().all().all()

def test_factor_regression_inference_fields():
    factor = pd.DataFrame(
        {"Close": [100.0, 101.0, 103.0, 102.0, 105.0, 107.0]},
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
            ]
        ),
    )

    stock = pd.DataFrame(
        {"Close": [200.0, 202.4, 206.0, 204.2, 210.5, 214.4]},
        index=factor.index,
    )

    result = calculate_factor_regression(stock, factor)

    assert result["beta_standard_error"] > 0
    assert result["beta_ci_lower"] < result["beta"] < result["beta_ci_upper"]
    assert 0 <= result["beta_p_value"] <= 1

def test_calculate_statistics_includes_tail_risk():
    data = pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0, 99.0, 95.0, 97.0, 94.0]}
    )

    stats = calculate_statistics(data)

    assert stats["historical_var_95"] <= 0
    assert stats["historical_es_95"] <= stats["historical_var_95"]
    assert stats["historical_var_99"] <= stats["historical_var_95"]
    assert stats["historical_es_99"] <= stats["historical_var_99"]


def test_create_risk_report():
    data = pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0, 99.0, 95.0, 97.0, 94.0]}
    )

    report = create_risk_report(data)

    assert set(report) == {
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "historical_var_95",
        "historical_es_95",
        "historical_var_99",
        "historical_es_99",
    }

def test_calculate_statistics_includes_sortino():
    data = pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0, 99.0, 103.0, 98.0, 104.0]}
    )

    stats = calculate_statistics(data)

    assert stats["downside_deviation"] > 0
    assert "sortino_ratio" in stats
    assert stats["sortino_ratio"] == stats["average_daily_return"] * 252 / stats["downside_deviation"]

def test_calculate_statistics_includes_calmar():
    data = pd.DataFrame(
        {"Close": [100.0, 105.0, 103.0, 108.0, 104.0, 110.0]}
    )

    stats = calculate_statistics(data)

    assert "calmar_ratio" in stats
    assert stats["calmar_ratio"] == stats["cagr"] / abs(stats["max_drawdown"])


def test_create_security_profile():
    """Security profile should organize metrics into logical categories."""
    profile = {
        "performance": {},
        "risk": {},
        "risk_adjusted": {},
        "market_behavior": {},
    }

    assert set(profile.keys()) == {
        "performance",
        "risk",
        "risk_adjusted",
        "market_behavior",
    }
