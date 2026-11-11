import pandas as pd
import pytest

from quant_ai_research_platform.execution_costs import cost_adjusted_returns


def test_entry_and_exit_costs_apply_on_next_bar():
    dates = pd.date_range("2026-01-01", periods=4)
    prices = pd.Series([100.0, 110.0, 121.0, 121.0], index=dates)
    positions = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates)
    result = cost_adjusted_returns(prices, positions, fee_bps=10)
    assert result["turnover"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result["gross_return"].iloc[0] == 0
    assert result["gross_return"].iloc[1] == pytest.approx(0.1)
    assert result["net_return"].iloc[1] == pytest.approx(0.099)
    assert result["net_return"].iloc[3] == pytest.approx(-0.001)


def test_zero_cost_equals_gross():
    prices = pd.Series([100.0, 105.0, 110.0])
    positions = pd.Series([1.0, 0.0, 1.0])
    result = cost_adjusted_returns(prices, positions)
    pd.testing.assert_series_equal(
        result["net_return"], result["gross_return"], check_names=False
    )


def test_invalid_inputs_fail():
    with pytest.raises(ValueError):
        cost_adjusted_returns(pd.Series([1.0]), pd.Series([1.0]))
    with pytest.raises(ValueError):
        cost_adjusted_returns(pd.Series([1.0, 2.0]), pd.Series([1.0, 1.0]), fee_bps=-1)
    with pytest.raises(ValueError):
        cost_adjusted_returns(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError):
        cost_adjusted_returns(pd.Series([1.0, 0.0]), pd.Series([1.0, 1.0]))
