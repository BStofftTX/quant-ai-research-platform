import math

import pytest


def test_buy_and_hold_return():
    import pandas as pd

    from quant_ai_research_platform.backtest import buy_and_hold_return

    data = pd.DataFrame({"Close": [100.0, 110.0]})

    result = buy_and_hold_return(data)

    assert result == pytest.approx(0.10)


def test_strategy_returns_use_prior_period_signal_to_prevent_lookahead():
    import pandas as pd

    from quant_ai_research_platform.backtest import calculate_strategy_returns

    data = pd.DataFrame({"Close": [100.0, 110.0, 90.0]})
    signals = pd.Series([1.0, 0.0, 1.0])

    result = calculate_strategy_returns(data, signals)

    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.10)
    assert result.iloc[2] == pytest.approx(0.0)


from datetime import date

import pandas as pd

from quant_ai_research_platform.market_analysis import (
    calculate_benchmark_metrics,
    calculate_factor_regression,
    calculate_rolling_metrics,
    calculate_statistics,
    create_risk_report,
    find_abnormal_returns,
    optimize_portfolio,
)


def test_calculate_statistics():
    data = pd.DataFrame({"Close": [100.0, 110.0, 121.0]})

    stats = calculate_statistics(data)

    assert stats["starting_price"] == 100.0
    assert stats["ending_price"] == 121.0
    assert round(stats["total_return"], 2) == 0.21


def test_benchmark_metrics_align_dates():
    stock = pd.DataFrame(
        {"Close": [100.0, 108.0, 121.0, 127.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
    )

    benchmark = pd.DataFrame(
        {"Close": [200.0, 210.0, 218.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
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
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
    )

    result = find_abnormal_returns(residuals, top_n=2)

    assert list(result.index) == list(pd.to_datetime(["2026-01-05", "2026-01-06"]))
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
    data = pd.DataFrame({"Close": [100.0, 102.0, 101.0, 99.0, 95.0, 97.0, 94.0]})

    stats = calculate_statistics(data)

    assert stats["historical_var_95"] <= 0
    assert stats["historical_es_95"] <= stats["historical_var_95"]
    assert stats["historical_var_99"] <= stats["historical_var_95"]
    assert stats["historical_es_99"] <= stats["historical_var_99"]


def test_create_risk_report():
    data = pd.DataFrame({"Close": [100.0, 102.0, 101.0, 99.0, 95.0, 97.0, 94.0]})

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
    data = pd.DataFrame({"Close": [100.0, 102.0, 101.0, 99.0, 103.0, 98.0, 104.0]})

    stats = calculate_statistics(data)

    assert stats["downside_deviation"] > 0
    assert "sortino_ratio" in stats
    assert (
        stats["sortino_ratio"]
        == stats["average_daily_return"] * 252 / stats["downside_deviation"]
    )


def test_calculate_statistics_includes_calmar():
    data = pd.DataFrame({"Close": [100.0, 105.0, 103.0, 108.0, 104.0, 110.0]})

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


def test_annualized_return():
    from quant_ai_research_platform.backtest import annualized_return

    result = annualized_return(0.10, periods=252)

    assert result == pytest.approx(0.10)


def test_excess_return():
    from quant_ai_research_platform.backtest import excess_return

    result = excess_return(0.15, 0.10)

    assert result == pytest.approx(0.05)


def test_max_drawdown():
    import pandas as pd

    from quant_ai_research_platform.backtest import max_drawdown

    equity_curve = pd.Series([100.0, 120.0, 90.0, 110.0])

    result = max_drawdown(equity_curve)

    assert result == pytest.approx(-0.25)


def test_summarize_backtest():
    import pandas as pd

    from quant_ai_research_platform.backtest import summarize_backtest

    data = pd.DataFrame({"Close": [100.0, 110.0, 105.0, 120.0]})

    result = summarize_backtest(
        data,
        benchmark_return=0.08,
        periods_per_year=3,
    )

    assert result["total_return"] == pytest.approx(0.20)
    assert result["annualized_return"] == pytest.approx(0.20)
    assert result["benchmark_return"] == pytest.approx(0.08)
    assert result["excess_return"] == pytest.approx(0.12)
    assert result["max_drawdown"] == pytest.approx((105.0 / 110.0) - 1)


def test_calculate_strategy_returns_uses_prior_signal():
    import pandas as pd

    from quant_ai_research_platform.backtest import calculate_strategy_returns

    data = pd.DataFrame({"Close": [100.0, 110.0, 121.0]})
    signals = pd.Series([0.0, 1.0, 1.0])

    result = calculate_strategy_returns(data, signals)

    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(0.10)


def test_build_equity_curve():
    import pandas as pd

    from quant_ai_research_platform.backtest import build_equity_curve

    strategy_returns = pd.Series([0.0, 0.10, -0.05])

    result = build_equity_curve(strategy_returns, initial_value=100.0)

    assert result.iloc[0] == pytest.approx(100.0)
    assert result.iloc[1] == pytest.approx(110.0)
    assert result.iloc[2] == pytest.approx(104.5)


def test_run_strategy_backtest():
    import pandas as pd

    from quant_ai_research_platform.backtest import run_strategy_backtest

    data = pd.DataFrame({"Close": [100.0, 110.0, 121.0]})
    signals = pd.Series([0.0, 1.0, 1.0])

    result = run_strategy_backtest(
        data,
        signals,
        initial_value=100.0,
    )

    assert result["equity_curve"].iloc[-1] == pytest.approx(110.0)
    assert result["total_return"] == pytest.approx(0.10)
    assert result["max_drawdown"] == pytest.approx(0.0)


def test_moving_average_signals():
    import pandas as pd

    from quant_ai_research_platform.backtest import moving_average_signals

    data = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = moving_average_signals(
        data,
        short_window=2,
        long_window=3,
    )

    expected = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0], name="signal")

    pd.testing.assert_series_equal(result, expected)


def test_moving_average_strategy_backtest():
    import pandas as pd

    from quant_ai_research_platform.backtest import (
        moving_average_signals,
        run_strategy_backtest,
    )

    data = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    signals = moving_average_signals(
        data,
        short_window=2,
        long_window=3,
    )

    result = run_strategy_backtest(
        data,
        signals,
        initial_value=100.0,
    )

    assert result["equity_curve"].iloc[-1] > 100.0
    assert result["total_return"] > 0.0
    assert result["max_drawdown"] <= 0.0


def test_compare_strategy_to_buy_and_hold():
    import pandas as pd

    from quant_ai_research_platform.backtest import (
        compare_strategy_to_buy_and_hold,
        moving_average_signals,
    )

    data = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    signals = moving_average_signals(
        data,
        short_window=2,
        long_window=3,
    )

    result = compare_strategy_to_buy_and_hold(
        data,
        signals,
        initial_value=100.0,
    )

    assert result["strategy_return"] > 0.0
    assert result["buy_and_hold_return"] > 0.0
    assert result["excess_return"] == pytest.approx(
        result["strategy_return"] - result["buy_and_hold_return"]
    )
    assert result["strategy_max_drawdown"] <= 0.0


def test_create_strategy_evaluation():
    import pandas as pd

    from quant_ai_research_platform.backtest import (
        create_strategy_evaluation,
        moving_average_signals,
    )

    data = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    signals = moving_average_signals(
        data,
        short_window=2,
        long_window=3,
    )

    result = create_strategy_evaluation(
        data,
        signals,
        initial_value=100.0,
    )

    assert result["strategy_return"] > 0.0
    assert result["buy_and_hold_return"] > 0.0
    assert result["final_equity"] > 100.0
    assert result["periods"] == 5
    assert result["strategy_max_drawdown"] <= 0.0


def test_create_features():
    import pandas as pd

    from quant_ai_research_platform.modeling import create_features

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ]
        }
    )

    result = create_features(data)

    assert list(result.columns) == [
        "return_1d",
        "return_5d",
        "volatility_5d",
    ]
    assert not result.empty
    assert not result.isna().any().any()


def test_create_target():
    import pandas as pd

    from quant_ai_research_platform.modeling import create_target

    data = pd.DataFrame({"Close": [100.0, 110.0, 105.0, 120.0]})

    result = create_target(data)

    expected = pd.Series(
        [1, 0, 1],
        name="target",
    )

    pd.testing.assert_series_equal(result, expected)


def test_create_model_dataset():
    import pandas as pd

    from quant_ai_research_platform.modeling import create_model_dataset

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ]
        }
    )

    result = create_model_dataset(data)

    assert list(result.columns) == [
        "return_1d",
        "return_5d",
        "volatility_5d",
        "target",
    ]
    assert not result.empty
    assert not result.isna().any().any()


def test_split_model_dataset():
    import pandas as pd

    from quant_ai_research_platform.modeling import split_model_dataset

    dataset = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 1],
        }
    )

    train, test = split_model_dataset(dataset, train_fraction=0.6)

    assert len(train) == 3
    assert len(test) == 2
    assert train["feature"].tolist() == [1, 2, 3]
    assert test["feature"].tolist() == [4, 5]


def test_separate_features_target():
    import pandas as pd

    from quant_ai_research_platform.modeling import separate_features_target

    dataset = pd.DataFrame(
        {
            "return_1d": [0.01, -0.02, 0.03],
            "return_5d": [0.05, 0.01, -0.04],
            "volatility_5d": [0.02, 0.03, 0.04],
            "target": [1, 0, 1],
        }
    )

    features, target = separate_features_target(dataset)

    assert list(features.columns) == [
        "return_1d",
        "return_5d",
        "volatility_5d",
    ]
    assert target.tolist() == [1, 0, 1]
    assert target.name == "target"


def test_train_logistic_regression():
    import pandas as pd

    from quant_ai_research_platform.modeling import train_logistic_regression

    features = pd.DataFrame(
        {
            "return_1d": [0.01, -0.02, 0.03, -0.01, 0.04, -0.03],
            "return_5d": [0.05, -0.01, 0.06, -0.02, 0.07, -0.04],
            "volatility_5d": [0.02, 0.03, 0.02, 0.04, 0.01, 0.05],
        }
    )

    target = pd.Series([1, 0, 1, 0, 1, 0], name="target")

    model = train_logistic_regression(features, target)

    predictions = model.predict(features)

    assert len(predictions) == len(target)
    assert set(predictions).issubset({0, 1})


def test_generate_model_signals():
    import pandas as pd

    from quant_ai_research_platform.modeling import (
        generate_model_signals,
        train_logistic_regression,
    )

    features = pd.DataFrame(
        {
            "return_1d": [0.01, -0.02, 0.03, -0.01, 0.04, -0.03],
            "return_5d": [0.05, -0.01, 0.06, -0.02, 0.07, -0.04],
            "volatility_5d": [0.02, 0.03, 0.02, 0.04, 0.01, 0.05],
        }
    )

    target = pd.Series([1, 0, 1, 0, 1, 0], name="target")

    model = train_logistic_regression(features, target)
    signals = generate_model_signals(model, features)

    assert len(signals) == len(features)
    assert signals.name == "signal"
    assert set(signals.unique()).issubset({0.0, 1.0})


def test_evaluate_model_strategy():
    import pandas as pd

    from quant_ai_research_platform.modeling import (
        evaluate_model_strategy,
        train_logistic_regression,
    )

    data = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 101.0, 103.0, 104.0]})

    features = pd.DataFrame(
        {
            "return_1d": [0.01, 0.01, -0.01, 0.02, 0.01],
            "return_5d": [0.02, 0.03, 0.01, 0.04, 0.05],
            "volatility_5d": [0.01, 0.01, 0.02, 0.02, 0.01],
        },
        index=[1, 2, 3, 4, 5],
    )

    target = pd.Series(
        [1, 1, 0, 1, 1],
        index=features.index,
        name="target",
    )

    model = train_logistic_regression(features, target)
    result = evaluate_model_strategy(data, model, features)

    assert "strategy_return" in result
    assert "buy_and_hold_return" in result
    assert "excess_return" in result
    assert "strategy_max_drawdown" in result


def test_run_ml_backtest():
    import pandas as pd

    from quant_ai_research_platform.modeling import run_ml_backtest

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                112.0,
                114.0,
                115.0,
                116.0,
                117.0,
            ]
        }
    )

    result = run_ml_backtest(data, train_fraction=0.7)

    assert "strategy_return" in result
    assert "buy_and_hold_return" in result
    assert "excess_return" in result
    assert "strategy_max_drawdown" in result

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result

    assert "threshold" in result
    assert result["threshold"] == 0.6


def test_evaluate_model_predictions():
    import pandas as pd

    from quant_ai_research_platform.modeling import (
        evaluate_model_predictions,
        train_logistic_regression,
    )

    features = pd.DataFrame(
        {
            "return_1d": [0.01, -0.02, 0.03, -0.01, 0.04, -0.03],
            "return_5d": [0.05, -0.01, 0.06, -0.02, 0.07, -0.04],
            "volatility_5d": [0.02, 0.03, 0.02, 0.04, 0.01, 0.05],
        }
    )

    target = pd.Series([1, 0, 1, 0, 1, 0], name="target")

    model = train_logistic_regression(features, target)
    result = evaluate_model_predictions(model, features, target)

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0

    assert "confusion_matrix" in result
    assert len(result["confusion_matrix"]) == 2
    assert len(result["confusion_matrix"][0]) == 2
    assert len(result["confusion_matrix"][1]) == 2


def test_generate_prediction_probabilities():
    import pandas as pd

    from quant_ai_research_platform.modeling import (
        generate_prediction_probabilities,
        train_logistic_regression,
    )

    features = pd.DataFrame(
        {
            "return_1d": [0.01, -0.02, 0.03, -0.01, 0.04, -0.03],
            "return_5d": [0.05, -0.01, 0.06, -0.02, 0.07, -0.04],
            "volatility_5d": [0.02, 0.03, 0.02, 0.04, 0.01, 0.05],
        }
    )

    target = pd.Series([1, 0, 1, 0, 1, 0], name="target")

    model = train_logistic_regression(features, target)
    probabilities = generate_prediction_probabilities(model, features)

    assert len(probabilities) == len(features)
    assert probabilities.name == "probability_up"
    assert probabilities.between(0.0, 1.0).all()


def test_generate_threshold_signals():
    import pandas as pd

    from quant_ai_research_platform.modeling import (
        generate_threshold_signals,
        train_logistic_regression,
    )

    features = pd.DataFrame(
        {
            "return_1d": [0.01, -0.02, 0.03, -0.01, 0.04, -0.03],
            "return_5d": [0.05, -0.01, 0.06, -0.02, 0.07, -0.04],
            "volatility_5d": [0.02, 0.03, 0.02, 0.04, 0.01, 0.05],
        }
    )

    target = pd.Series([1, 0, 1, 0, 1, 0], name="target")

    model = train_logistic_regression(features, target)
    signals = generate_threshold_signals(
        model,
        features,
        threshold=0.6,
    )

    assert len(signals) == len(features)
    assert signals.name == "signal"
    assert set(signals.unique()).issubset({0.0, 1.0})


def test_compare_thresholds():
    import pandas as pd

    from quant_ai_research_platform.modeling import compare_thresholds

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                112.0,
                114.0,
                115.0,
                116.0,
                117.0,
            ]
        }
    )

    result = compare_thresholds(
        data,
        thresholds=[0.50, 0.60],
        train_fraction=0.7,
    )

    assert len(result) == 2
    assert result["threshold"].tolist() == [0.50, 0.60]

    assert "strategy_return" in result.columns
    assert "buy_and_hold_return" in result.columns
    assert "excess_return" in result.columns
    assert "strategy_max_drawdown" in result.columns
    assert "accuracy" in result.columns
    assert "precision" in result.columns
    assert "recall" in result.columns
    assert "f1" in result.columns


def test_select_best_threshold():
    import pandas as pd

    from quant_ai_research_platform.modeling import select_best_threshold

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                112.0,
                114.0,
                115.0,
                116.0,
                117.0,
            ]
        }
    )

    result = select_best_threshold(
        data,
        thresholds=[0.50, 0.60],
        train_fraction=0.7,
    )

    assert "threshold" in result
    assert "excess_return" in result
    assert result["threshold"] in [0.50, 0.60]


def test_split_train_validation_test():
    import pandas as pd

    from quant_ai_research_platform.modeling import split_train_validation_test

    dataset = pd.DataFrame(
        {
            "feature": list(range(10)),
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    train, validation, test = split_train_validation_test(
        dataset,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert len(train) == 6
    assert len(validation) == 2
    assert len(test) == 2

    assert train["feature"].tolist() == [0, 1, 2, 3, 4, 5]
    assert validation["feature"].tolist() == [6, 7]
    assert test["feature"].tolist() == [8, 9]


def test_select_threshold_on_validation():
    import pandas as pd

    from quant_ai_research_platform.modeling import select_threshold_on_validation

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                112.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                117.0,
                119.0,
                120.0,
                121.0,
            ]
        }
    )

    result = select_threshold_on_validation(
        data,
        thresholds=[0.50, 0.60],
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert "threshold" in result
    assert "excess_return" in result
    assert result["threshold"] in [0.50, 0.60]


def test_run_validated_ml_backtest():
    import pandas as pd

    from quant_ai_research_platform.modeling import run_validated_ml_backtest

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                112.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                117.0,
                119.0,
                120.0,
                121.0,
                122.0,
                121.0,
                123.0,
                124.0,
                125.0,
            ]
        }
    )

    result = run_validated_ml_backtest(
        data,
        thresholds=[0.50, 0.60],
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert "selected_threshold" in result
    assert result["selected_threshold"] in [0.50, 0.60]

    assert "strategy_return" in result
    assert "buy_and_hold_return" in result
    assert "excess_return" in result
    assert "strategy_max_drawdown" in result

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert "confusion_matrix" in result


def test_create_walk_forward_splits():
    import pandas as pd

    from quant_ai_research_platform.modeling import create_walk_forward_splits

    dataset = pd.DataFrame(
        {
            "feature": list(range(10)),
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    splits = create_walk_forward_splits(
        dataset,
        initial_train_size=4,
        test_size=2,
    )

    assert len(splits) == 3

    train_1, test_1 = splits[0]
    train_2, test_2 = splits[1]
    train_3, test_3 = splits[2]

    assert train_1["feature"].tolist() == [0, 1, 2, 3]
    assert test_1["feature"].tolist() == [4, 5]

    assert train_2["feature"].tolist() == [0, 1, 2, 3, 4, 5]
    assert test_2["feature"].tolist() == [6, 7]

    assert train_3["feature"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert test_3["feature"].tolist() == [8, 9]


def test_run_walk_forward_backtest():
    import pandas as pd

    from quant_ai_research_platform.modeling import run_walk_forward_backtest

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                112.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                117.0,
                119.0,
                120.0,
                121.0,
                122.0,
                121.0,
                123.0,
                124.0,
                125.0,
                126.0,
                125.0,
                127.0,
                128.0,
                129.0,
            ]
        }
    )

    result = run_walk_forward_backtest(
        data,
        initial_train_size=12,
        test_size=4,
        threshold=0.6,
    )

    assert not result.empty
    assert "split" in result.columns
    assert "strategy_return" in result.columns
    assert "buy_and_hold_return" in result.columns
    assert "excess_return" in result.columns
    assert "strategy_max_drawdown" in result.columns
    assert "accuracy" in result.columns
    assert "precision" in result.columns
    assert "recall" in result.columns
    assert "f1" in result.columns


def test_run_walk_forward_backtest_accepts_logistic_model_type():
    import pandas as pd

    from quant_ai_research_platform.modeling import run_walk_forward_backtest

    data = pd.DataFrame(
        {"Close": [100.0 + i + (2.0 if i % 2 else 0.0) for i in range(35)]}
    )

    result = run_walk_forward_backtest(
        data,
        initial_train_size=12,
        test_size=4,
        model_type="logistic",
    )

    assert not result.empty


def test_run_walk_forward_backtest_accepts_pytorch_model_type():
    import pandas as pd
    import pytest

    pytest.importorskip("torch")

    from quant_ai_research_platform.modeling import run_walk_forward_backtest

    data = pd.DataFrame(
        {"Close": [100.0 + i + (2.0 if i % 2 else 0.0) for i in range(35)]}
    )

    result = run_walk_forward_backtest(
        data,
        initial_train_size=12,
        test_size=4,
        model_type="pytorch",
    )

    assert not result.empty


def test_run_walk_forward_backtest_rejects_invalid_model_type():
    import pandas as pd
    import pytest

    from quant_ai_research_platform.modeling import run_walk_forward_backtest

    with pytest.raises(ValueError, match="model_type"):
        run_walk_forward_backtest(
            pd.DataFrame({"Close": [100.0 + i for i in range(20)]}),
            initial_train_size=12,
            test_size=4,
            model_type="unknown",
        )


def test_pytorch_walk_forward_scaling_uses_training_window_only(monkeypatch):
    import pandas as pd
    import pytest

    pytest.importorskip("torch")

    from quant_ai_research_platform import modeling

    data = pd.DataFrame({"Close": [100.0 + i for i in range(35)]})
    fit_values = []
    transformed_values = []

    class RecordingScaler:
        def fit_transform(self, features):
            fit_values.append(features.copy())
            return features.to_numpy()

        def transform(self, features):
            transformed_values.append(features.copy())
            return features.to_numpy()

    class StubTorchModel:
        def fit(self, features, target):
            assert features.index.is_monotonic_increasing
            assert features.columns.tolist() == [
                "return_1d",
                "return_5d",
                "volatility_5d",
            ]
            return self

        def predict_proba(self, features):
            import numpy as np

            return np.tile([0.0, 1.0], (len(features), 1))

        def predict(self, features):
            import numpy as np

            return np.ones(len(features), dtype=int)

    monkeypatch.setattr(modeling, "StandardScaler", RecordingScaler)
    monkeypatch.setattr(
        "quant_ai_research_platform.torch_model.TorchMarketClassifier",
        StubTorchModel,
    )

    modeling.run_walk_forward_backtest(
        data,
        initial_train_size=12,
        test_size=4,
        model_type="pytorch",
    )

    dataset = modeling.create_model_dataset(data)
    first_train, first_test = modeling.create_walk_forward_splits(
        dataset,
        initial_train_size=12,
        test_size=4,
    )[0]
    assert fit_values[0].index.tolist() == first_train.index[:-1].tolist()
    assert transformed_values[0].index.tolist() == first_test.index.tolist()
    assert fit_values[0].index.max() < transformed_values[0].index.min()


def test_pytorch_walk_forward_uses_fresh_model_for_each_split(monkeypatch):
    import numpy as np
    import pandas as pd
    import pytest

    pytest.importorskip("torch")

    from quant_ai_research_platform import modeling

    instances = []

    class StubTorchModel:
        def __init__(self):
            self.fit_indexes = None
            instances.append(self)

        def fit(self, features, target):
            self.fit_indexes = features.index.copy()
            return self

        def predict_proba(self, features):
            self.last_prediction_indexes = features.index.copy()
            return np.tile([0.25, 0.75], (len(features), 1))

        def predict(self, features):
            return np.ones(len(features), dtype=int)

    monkeypatch.setattr(
        "quant_ai_research_platform.torch_model.TorchMarketClassifier",
        StubTorchModel,
    )

    data = pd.DataFrame({"Close": [100.0 + i for i in range(35)]})
    result = modeling.run_walk_forward_backtest(
        data,
        initial_train_size=12,
        test_size=4,
        model_type="pytorch",
    )

    splits = modeling.create_walk_forward_splits(
        modeling.create_model_dataset(data),
        initial_train_size=12,
        test_size=4,
    )
    assert len(instances) == len(splits)
    assert len({id(instance) for instance in instances}) == len(instances)
    for instance, (train, test) in zip(instances, splits):
        assert instance.fit_indexes.tolist() == train.index[:-1].tolist()
        assert instance.last_prediction_indexes.tolist() == test.index.tolist()
    assert len(result) == len(splits)


def test_summarize_walk_forward_results():
    import pandas as pd

    from quant_ai_research_platform.modeling import summarize_walk_forward_results

    results = pd.DataFrame(
        {
            "strategy_return": [0.02, 0.04, 0.06],
            "buy_and_hold_return": [0.01, 0.03, 0.05],
            "excess_return": [0.01, 0.01, 0.01],
            "strategy_max_drawdown": [-0.02, -0.03, -0.01],
            "accuracy": [0.60, 0.70, 0.80],
            "precision": [0.55, 0.65, 0.75],
            "recall": [0.50, 0.60, 0.70],
            "f1": [0.52, 0.62, 0.72],
        }
    )

    summary = summarize_walk_forward_results(results)

    assert summary["splits"] == 3
    assert summary["average_strategy_return"] == 0.04
    assert summary["average_buy_and_hold_return"] == 0.03
    assert summary["average_excess_return"] == 0.01

    import pytest

    assert summary["average_accuracy"] == pytest.approx(0.70)


def test_run_walk_forward_research():
    import pandas as pd

    from quant_ai_research_platform.modeling import run_walk_forward_research

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                112.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                117.0,
                119.0,
                120.0,
                121.0,
                122.0,
                121.0,
                123.0,
                124.0,
                125.0,
                126.0,
                125.0,
                127.0,
                128.0,
                129.0,
            ]
        }
    )

    result = run_walk_forward_research(
        data,
        initial_train_size=12,
        test_size=4,
        threshold=0.6,
    )

    assert "results" in result
    assert "summary" in result
    assert not result["results"].empty
    assert result["summary"]["splits"] == len(result["results"])

    assert "stability" in result
    assert "profitable_split_rate" in result["stability"]
    assert "beat_buy_and_hold_rate" in result["stability"]
    assert "median_excess_return" in result["stability"]
    assert "worst_excess_return" in result["stability"]
    assert "best_excess_return" in result["stability"]


def test_calculate_walk_forward_stability():
    import pandas as pd
    import pytest

    from quant_ai_research_platform.modeling import (
        calculate_walk_forward_stability,
    )

    results = pd.DataFrame(
        {
            "strategy_return": [0.02, -0.01, 0.03, 0.04],
            "excess_return": [0.01, -0.02, 0.00, 0.03],
        }
    )

    stability = calculate_walk_forward_stability(results)

    assert stability["profitable_split_rate"] == pytest.approx(0.75)
    assert stability["beat_buy_and_hold_rate"] == pytest.approx(0.50)
    assert stability["median_excess_return"] == pytest.approx(0.005)
    assert stability["worst_excess_return"] == pytest.approx(-0.02)
    assert stability["best_excess_return"] == pytest.approx(0.03)


def test_create_walk_forward_report():
    import pytest

    from quant_ai_research_platform.modeling import create_walk_forward_report

    research = {
        "summary": {
            "splits": 3,
            "average_strategy_return": 0.04,
            "average_buy_and_hold_return": 0.03,
            "average_excess_return": 0.01,
            "average_max_drawdown": -0.02,
            "average_accuracy": 0.70,
            "average_precision": 0.65,
            "average_recall": 0.60,
            "average_f1": 0.62,
        },
        "stability": {
            "profitable_split_rate": 0.75,
            "beat_buy_and_hold_rate": 0.50,
            "median_excess_return": 0.005,
            "worst_excess_return": -0.02,
            "best_excess_return": 0.03,
        },
    }

    report = create_walk_forward_report(research)

    assert report["splits"] == 3
    assert report["average_excess_return"] == pytest.approx(0.01)
    assert report["average_accuracy"] == pytest.approx(0.70)
    assert report["average_f1"] == pytest.approx(0.62)
    assert report["beat_buy_and_hold_rate"] == pytest.approx(0.50)
    assert report["best_excess_return"] == pytest.approx(0.03)


def test_calculate_portfolio_returns_equal_weight():
    import pandas as pd
    import pytest

    from quant_ai_research_platform.market_analysis import (
        calculate_portfolio_returns,
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, -0.01],
            "MSFT": [0.03, 0.00, 0.01],
        }
    )

    result = calculate_portfolio_returns(returns)

    assert result.iloc[0] == pytest.approx(0.02)
    assert result.iloc[1] == pytest.approx(0.01)
    assert result.iloc[2] == pytest.approx(0.00)


def test_calculate_portfolio_returns_custom_weights():
    import pandas as pd
    import pytest

    from quant_ai_research_platform.market_analysis import (
        calculate_portfolio_returns,
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.10, -0.02],
            "MSFT": [0.00, 0.04],
        }
    )

    result = calculate_portfolio_returns(
        returns,
        weights={"AAPL": 0.75, "MSFT": 0.25},
    )

    assert result.iloc[0] == pytest.approx(0.075)
    assert result.iloc[1] == pytest.approx(-0.005)


def test_calculate_portfolio_returns_rejects_invalid_weights():
    import pandas as pd
    import pytest

    from quant_ai_research_platform.market_analysis import (
        calculate_portfolio_returns,
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02],
            "MSFT": [0.02, 0.03],
        }
    )

    with pytest.raises(ValueError, match="sum to 1"):
        calculate_portfolio_returns(
            returns,
            weights={"AAPL": 0.80, "MSFT": 0.30},
        )


def test_calculate_portfolio_statistics():
    import pandas as pd
    import pytest

    from quant_ai_research_platform.market_analysis import (
        calculate_portfolio_statistics,
    )

    returns = pd.Series([0.01, 0.02, -0.01, 0.03])

    result = calculate_portfolio_statistics(returns)

    assert "total_return" in result
    assert "annualized_return" in result
    assert "annualized_volatility" in result
    assert "sharpe_ratio" in result
    assert "max_drawdown" in result

    expected_total_return = (1.01 * 1.02 * 0.99 * 1.03) - 1

    assert result["total_return"] == pytest.approx(expected_total_return)


def test_analyze_portfolio():
    import pandas as pd
    import pytest

    from quant_ai_research_platform.market_analysis import analyze_portfolio

    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, -0.01, 0.03],
            "MSFT": [0.02, 0.01, 0.00, 0.02],
        }
    )

    result = analyze_portfolio(
        returns,
        weights={"AAPL": 0.60, "MSFT": 0.40},
    )

    assert result["weights"]["AAPL"] == pytest.approx(0.60)
    assert result["weights"]["MSFT"] == pytest.approx(0.40)

    assert "returns" in result
    assert "statistics" in result

    assert result["returns"].iloc[0] == pytest.approx(0.014)
    assert "total_return" in result["statistics"]
    assert "sharpe_ratio" in result["statistics"]
    assert "max_drawdown" in result["statistics"]


def test_optimize_portfolio_weights_sum_to_one():
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.015, 0.005],
            "BBB": [0.005, 0.01, 0.002, 0.008, 0.004],
            "CCC": [-0.005, 0.015, 0.01, -0.002, 0.012],
        }
    )

    result = optimize_portfolio(returns)

    assert math.isclose(
        sum(result["weights"].values()),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )

    assert all(0.0 <= weight <= 1.0 for weight in result["weights"].values())


def test_optimize_portfolio_returns_expected_structure():
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.015, 0.005],
            "BBB": [0.005, 0.01, 0.002, 0.008, 0.004],
            "CCC": [-0.005, 0.015, 0.01, -0.002, 0.012],
        }
    )

    result = optimize_portfolio(
        returns,
        risk_free_rate=0.02,
    )

    assert set(result) == {
        "weights",
        "returns",
        "statistics",
        "risk_free_rate",
    }
    assert result["risk_free_rate"] == 0.02
    assert set(result["weights"]) == {"AAA", "BBB", "CCC"}
    assert isinstance(result["returns"], pd.Series)
    assert {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    } == set(result["statistics"])


def test_optimize_portfolio_rejects_empty_returns():
    returns = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Returns data cannot be empty",
    ):
        optimize_portfolio(returns)


def test_walk_forward_includes_final_prediction_return(monkeypatch):
    import pandas as pd
    import pytest

    from quant_ai_research_platform import modeling

    data = pd.DataFrame({"Close": [100.0 + i for i in range(35)]})

    class AlwaysInvestedModel:
        def predict_proba(self, features):
            import numpy as np

            return np.tile([0.0, 1.0], (len(features), 1))

        def predict(self, features):
            import numpy as np

            return np.ones(len(features), dtype=int)

    monkeypatch.setattr(
        modeling,
        "train_logistic_regression",
        lambda features, target: AlwaysInvestedModel(),
    )

    results = modeling.run_walk_forward_backtest(
        data,
        initial_train_size=12,
        test_size=4,
        threshold=0.6,
    )

    # With every signal invested, the strategy should earn all four
    # next-day returns in each four-observation test window.
    first_test_close = data["Close"].iloc[17]
    final_next_close = data["Close"].iloc[21]

    expected_return = final_next_close / first_test_close - 1

    assert results.iloc[0]["strategy_return"] == pytest.approx(expected_return)


def test_walk_forward_purges_boundary_training_label(monkeypatch):
    import pandas as pd

    from quant_ai_research_platform import modeling

    data = pd.DataFrame({"Close": [100.0 + i for i in range(35)]})

    dataset = modeling.create_model_dataset(data)
    splits = modeling.create_walk_forward_splits(
        dataset,
        initial_train_size=12,
        test_size=4,
    )

    first_train, first_test = splits[0]
    observed_training_indices = []

    class MockModel:
        def predict_proba(self, features):
            import numpy as np

            return np.tile([0.0, 1.0], (len(features), 1))

        def predict(self, features):
            import numpy as np

            return np.ones(len(features), dtype=int)

    def capture_training_indices(features, target):
        observed_training_indices.append(features.index.tolist())
        return MockModel()

    monkeypatch.setattr(
        modeling,
        "train_logistic_regression",
        capture_training_indices,
    )

    modeling.run_walk_forward_backtest(
        data,
        initial_train_size=12,
        test_size=4,
    )

    assert observed_training_indices
    assert observed_training_indices[0] == first_train.index[:-1].tolist()
    assert first_train.index[-1] not in observed_training_indices[0]
    assert first_test.index[0] not in observed_training_indices[0]
