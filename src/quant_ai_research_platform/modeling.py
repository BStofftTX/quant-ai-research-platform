import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from quant_ai_research_platform.backtest import compare_strategy_to_buy_and_hold

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)

    features["return_1d"] = data["Close"].pct_change()
    features["return_5d"] = data["Close"].pct_change(5)
    features["volatility_5d"] = features["return_1d"].rolling(5).std()

    return features.dropna()

def create_target(data: pd.DataFrame) -> pd.Series:
    future_return = data["Close"].shift(-1) / data["Close"] - 1
    target = (future_return > 0).astype(int)

    return target.iloc[:-1].rename("target")

def create_model_dataset(data: pd.DataFrame) -> pd.DataFrame:
    features = create_features(data)
    target = create_target(data)

    dataset = features.join(target, how="inner")

    return dataset.dropna()

def split_model_dataset(
    dataset: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    split_index = int(len(dataset) * train_fraction)

    train = dataset.iloc[:split_index].copy()
    test = dataset.iloc[split_index:].copy()

    return train, test

def separate_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if "target" not in dataset.columns:
        raise ValueError("dataset must contain a target column")

    features = dataset.drop(columns=["target"]).copy()
    target = dataset["target"].copy()

    return features, target

def train_logistic_regression(
    features: pd.DataFrame,
    target: pd.Series,
) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(features, target)

    return model

def generate_model_signals(
    model: LogisticRegression,
    features: pd.DataFrame,
) -> pd.Series:
    predictions = model.predict(features)

    return pd.Series(
        predictions,
        index=features.index,
        name="signal",
        dtype=float,
    )

def evaluate_model_strategy(
    data: pd.DataFrame,
    model: LogisticRegression,
    features: pd.DataFrame,
) -> dict:
    signals = generate_model_signals(model, features)

    aligned_data = data.loc[features.index]

    return compare_strategy_to_buy_and_hold(
        aligned_data,
        signals,
    )

def run_ml_backtest(
    data: pd.DataFrame,
    train_fraction: float = 0.8,
    threshold: float = 0.6,
) -> dict:
    dataset = create_model_dataset(data)

    train, test = split_model_dataset(
        dataset,
        train_fraction=train_fraction,
    )

    train_features, train_target = separate_features_target(train)
    test_features, test_target = separate_features_target(test)

    model = train_logistic_regression(
        train_features,
        train_target,
    )

    signals = generate_threshold_signals(
        model,
        test_features,
        threshold=threshold,
    )

    aligned_data = data.loc[test_features.index]

    trading_results = compare_strategy_to_buy_and_hold(
        aligned_data,
        signals,
    )

    prediction_results = evaluate_model_predictions(
        model,
        test_features,
        test_target,
    )

    return {
        **trading_results,
        **prediction_results,
        "threshold": threshold,
    }

def evaluate_model_predictions(
    model: LogisticRegression,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict:
    predictions = model.predict(features)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])

    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "confusion_matrix": matrix.tolist(),
    }
def generate_prediction_probabilities(
    model: LogisticRegression,
    features: pd.DataFrame,
) -> pd.Series:
    probabilities = model.predict_proba(features)[:, 1]

    return pd.Series(
        probabilities,
        index=features.index,
        name="probability_up",
    )

def generate_threshold_signals(
    model: LogisticRegression,
    features: pd.DataFrame,
    threshold: float = 0.6,
) -> pd.Series:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    probabilities = generate_prediction_probabilities(model, features)

    return (probabilities >= threshold).astype(float).rename("signal")

def compare_thresholds(
    data: pd.DataFrame,
    thresholds: list[float] | None = None,
    train_fraction: float = 0.8,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65]

    results = []

    for threshold in thresholds:
        result = run_ml_backtest(
            data,
            train_fraction=train_fraction,
            threshold=threshold,
        )

        results.append(
            {
                "threshold": threshold,
                "strategy_return": result["strategy_return"],
                "buy_and_hold_return": result["buy_and_hold_return"],
                "excess_return": result["excess_return"],
                "strategy_max_drawdown": result["strategy_max_drawdown"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
        )

    return pd.DataFrame(results)

