import pandas as pd

from sklearn.linear_model import LogisticRegression

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
