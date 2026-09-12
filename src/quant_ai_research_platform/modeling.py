import pandas as pd


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
