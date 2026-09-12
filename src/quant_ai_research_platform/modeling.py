import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)

    features["return_1d"] = data["Close"].pct_change()
    features["return_5d"] = data["Close"].pct_change(5)
    features["volatility_5d"] = features["return_1d"].rolling(5).std()

    return features.dropna()

