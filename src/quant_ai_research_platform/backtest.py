import pandas as pd


def buy_and_hold_return(data: pd.DataFrame) -> float:
    if data.empty:
        raise ValueError("data must not be empty")

    start_price = data["Close"].iloc[0]
    end_price = data["Close"].iloc[-1]

    return (end_price / start_price) - 1

