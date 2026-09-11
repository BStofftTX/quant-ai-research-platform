import pandas as pd
def annualized_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    if periods <= 0:
        raise ValueError("periods must be positive")

    return (1 + total_return) ** (periods_per_year / periods) - 1

def buy_and_hold_return(data: pd.DataFrame) -> float:
    if data.empty:
        raise ValueError("data must not be empty")

    start_price = data["Close"].iloc[0]
    end_price = data["Close"].iloc[-1]

    return (end_price / start_price) - 1

def excess_return(strategy_return: float, benchmark_return: float) -> float:
    return strategy_return - benchmark_return
