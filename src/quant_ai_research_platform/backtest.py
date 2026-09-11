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

def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        raise ValueError("equity_curve must not be empty")

    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1

    return drawdown.min()

def summarize_backtest(
    data: pd.DataFrame,
    benchmark_return: float,
    periods_per_year: int = 252,
) -> dict:
    if len(data) < 2:
        raise ValueError("data must contain at least two rows")

    total_return = buy_and_hold_return(data)
    periods = len(data) - 1

    equity_curve = data["Close"] / data["Close"].iloc[0]

    return {
        "total_return": total_return,
        "annualized_return": annualized_return(
            total_return,
            periods=periods,
            periods_per_year=periods_per_year,
        ),
        "benchmark_return": benchmark_return,
        "excess_return": excess_return(total_return, benchmark_return),
        "max_drawdown": max_drawdown(equity_curve),
    }

def calculate_strategy_returns(
    data: pd.DataFrame,
    signals: pd.Series,
) -> pd.Series:
    if len(data) != len(signals):
        raise ValueError("data and signals must have the same length")

    market_returns = data["Close"].pct_change().fillna(0.0)
    strategy_returns = market_returns * signals.shift(1).fillna(0.0)

    return strategy_returns


def build_equity_curve(
    strategy_returns: pd.Series,
    initial_value: float = 1.0,
) -> pd.Series:
    if strategy_returns.empty:
        raise ValueError("strategy_returns must not be empty")

    return initial_value * (1.0 + strategy_returns).cumprod()

