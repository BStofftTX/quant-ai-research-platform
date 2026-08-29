import math

import pandas as pd
import yfinance as yf


TRADING_DAYS = 252


def get_market_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Download historical market data for a ticker symbol."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    if data.empty:
        raise ValueError(f"No market data returned for {symbol}")

    return data


def calculate_statistics(data: pd.DataFrame) -> dict:
    """Calculate quantitative statistics from valid closing prices."""
    close = data["Close"].dropna()

    if len(close) < 2:
        raise ValueError("Not enough valid closing prices to calculate statistics")

    daily_returns = close.pct_change().dropna()

    starting_price = close.iloc[0]
    ending_price = close.iloc[-1]

    total_return = (ending_price / starting_price) - 1
    years = len(daily_returns) / TRADING_DAYS
    cagr = (ending_price / starting_price) ** (1 / years) - 1

    daily_volatility = daily_returns.std()
    annualized_volatility = daily_volatility * math.sqrt(TRADING_DAYS)

    annualized_return = daily_returns.mean() * TRADING_DAYS
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility != 0
        else float("nan")
    )

    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / running_max) - 1
    max_drawdown = drawdowns.min()

    return {
        "starting_price": starting_price,
        "ending_price": ending_price,
        "total_return": total_return,
        "cagr": cagr,
        "average_daily_return": daily_returns.mean(),
        "daily_volatility": daily_volatility,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def main() -> None:
    symbol = input("Enter ticker symbol [SPY]: ").strip().upper() or "SPY"

    print(f"\nDownloading market data for {symbol}...")

    data = get_market_data(symbol)
    statistics = calculate_statistics(data)

    print(f"\nQuantitative Analysis: {symbol}")
    print("-" * 40)
    print(f"Starting price:        ${statistics['starting_price']:.2f}")
    print(f"Ending price:          ${statistics['ending_price']:.2f}")
    print(f"Total return:           {statistics['total_return']:.2%}")
    print(f"CAGR:                   {statistics['cagr']:.2%}")
    print(f"Average daily return:   {statistics['average_daily_return']:.4%}")
    print(f"Daily volatility:       {statistics['daily_volatility']:.4%}")
    print(f"Annualized volatility:  {statistics['annualized_volatility']:.2%}")
    print(f"Sharpe ratio:           {statistics['sharpe_ratio']:.2f}")
    print(f"Maximum drawdown:       {statistics['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
