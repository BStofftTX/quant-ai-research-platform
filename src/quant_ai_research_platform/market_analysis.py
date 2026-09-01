import math

import pandas as pd
import yfinance as yf


TRADING_DAYS = 252
BENCHMARK = "SPY"


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


def calculate_benchmark_metrics(
    stock_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
) -> dict:
    """Compare a stock's daily returns with the benchmark."""
    stock_returns = stock_data["Close"].dropna().pct_change()
    benchmark_returns = benchmark_data["Close"].dropna().pct_change()

    returns = pd.concat(
        [stock_returns, benchmark_returns],
        axis=1,
        join="inner",
    ).dropna()

    returns.columns = ["stock", "benchmark"]

    correlation = returns["stock"].corr(returns["benchmark"])

    benchmark_variance = returns["benchmark"].var()
    covariance = returns["stock"].cov(returns["benchmark"])

    beta = (
        covariance / benchmark_variance
        if benchmark_variance != 0
        else float("nan")
    )

    return {
        "correlation": correlation,
        "beta": beta,
    }


def main() -> None:
    symbol = input("Enter ticker symbol [AAPL]: ").strip().upper() or "AAPL"

    print(f"\nDownloading market data for {symbol} and {BENCHMARK}...")

    stock_data = get_market_data(symbol)
    benchmark_data = get_market_data(BENCHMARK)

    stock_stats = calculate_statistics(stock_data)
    benchmark_stats = calculate_statistics(benchmark_data)

    comparison = calculate_benchmark_metrics(
        stock_data,
        benchmark_data,
    )

    excess_return = (
        stock_stats["total_return"]
        - benchmark_stats["total_return"]
    )

    print(f"\nQuantitative Analysis: {symbol}")
    print("-" * 45)
    print(f"{symbol} total return:       {stock_stats['total_return']:.2%}")
    print(f"{BENCHMARK} total return:        {benchmark_stats['total_return']:.2%}")
    print(f"Excess return vs SPY:    {excess_return:.2%}")
    print(f"{symbol} CAGR:               {stock_stats['cagr']:.2%}")
    print(f"{symbol} volatility:         {stock_stats['annualized_volatility']:.2%}")
    print(f"{symbol} Sharpe ratio:        {stock_stats['sharpe_ratio']:.2f}")
    print(f"{symbol} maximum drawdown:    {stock_stats['max_drawdown']:.2%}")
    print(f"Correlation with SPY:    {comparison['correlation']:.2f}")
    print(f"Beta vs SPY:             {comparison['beta']:.2f}")


if __name__ == "__main__":
    main()
