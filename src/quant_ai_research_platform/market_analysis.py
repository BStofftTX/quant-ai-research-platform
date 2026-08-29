import pandas as pd
import yfinance as yf


def get_market_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Download historical market data for a ticker symbol."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    if data.empty:
        raise ValueError(f"No market data returned for {symbol}")

    return data


def calculate_statistics(data: pd.DataFrame) -> dict:
    """Calculate basic quantitative statistics from valid closing prices."""
    close = data["Close"].dropna()

    if len(close) < 2:
        raise ValueError("Not enough valid closing prices to calculate statistics")

    daily_returns = close.pct_change().dropna()

    starting_price = close.iloc[0]
    ending_price = close.iloc[-1]

    statistics = {
        "starting_price": starting_price,
        "ending_price": ending_price,
        "total_return": (ending_price / starting_price) - 1,
        "average_daily_return": daily_returns.mean(),
        "daily_volatility": daily_returns.std(),
        "annualized_volatility": daily_returns.std() * (252 ** 0.5),
    }

    return statistics


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
    print(f"Average daily return:   {statistics['average_daily_return']:.4%}")
    print(f"Daily volatility:       {statistics['daily_volatility']:.4%}")
    print(f"Annualized volatility:  {statistics['annualized_volatility']:.2%}")


if __name__ == "__main__":
    main()
