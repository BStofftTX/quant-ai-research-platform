import math

import pandas as pd
import yfinance as yf
from quant_ai_research_platform.database import save_market_data


TRADING_DAYS = 252
BENCHMARK = "SPY"
DEFAULT_SYMBOLS = (
    "AAPL,META,AMZN,GOOG,BABA,MSFT,TSLA,NVDA,INTC,"
    "PWR,TSM,AMD,CRWD,PANW,ADBE,NNE,SPCX,ELVR"
)


def get_market_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Download historical market data for a ticker symbol."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    if data.empty:
        raise ValueError(f"No market data returned for {symbol}")

    save_market_data(symbol, data)

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
    """Compare a stock with the benchmark using aligned trading dates."""
    aligned = pd.concat(
        [
            stock_data["Close"].rename("stock"),
            benchmark_data["Close"].rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2:
        raise ValueError("Not enough overlapping market data")

    stock_total_return = (
        aligned["stock"].iloc[-1] / aligned["stock"].iloc[0]
    ) - 1

    benchmark_total_return = (
        aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0]
    ) - 1

    returns = aligned.pct_change().dropna()

    correlation = returns["stock"].corr(returns["benchmark"])

    benchmark_variance = returns["benchmark"].var()
    covariance = returns["stock"].cov(returns["benchmark"])

    beta = (
        covariance / benchmark_variance
        if benchmark_variance != 0
        else float("nan")
    )

    annualized_stock_return = returns["stock"].mean() * TRADING_DAYS
    annualized_benchmark_return = returns["benchmark"].mean() * TRADING_DAYS

    alpha = annualized_stock_return - (beta * annualized_benchmark_return)

    return {
        "start_date": aligned.index[0].date(),
        "end_date": aligned.index[-1].date(),
        "observations": len(aligned),
        "stock_total_return": stock_total_return,
        "benchmark_total_return": benchmark_total_return,
        "correlation": correlation,
        "beta": beta,
        "alpha": alpha,
    }

def analyze_symbol(
    symbol: str,
    benchmark_data: pd.DataFrame,
    _benchmark_stats: dict,
) -> dict:
    """Analyze one security relative to the benchmark."""
    stock_data = get_market_data(symbol)
    stock_stats = calculate_statistics(stock_data)
    comparison = calculate_benchmark_metrics(stock_data, benchmark_data)

    excess_return = (
        comparison["stock_total_return"]
        - comparison["benchmark_total_return"]
    )

    print(f"\n{symbol} vs {BENCHMARK}")
    print("-" * 40)
    print(
        f"Analysis period:         "
        f"{comparison['start_date']} to {comparison['end_date']}"
    )
    print(f"Aligned observations:    {comparison['observations']}")
    print(f"{symbol} total return:       {comparison['stock_total_return']:.2%}")
    print(f"{BENCHMARK} total return:        {comparison['benchmark_total_return']:.2%}")
    print(f"Excess return:           {excess_return:.2%}")
    print(f"CAGR:                    {stock_stats['cagr']:.2%}")
    print(f"Annualized volatility:   {stock_stats['annualized_volatility']:.2%}")
    print(f"Sharpe ratio:            {stock_stats['sharpe_ratio']:.2f}")
    print(f"Maximum drawdown:        {stock_stats['max_drawdown']:.2%}")
    print(f"Correlation with SPY:    {comparison['correlation']:.2f}")
    print(f"Beta vs SPY:             {comparison['beta']:.2f}")
    print(f"Annualized alpha:        {comparison['alpha']:.2%}")

    return {
        "symbol": symbol,
        "start_date": comparison["start_date"],
        "end_date": comparison["end_date"],
        "observations": comparison["observations"],
        "total_return": comparison["stock_total_return"],
        "benchmark_return": comparison["benchmark_total_return"],
        "excess_return": excess_return,
        "cagr": stock_stats["cagr"],
        "annualized_volatility": stock_stats["annualized_volatility"],
        "sharpe_ratio": stock_stats["sharpe_ratio"],
        "max_drawdown": stock_stats["max_drawdown"],
        "correlation": comparison["correlation"],
        "beta": comparison["beta"],
        "alpha": comparison["alpha"],
    }

def main() -> None:
    raw_symbols = input(
        f"Enter ticker symbols separated by commas [{DEFAULT_SYMBOLS}]: "
    ).strip()

    if not raw_symbols:
        raw_symbols = DEFAULT_SYMBOLS

    symbols = [
        symbol.strip().upper()
        for symbol in raw_symbols.split(",")
        if symbol.strip()
    ]

    symbols = [symbol for symbol in symbols if symbol != BENCHMARK]

    if not symbols:
        raise ValueError("Enter at least one ticker other than SPY")

    print(f"\nDownloading benchmark data for {BENCHMARK}...")
    benchmark_data = get_market_data(BENCHMARK)
    benchmark_stats = calculate_statistics(benchmark_data)

    results = []

    for symbol in symbols:
        print(f"\nDownloading market data for {symbol}...")

        try:
            result = analyze_symbol(
                symbol,
                benchmark_data,
                benchmark_stats,
            )
            results.append(result)

        except ValueError as error:
            print(f"Unable to analyze {symbol}: {error}")

    if not results:
        raise ValueError("No securities were successfully analyzed")

    comparison_table = pd.DataFrame(results)

    comparison_table = comparison_table.sort_values(
        by="sharpe_ratio",
        ascending=False,
    ).reset_index(drop=True)

    comparison_table.index = comparison_table.index + 1
    comparison_table.index.name = "Rank"

    print("\n")
    print("=" * 100)
    print("SECURITY COMPARISON — RANKED BY SHARPE RATIO")
    print("=" * 100)

    display_columns = [
        "symbol",
        "start_date",
        "end_date",
        "observations",
        "total_return",
        "excess_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "beta",
        "alpha",
    ]

    print(
        comparison_table[display_columns].to_string(
            formatters={
                "total_return": lambda value: f"{value:.2%}",
                "excess_return": lambda value: f"{value:.2%}",
                "annualized_volatility": lambda value: f"{value:.2%}",
                "sharpe_ratio": lambda value: f"{value:.2f}",
                "max_drawdown": lambda value: f"{value:.2%}",
                "beta": lambda value: f"{value:.2f}",
                "alpha": lambda value: f"{value:.2%}",
            }
        )
    )

    output_file = "security_comparison.csv"
    comparison_table.to_csv(output_file, index=True)

    print(f"\nSaved ranked results to {output_file}")


if __name__ == "__main__":
    main()
