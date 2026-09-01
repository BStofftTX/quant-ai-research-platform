"""Extract, transform, and load market data."""

import pandas as pd
import yfinance as yf

from quant_ai_research_platform.database import load_market_data, save_market_data


def get_market_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Download historical market data and persist it to PostgreSQL."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    if data.empty:
        raise ValueError(f"No market data returned for {symbol}")

    save_market_data(symbol, data)

    return data


def get_stored_market_data(symbol: str) -> pd.DataFrame:
    """Load historical market data from PostgreSQL in analysis-ready form."""
    data = load_market_data(symbol)

    if data.empty:
        raise ValueError(f"No stored market data found for {symbol}")

    data = data.rename(
        columns={
            "open_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "close_price": "Close",
            "volume": "Volume",
        }
    )

    data["trading_date"] = pd.to_datetime(data["trading_date"])
    data = data.set_index("trading_date")
    data.index.name = "Date"

    return data
