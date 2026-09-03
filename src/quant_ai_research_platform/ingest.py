"""Load market data into PostgreSQL."""

from quant_ai_research_platform.etl import get_market_data


DEFAULT_SYMBOLS = (
    "AAPL,META,AMZN,GOOG,BABA,MSFT,TSLA,NVDA,INTC,"
    "PWR,TSM,AMD,CRWD,PANW,ADBE,NNE,SPCX,ELVR,SPY"
)


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

    for symbol in symbols:
        print(f"\nUpdating {symbol}...")
        try:
            get_market_data(symbol)
        except ValueError as error:
            print(f"Unable to update {symbol}: {error}")

    print("\nMarket data update complete.")


if __name__ == "__main__":
    main()
