import pandas as pd
import psycopg


DATABASE_NAME = "quant_ai_research"


def get_connection():
    return psycopg.connect(f"dbname={DATABASE_NAME}")


def test_connection():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user;")
            database_name, current_user = cur.fetchone()

    print(f"Connected to database: {database_name}")
    print(f"Connected as user: {current_user}")


if __name__ == "__main__":
    test_connection()


def save_market_data(symbol, data):
    insert_sql = """
        INSERT INTO market_prices (
            symbol,
            trading_date,
            open_price,
            high_price,
            low_price,
            close_price,
            volume
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, trading_date)
        DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume;
    """

    clean_data = data.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )

    rows = []

    for trading_date, row in clean_data.iterrows():
        rows.append(
            (
                symbol,
                trading_date.date(),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"]),
            )
        )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, rows)

    print(f"Saved {len(rows)} rows for {symbol} to PostgreSQL.")


def load_market_data(symbol):
    query = """
        SELECT
            trading_date,
            open_price,
            high_price,
            low_price,
            close_price,
            volume
        FROM market_prices
        WHERE symbol = %s
        ORDER BY trading_date;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    data = pd.DataFrame(rows, columns=columns)

    price_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    data[price_columns] = data[price_columns].astype(float)

    return data
