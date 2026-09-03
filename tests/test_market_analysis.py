from datetime import date

import pandas as pd

from quant_ai_research_platform.market_analysis import (
    calculate_benchmark_metrics,
    calculate_statistics,
)


def test_calculate_statistics():
    data = pd.DataFrame(
        {"Close": [100.0, 110.0, 121.0]}
    )

    stats = calculate_statistics(data)

    assert stats["starting_price"] == 100.0
    assert stats["ending_price"] == 121.0
    assert round(stats["total_return"], 2) == 0.21


def test_benchmark_metrics_align_dates():
    stock = pd.DataFrame(
        {"Close": [100.0, 108.0, 121.0, 127.0]},
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
        ),
    )

    benchmark = pd.DataFrame(
        {"Close": [200.0, 210.0, 218.0]},
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06"]
        ),
    )

    result = calculate_benchmark_metrics(stock, benchmark)

    assert result["observations"] == 3
    assert result["start_date"] == date(2026, 1, 2)
    assert result["end_date"] == date(2026, 1, 6)
