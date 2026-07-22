import pandas as pd

from quant_backtest.data.cleaner import MarketDataCleaner


def test_cleaner_deduplicates_fills_and_rejects_invalid_ohlc():
    data = pd.DataFrame(
        [
            ["2024-01-02", "AAPL.US", "US", 10, 12, 9, 11, 1, 100],
            ["2024-01-02", "AAPL.US", "US", 10, 12, 9, 11.5, 1, 100],
            ["2024-01-03", "AAPL.US", "US", None, 13, 10, 12, 1, 120],
            ["2024-01-04", "AAPL.US", "US", 12, 11, 10, 12, 1, 120],
        ],
        columns=[
            "date",
            "code",
            "market",
            "open",
            "high",
            "low",
            "close",
            "adj_factor",
            "volume",
        ],
    )

    cleaned = MarketDataCleaner().clean(data)

    assert len(cleaned) == 2
    assert cleaned["date"].is_unique
    assert cleaned.iloc[1]["open"] == 10


def test_mad_outlier_detection():
    data = pd.DataFrame(
        {
            "code": ["AAPL.US"] * 5,
            "close": [10, 10, 11, 9, 100],
        }
    )

    outliers = MarketDataCleaner().find_outliers(data)

    assert outliers["close"].tolist() == [100]
