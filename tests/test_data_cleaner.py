import pandas as pd

from quant_backtest.data.cleaner import MarketDataCleaner


def test_cleaner_deduplicates_and_rejects_missing_or_invalid_ohlc():
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

    assert len(cleaned) == 1
    assert cleaned["date"].is_unique
    assert cleaned.iloc[0]["close"] == 11.5


def test_cleaner_only_fills_adjustment_factor_and_marks_missing_volume_untradable():
    data = pd.DataFrame(
        [
            ["2024-01-02", "AAPL.US", "US", 10, 12, 9, 11, 1, 100],
            ["2024-01-03", "AAPL.US", "US", 11, 13, 10, 12, None, None],
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

    assert cleaned.iloc[1]["adj_factor"] == 1
    assert cleaned.iloc[1]["volume"] == 0


def test_mad_outlier_detection():
    data = pd.DataFrame(
        {
            "code": ["AAPL.US"] * 5,
            "close": [10, 10, 11, 9, 100],
        }
    )

    outliers = MarketDataCleaner().find_outliers(data)

    assert outliers["close"].tolist() == [100]
