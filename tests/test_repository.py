import mongomock
import pandas as pd

from quant_backtest.config import Settings
from quant_backtest.data.repository import MongoMarketDataRepository


def make_repository():
    database = mongomock.MongoClient()["test_quant"]
    repository = MongoMarketDataRepository(Settings(), database=database)
    repository.create_indexes()
    return repository


def sample_prices(close=10):
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "code": "AAPL.US",
                "market": "US",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": close,
                "adj_factor": 1,
                "volume": 100,
            }
        ]
    )


def test_price_upsert_is_idempotent():
    repository = make_repository()

    repository.upsert_prices(sample_prices())
    repository.upsert_prices(sample_prices(close=10.5))

    assert repository.daily.count_documents({}) == 1
    assert repository.daily.find_one()["close"] == 10.5


def test_pool_and_date_filtered_read():
    repository = make_repository()
    repository.upsert_prices(sample_prices())
    repository.upsert_pool("sample", ["AAPL.US"], "2020-01-01")

    result = repository.read_prices(
        pool_id="sample",
        as_of="2024-01-02",
        start="2024-01-01",
        end="2024-01-31",
    )

    assert result["code"].tolist() == ["AAPL.US"]
    assert result["date"].tolist() == [pd.Timestamp("2024-01-02")]


def test_empty_pool_returns_standard_empty_frame():
    repository = make_repository()

    result = repository.read_prices(
        pool_id="missing",
        start="2024-01-01",
        end="2024-01-31",
    )

    assert result.empty
    assert "adj_factor" in result.columns


def pool_snapshot(date, codes):
    return pd.DataFrame(
        {
            "pool_id": ["test_index"] * len(codes),
            "code": codes,
            "market": ["US"] * len(codes),
            "name": codes,
            "source": ["fixture"] * len(codes),
            "snapshot_date": [pd.Timestamp(date)] * len(codes),
        }
    )


def test_pool_snapshot_closes_previous_membership_intervals():
    repository = make_repository()
    repository.sync_pool_snapshot(
        pool_snapshot("2026-07-22", ["AAA.US", "BBB.US"])
    )
    repository.sync_pool_snapshot(
        pool_snapshot("2026-07-23", ["BBB.US", "CCC.US"])
    )

    assert repository.get_pool_codes("test_index", "2026-07-22") == [
        "AAA.US",
        "BBB.US",
    ]
    assert repository.get_pool_codes("test_index", "2026-07-23") == [
        "BBB.US",
        "CCC.US",
    ]
    assert repository.latest_pool_snapshot("test_index") == pd.Timestamp(
        "2026-07-23"
    )
