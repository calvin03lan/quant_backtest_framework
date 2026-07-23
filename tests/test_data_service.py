import pandas as pd

from quant_backtest.data.providers import normalize_cn_code, normalize_us_code
from quant_backtest.data.service import MarketDataService, last_completed_month_end


class FakeProvider:
    market = "US"

    def __init__(self):
        self.calls = []

    def fetch(self, code, start, end):
        self.calls.append((code, pd.Timestamp(start), pd.Timestamp(end)))
        return pd.DataFrame(
            [
                {
                    "date": start,
                    "code": code,
                    "market": "US",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10,
                    "adj_factor": 1,
                    "volume": 100,
                }
            ]
        )


class FakeRepository:
    def __init__(self, latest=None):
        self.latest = latest
        self.saved = []
        self.pool_data = None

    def latest_date(self, code):
        return self.latest

    def upsert_prices(self, data):
        self.saved.append(data)
        return len(data)

    def sync_pool_snapshot(self, data):
        self.pool_data = data
        return len(data)

    def latest_pool_snapshot(self, pool_id):
        if self.pool_data is None:
            return None
        return self.pool_data["snapshot_date"].iloc[0]

    def get_pool_members(self, pool_id, as_of):
        if self.pool_data is None:
            return []
        return self.pool_data[["code", "market"]].to_dict("records")


class StaticPoolProvider:
    pool_id = "test_pool"
    market = "US"
    expected_size = (3, 3)

    def fetch_current(self):
        return pd.DataFrame(
            {
                "pool_id": [self.pool_id] * 3,
                "code": ["AAA.US", "BBB.US", "CCC.US"],
                "market": [self.market] * 3,
                "name": ["AAA", "BBB", "CCC"],
                "source": ["fixture"] * 3,
                "snapshot_date": [pd.Timestamp("2026-07-22")] * 3,
            }
        )


def test_code_normalization():
    assert normalize_cn_code("1") == "000001.SZ"
    assert normalize_cn_code("600519") == "600519.SH"
    assert normalize_us_code("aapl") == "AAPL.US"


def test_last_completed_month_end():
    assert last_completed_month_end("2026-07-22") == pd.Timestamp("2026-06-30")


def test_incremental_load_starts_after_latest_date():
    provider = FakeProvider()
    repository = FakeRepository(pd.Timestamp("2024-01-05"))
    service = MarketDataService(repository, {"US": provider})

    counts = service.load(
        {"US": ["AAPL.US"]}, start="2024-01-01", end="2024-01-10"
    )

    assert counts == {"AAPL.US": 1}
    assert provider.calls[0][1] == pd.Timestamp("2024-01-06")


def test_sync_and_batch_load_pool():
    provider = FakeProvider()
    repository = FakeRepository()
    service = MarketDataService(repository, {"US": provider})

    report = service.sync_pool(StaticPoolProvider())
    counts = service.load_pool(
        "test_pool",
        start="2024-01-01",
        end="2024-01-10",
        batch_offset=1,
        batch_size=1,
    )

    assert report["members"] == 3
    assert counts == {"BBB.US": 1}
    assert provider.calls[0][0] == "BBB.US"


def test_load_isolates_provider_failures():
    class FailingProvider(FakeProvider):
        def fetch(self, code, start, end):
            if code == "BAD.US":
                raise RuntimeError("network failure")
            return super().fetch(code, start, end)

    service = MarketDataService(FakeRepository(), {"US": FailingProvider()})

    counts = service.load(
        {"US": ["BAD.US", "GOOD.US"]},
        start="2024-01-01",
        end="2024-01-10",
    )

    assert counts["BAD.US"].startswith("ERROR:")
    assert counts["GOOD.US"] == 1
