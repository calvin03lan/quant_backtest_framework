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

    def latest_date(self, code):
        return self.latest

    def upsert_prices(self, data):
        self.saved.append(data)
        return len(data)


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
