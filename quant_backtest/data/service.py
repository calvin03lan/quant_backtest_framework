from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from .cleaner import MarketDataCleaner
from .providers import MarketDataProvider
from .repository import MongoMarketDataRepository


class MarketDataService:
    def __init__(
        self,
        repository: MongoMarketDataRepository,
        providers: Mapping[str, MarketDataProvider],
        cleaner: MarketDataCleaner | None = None,
    ) -> None:
        self.repository = repository
        self.providers = dict(providers)
        self.cleaner = cleaner or MarketDataCleaner()

    def load(
        self,
        codes_by_market: Mapping[str, Sequence[str]],
        *,
        start: str | pd.Timestamp = "2015-01-01",
        end: str | pd.Timestamp | None = None,
        incremental: bool = True,
    ) -> dict[str, int]:
        end_date = pd.Timestamp(end) if end is not None else last_completed_month_end()
        requested_start = pd.Timestamp(start)
        counts: dict[str, int] = {}

        for market, codes in codes_by_market.items():
            provider = self.providers.get(market)
            if provider is None:
                raise ValueError(f"no provider configured for market: {market}")
            for code in codes:
                fetch_start = requested_start
                if incremental:
                    latest = self.repository.latest_date(code)
                    if latest is not None:
                        fetch_start = max(fetch_start, latest + pd.Timedelta(days=1))
                if fetch_start > end_date:
                    counts[code] = 0
                    continue

                raw = provider.fetch(code, fetch_start, end_date)
                cleaned = self.cleaner.clean(raw)
                counts[code] = self.repository.upsert_prices(cleaned)
        return counts

    def read(
        self,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        codes: Sequence[str] | None = None,
        pool_id: str | None = None,
        as_of: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if codes is None and pool_id is None:
            raise ValueError("codes or pool_id is required")
        return self.repository.read_prices(
            start=start,
            end=end,
            codes=codes,
            pool_id=pool_id,
            as_of=as_of,
        )


def last_completed_month_end(today: str | pd.Timestamp | None = None) -> pd.Timestamp:
    current = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    return current.to_period("M").start_time - pd.Timedelta(days=1)
