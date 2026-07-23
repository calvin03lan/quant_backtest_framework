from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import pandas as pd

from .cleaner import MarketDataCleaner
from .pools import IndexConstituentProvider, validate_snapshot
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
        request_delay: float = 0.0,
    ) -> dict[str, int | str]:
        if request_delay < 0:
            raise ValueError("request_delay must be non-negative")
        end_date = pd.Timestamp(end) if end is not None else last_completed_month_end()
        requested_start = pd.Timestamp(start)
        counts: dict[str, int | str] = {}

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

                try:
                    raw = provider.fetch(code, fetch_start, end_date)
                    cleaned = self.cleaner.clean(raw)
                    counts[code] = self.repository.upsert_prices(cleaned)
                except Exception as exc:
                    counts[code] = f"ERROR: {exc}"
                if request_delay:
                    time.sleep(request_delay)
        return counts

    def sync_pool(self, provider: IndexConstituentProvider) -> dict[str, object]:
        snapshot = provider.fetch_current()
        validate_snapshot(snapshot, provider.expected_size)
        changed = self.repository.sync_pool_snapshot(snapshot)
        return {
            "pool_id": provider.pool_id,
            "snapshot_date": pd.Timestamp(snapshot["snapshot_date"].iloc[0]),
            "members": len(snapshot),
            "changed": changed,
        }

    def resolve_pool_codes(
        self, pool_id: str, as_of: str | pd.Timestamp | None = None
    ) -> dict[str, list[str]]:
        date = (
            pd.Timestamp(as_of)
            if as_of is not None
            else self.repository.latest_pool_snapshot(pool_id)
        )
        if date is None:
            raise ValueError(f"pool not found: {pool_id}")
        members = self.repository.get_pool_members(pool_id, date)
        grouped: dict[str, list[str]] = {}
        for member in members:
            grouped.setdefault(member["market"], []).append(member["code"])
        return grouped

    def load_pool(
        self,
        pool_id: str,
        *,
        start: str | pd.Timestamp = "2015-01-01",
        end: str | pd.Timestamp | None = None,
        as_of: str | pd.Timestamp | None = None,
        incremental: bool = True,
        batch_offset: int = 0,
        batch_size: int | None = None,
        request_delay: float = 0.0,
    ) -> dict[str, int | str]:
        if batch_offset < 0:
            raise ValueError("batch_offset must be non-negative")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        grouped = self.resolve_pool_codes(pool_id, as_of)
        flattened = [
            (market, code)
            for market, codes in sorted(grouped.items())
            for code in sorted(codes)
        ]
        selected = flattened[
            batch_offset : None if batch_size is None else batch_offset + batch_size
        ]
        selected_by_market: dict[str, list[str]] = {}
        for market, code in selected:
            selected_by_market.setdefault(market, []).append(code)
        return self.load(
            selected_by_market,
            start=start,
            end=end,
            incremental=incremental,
            request_delay=request_delay,
        )

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
