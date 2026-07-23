from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.database import Database

from quant_backtest.config import Settings

from .providers import STANDARD_COLUMNS


class MongoMarketDataRepository:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        database: Database | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._client = None
        if database is None:
            self._client = MongoClient(
                self.settings.mongo_uri, serverSelectionTimeoutMS=3_000
            )
            database = self._client[self.settings.database_name]
        self.db = database
        self.daily = self.db[self.settings.daily_collection]
        self.pools = self.db[self.settings.pool_collection]

    def create_indexes(self) -> None:
        self.daily.create_index(
            [("code", ASCENDING), ("date", ASCENDING)],
            unique=True,
            name="uq_code_date",
        )
        self.pools.create_index(
            [
                ("pool_id", ASCENDING),
                ("code", ASCENDING),
                ("valid_from", ASCENDING),
            ],
            unique=True,
            name="uq_pool_code_valid_from",
        )

    def ping(self) -> None:
        self.db.command("ping")

    def upsert_prices(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        operations = []
        records = []
        for record in data[STANDARD_COLUMNS].to_dict("records"):
            record["date"] = _to_datetime(record["date"])
            records.append(record)
            operations.append(
                UpdateOne(
                    {"code": record["code"], "date": record["date"]},
                    {"$set": record},
                    upsert=True,
                )
            )
        if _is_mongomock(self.daily):
            return sum(
                _changed(
                    self.daily.update_one(
                        {"code": record["code"], "date": record["date"]},
                        {"$set": record},
                        upsert=True,
                    )
                )
                for record in records
            )
        result = self.daily.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count

    def upsert_pool(
        self,
        pool_id: str,
        codes: Sequence[str],
        valid_from: str | pd.Timestamp,
        valid_to: str | pd.Timestamp | None = None,
    ) -> int:
        start = _to_datetime(valid_from)
        end = _to_datetime(valid_to) if valid_to is not None else None
        operations = []
        documents = []
        for code in codes:
            market = "CN" if code.endswith((".SH", ".SZ", ".BJ")) else "US"
            document = {
                "pool_id": pool_id,
                "code": code,
                "market": market,
                "valid_from": start,
                "valid_to": end,
            }
            documents.append(document)
            operations.append(
                UpdateOne(
                    {"pool_id": pool_id, "code": code, "valid_from": start},
                    {"$set": document},
                    upsert=True,
                )
            )
        if not operations:
            return 0
        if _is_mongomock(self.pools):
            return sum(
                _changed(
                    self.pools.update_one(
                        {
                            "pool_id": document["pool_id"],
                            "code": document["code"],
                            "valid_from": document["valid_from"],
                        },
                        {"$set": document},
                        upsert=True,
                    )
                )
                for document in documents
            )
        result = self.pools.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count

    def sync_pool_snapshot(self, data: pd.DataFrame) -> int:
        required = {
            "pool_id",
            "code",
            "market",
            "name",
            "source",
            "snapshot_date",
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"pool snapshot missing columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("pool snapshot must not be empty")
        if data["code"].duplicated().any():
            raise ValueError("pool snapshot contains duplicate codes")
        pool_ids = data["pool_id"].unique()
        snapshot_dates = pd.to_datetime(data["snapshot_date"]).dt.normalize().unique()
        if len(pool_ids) != 1 or len(snapshot_dates) != 1:
            raise ValueError("a pool sync must contain one pool and one snapshot date")

        pool_id = str(pool_ids[0])
        snapshot = _to_datetime(pd.Timestamp(snapshot_dates[0]))
        previous_day = snapshot - pd.Timedelta(days=1)
        codes = data["code"].tolist()
        self.pools.update_many(
            {
                "pool_id": pool_id,
                "valid_to": None,
                "valid_from": {"$lt": snapshot},
            },
            {"$set": {"valid_to": previous_day}},
        )
        self.pools.delete_many(
            {
                "pool_id": pool_id,
                "valid_from": snapshot,
                "code": {"$nin": codes},
            }
        )

        documents = []
        operations = []
        for row in data.to_dict("records"):
            document = {
                "pool_id": pool_id,
                "code": row["code"],
                "market": row["market"],
                "name": row["name"],
                "source": row["source"],
                "snapshot_date": snapshot,
                "valid_from": snapshot,
                "valid_to": None,
            }
            documents.append(document)
            operations.append(
                UpdateOne(
                    {
                        "pool_id": pool_id,
                        "code": row["code"],
                        "valid_from": snapshot,
                    },
                    {"$set": document},
                    upsert=True,
                )
            )
        if _is_mongomock(self.pools):
            return sum(
                _changed(
                    self.pools.update_one(
                        {
                            "pool_id": document["pool_id"],
                            "code": document["code"],
                            "valid_from": document["valid_from"],
                        },
                        {"$set": document},
                        upsert=True,
                    )
                )
                for document in documents
            )
        result = self.pools.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count

    def latest_pool_snapshot(self, pool_id: str) -> pd.Timestamp | None:
        document = self.pools.find_one(
            {"pool_id": pool_id},
            sort=[("valid_from", -1)],
            projection={"valid_from": 1},
        )
        return pd.Timestamp(document["valid_from"]) if document else None

    def get_pool_members(
        self, pool_id: str, as_of: str | pd.Timestamp
    ) -> list[dict]:
        date = _to_datetime(as_of)
        query = {
            "pool_id": pool_id,
            "valid_from": {"$lte": date},
            "$or": [{"valid_to": None}, {"valid_to": {"$gte": date}}],
        }
        return list(
            self.pools.find(query, {"_id": 0}).sort([("market", 1), ("code", 1)])
        )

    def latest_date(self, code: str) -> pd.Timestamp | None:
        document = self.daily.find_one(
            {"code": code}, sort=[("date", -1)], projection={"date": 1}
        )
        return pd.Timestamp(document["date"]) if document else None

    def list_price_codes(self) -> list[str]:
        return sorted(self.daily.distinct("code"))

    def get_pool_codes(
        self, pool_id: str, as_of: str | pd.Timestamp
    ) -> list[str]:
        return [member["code"] for member in self.get_pool_members(pool_id, as_of)]

    def read_prices(
        self,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        codes: Iterable[str] | None = None,
        pool_id: str | None = None,
        as_of: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        selected = list(codes) if codes is not None else None
        if pool_id is not None:
            selected = self.get_pool_codes(pool_id, as_of or end)
        if selected == []:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        query: dict = {
            "date": {"$gte": _to_datetime(start), "$lte": _to_datetime(end)}
        }
        if selected is not None:
            query["code"] = {"$in": selected}
        documents = list(
            self.daily.find(query, {"_id": 0}).sort([("date", 1), ("code", 1)])
        )
        if not documents:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        return pd.DataFrame(documents)[STANDARD_COLUMNS]


def _to_datetime(value: str | pd.Timestamp | datetime) -> datetime:
    return pd.Timestamp(value).normalize().to_pydatetime()


def _is_mongomock(collection) -> bool:
    return collection.__class__.__module__.startswith("mongomock.")


def _changed(result) -> int:
    return int(result.upserted_id is not None or result.modified_count > 0)
