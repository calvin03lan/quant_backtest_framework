from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = "mongodb://localhost:27017"
    database_name: str = "quant_db"
    daily_collection: str = "daily_ohlc"
    pool_collection: str = "stock_pool"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            mongo_uri=os.getenv("MONGO_URI", cls.mongo_uri),
            database_name=os.getenv("MONGO_DB", cls.database_name),
            daily_collection=os.getenv(
                "MONGO_DAILY_COLLECTION", cls.daily_collection
            ),
            pool_collection=os.getenv(
                "MONGO_POOL_COLLECTION", cls.pool_collection
            ),
        )
