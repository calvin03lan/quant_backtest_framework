from __future__ import annotations

import numpy as np
import pandas as pd

from .providers import STANDARD_COLUMNS


class MarketDataCleaner:
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_factor",
        "volume",
    ]

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = set(STANDARD_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        if data.empty:
            return data[STANDARD_COLUMNS].copy()

        cleaned = data[STANDARD_COLUMNS].copy()
        cleaned["date"] = pd.to_datetime(cleaned["date"]).dt.normalize()
        for column in self.numeric_columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

        cleaned = cleaned.sort_values(["code", "date"])
        cleaned = cleaned.drop_duplicates(["code", "date"], keep="last")
        cleaned["adj_factor"] = cleaned.groupby("code")["adj_factor"].ffill(limit=5)
        # Missing volume means the bar cannot be traded; do not invent liquidity.
        cleaned["volume"] = cleaned["volume"].fillna(0)
        cleaned = cleaned.dropna(
            subset=["date", "code", "open", "high", "low", "close", "adj_factor"]
        )

        finite = np.isfinite(cleaned[self.numeric_columns]).all(axis=1)
        valid = (
            finite
            & (cleaned["open"] > 0)
            & (cleaned["high"] > 0)
            & (cleaned["low"] > 0)
            & (cleaned["close"] > 0)
            & (cleaned["adj_factor"] > 0)
            & (cleaned["volume"] >= 0)
            & (cleaned["high"] >= cleaned[["open", "close", "low"]].max(axis=1))
            & (cleaned["low"] <= cleaned[["open", "close", "high"]].min(axis=1))
        )
        return cleaned.loc[valid, STANDARD_COLUMNS].reset_index(drop=True)

    def find_outliers(
        self, data: pd.DataFrame, column: str = "close", threshold: float = 3.0
    ) -> pd.DataFrame:
        if column not in data.columns:
            raise ValueError(f"unknown column: {column}")
        median = data.groupby("code")[column].transform("median")
        deviation = (data[column] - median).abs()
        mad = deviation.groupby(data["code"]).transform("median")
        is_outlier = (mad > 0) & (deviation > threshold * mad)
        return data.loc[is_outlier].copy()
