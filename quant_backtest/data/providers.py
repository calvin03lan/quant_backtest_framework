from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "date",
    "code",
    "market",
    "open",
    "high",
    "low",
    "close",
    "adj_factor",
    "volume",
]


class MarketDataProvider(Protocol):
    market: str

    def fetch(self, code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch raw OHLC data in the standard schema."""


class AkShareProvider:
    market = "CN"

    def fetch(self, code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        import akshare as ak

        symbol = code.split(".")[0]
        arguments = {
            "symbol": symbol,
            "period": "daily",
            "start_date": pd.Timestamp(start).strftime("%Y%m%d"),
            "end_date": pd.Timestamp(end).strftime("%Y%m%d"),
        }
        raw = ak.stock_zh_a_hist(**arguments, adjust="")
        adjusted = ak.stock_zh_a_hist(**arguments, adjust="qfq")
        if raw.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        raw = self._rename(raw)
        adjusted = self._rename(adjusted)
        factors = adjusted[["date", "close"]].rename(
            columns={"close": "adjusted_close"}
        )
        data = raw.merge(factors, on="date", how="left")
        data["adj_factor"] = data["adjusted_close"] / data["close"]
        data["adj_factor"] = data["adj_factor"].replace(
            [np.inf, -np.inf], np.nan
        )
        data["code"] = normalize_cn_code(code)
        data["market"] = self.market
        return data[STANDARD_COLUMNS]

    @staticmethod
    def _rename(data: pd.DataFrame) -> pd.DataFrame:
        renamed = data.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )
        renamed["date"] = pd.to_datetime(renamed["date"])
        return renamed


class YFinanceProvider:
    market = "US"

    def fetch(self, code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        import yfinance as yf

        symbol = code.removesuffix(".US")
        # yfinance treats end as exclusive.
        data = yf.download(
            symbol,
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            actions=False,
            progress=False,
        )
        if data.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )
        data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
        adjusted_close = data.get("adjusted_close", data["close"])
        data["adj_factor"] = adjusted_close / data["close"]
        data["code"] = normalize_us_code(code)
        data["market"] = self.market
        return data[STANDARD_COLUMNS]


def normalize_cn_code(code: str) -> str:
    symbol = code.split(".")[0].zfill(6)
    if "." in code:
        suffix = code.rsplit(".", 1)[1].upper()
    else:
        suffix = "SH" if symbol.startswith(("5", "6", "9")) else "SZ"
    return f"{symbol}.{suffix}"


def normalize_us_code(code: str) -> str:
    symbol = code.upper().removesuffix(".US")
    return f"{symbol}.US"
