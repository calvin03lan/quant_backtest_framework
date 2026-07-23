from __future__ import annotations

from io import BytesIO, StringIO
from typing import Callable, Protocol

import pandas as pd

from .providers import normalize_cn_code, normalize_us_code


POOL_COLUMNS = [
    "pool_id",
    "code",
    "market",
    "name",
    "source",
    "snapshot_date",
]


class IndexConstituentProvider(Protocol):
    pool_id: str
    market: str
    expected_size: tuple[int, int]

    def fetch_current(self) -> pd.DataFrame:
        """Return the current constituent snapshot."""


class Csi300ConstituentProvider:
    pool_id = "csi300"
    market = "CN"
    expected_size = (290, 310)

    def __init__(self, fetcher: Callable[..., pd.DataFrame] | None = None) -> None:
        self._fetcher = fetcher

    def fetch_current(self) -> pd.DataFrame:
        if self._fetcher is None:
            raw = self._fetch_csindex()
        else:
            raw = self._fetcher(symbol="000300")
        if raw.empty:
            raise ValueError("CSI 300 constituent source returned no rows")

        code_column = _find_column(raw, ["成分券代码", "品种代码", "代码"])
        name_column = _find_column(
            raw, ["成分券名称", "品种名称", "名称"], required=False
        )
        date_column = _find_column(
            raw, ["日期", "更新时间", "纳入日期"], required=False
        )
        snapshot_date = (
            _parse_dates(raw[date_column]).dropna().max()
            if date_column
            else pd.Timestamp.today().normalize()
        )
        if pd.isna(snapshot_date):
            snapshot_date = pd.Timestamp.today().normalize()

        result = pd.DataFrame(
            {
                "pool_id": self.pool_id,
                "code": raw[code_column].astype(str).map(normalize_cn_code),
                "market": self.market,
                "name": (
                    raw[name_column].astype(str)
                    if name_column
                    else pd.Series("", index=raw.index)
                ),
                "source": "akshare:csindex",
                "snapshot_date": pd.Timestamp(snapshot_date).normalize(),
            }
        )
        return _finalize_snapshot(result)

    @staticmethod
    def _fetch_csindex() -> pd.DataFrame:
        import requests

        url = (
            "https://oss-ch.csindex.com.cn/static/html/csindex/public/"
            "uploads/file/autofile/cons/000300cons.xls"
        )
        response = requests.get(
            url,
            headers={"User-Agent": "quant-backtest-framework/0.1"},
            timeout=30,
        )
        response.raise_for_status()
        data = pd.read_excel(BytesIO(response.content))
        data.columns = [
            "日期",
            "指数代码",
            "指数名称",
            "指数英文名称",
            "成分券代码",
            "成分券名称",
            "成分券英文名称",
            "交易所",
            "交易所英文名称",
        ]
        data["成分券代码"] = data["成分券代码"].astype(str).str.zfill(6)
        return data


class Sp500ConstituentProvider:
    pool_id = "sp500"
    market = "US"
    expected_size = (490, 510)
    source_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(
        self,
        fetcher: Callable[[], pd.DataFrame] | None = None,
        *,
        snapshot_date: str | pd.Timestamp | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._snapshot_date = snapshot_date

    def fetch_current(self) -> pd.DataFrame:
        raw = self._fetcher() if self._fetcher is not None else self._fetch_wikipedia()
        if raw.empty:
            raise ValueError("S&P 500 constituent source returned no rows")

        symbol_column = _find_column(raw, ["Symbol", "Ticker", "代码"])
        name_column = _find_column(
            raw, ["Security", "Company", "名称"], required=False
        )
        snapshot_date = pd.Timestamp(
            self._snapshot_date or pd.Timestamp.today()
        ).normalize()
        result = pd.DataFrame(
            {
                "pool_id": self.pool_id,
                "code": raw[symbol_column].astype(str).map(normalize_us_code),
                "market": self.market,
                "name": (
                    raw[name_column].astype(str)
                    if name_column
                    else pd.Series("", index=raw.index)
                ),
                "source": "wikipedia",
                "snapshot_date": snapshot_date,
            }
        )
        return _finalize_snapshot(result)

    def _fetch_wikipedia(self) -> pd.DataFrame:
        import requests

        response = requests.get(
            self.source_url,
            headers={"User-Agent": "quant-backtest-framework/0.1"},
            timeout=30,
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        if not tables:
            raise ValueError("S&P 500 Wikipedia page contained no tables")
        return tables[0]


def validate_snapshot(
    data: pd.DataFrame, expected_size: tuple[int, int] | None = None
) -> None:
    missing = set(POOL_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"pool snapshot missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("pool snapshot must not be empty")
    if data["code"].duplicated().any():
        raise ValueError("pool snapshot contains duplicate codes")
    if expected_size is not None and not expected_size[0] <= len(data) <= expected_size[1]:
        raise ValueError(
            f"pool snapshot size {len(data)} outside expected range "
            f"{expected_size[0]}-{expected_size[1]}"
        )


def _find_column(
    data: pd.DataFrame, candidates: list[str], *, required: bool = True
) -> str | None:
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    if required:
        raise ValueError(f"none of the required columns found: {candidates}")
    return None


def _finalize_snapshot(data: pd.DataFrame) -> pd.DataFrame:
    result = data[POOL_COLUMNS].drop_duplicates("code", keep="last")
    result = result.sort_values("code").reset_index(drop=True)
    validate_snapshot(result)
    return result


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    return compact.fillna(fallback)
