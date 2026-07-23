import pandas as pd
import pytest

from quant_backtest.data.pools import (
    Csi300ConstituentProvider,
    Sp500ConstituentProvider,
    validate_snapshot,
)
from quant_backtest.data.providers import to_yfinance_symbol


def test_csi300_provider_normalizes_codes_and_snapshot_date():
    raw = pd.DataFrame(
        {
            "成分券代码": ["000001", "600519"],
            "成分券名称": ["平安银行", "贵州茅台"],
            "日期": ["2026-07-22", "2026-07-22"],
        }
    )
    provider = Csi300ConstituentProvider(fetcher=lambda **_: raw)

    result = provider.fetch_current()

    assert result["code"].tolist() == ["000001.SZ", "600519.SH"]
    assert result["snapshot_date"].unique().tolist() == [
        pd.Timestamp("2026-07-22")
    ]


def test_sp500_provider_preserves_canonical_ticker_and_maps_yfinance_symbol():
    raw = pd.DataFrame(
        {
            "Symbol": ["BRK.B", "AAPL"],
            "Security": ["Berkshire Hathaway", "Apple"],
        }
    )
    provider = Sp500ConstituentProvider(
        fetcher=lambda: raw, snapshot_date="2026-07-22"
    )

    result = provider.fetch_current()

    assert result["code"].tolist() == ["AAPL.US", "BRK.B.US"]
    assert to_yfinance_symbol("BRK.B.US") == "BRK-B"


def test_snapshot_validation_checks_expected_size():
    raw = pd.DataFrame({"Symbol": ["AAPL"], "Security": ["Apple"]})
    snapshot = Sp500ConstituentProvider(
        fetcher=lambda: raw, snapshot_date="2026-07-22"
    ).fetch_current()

    with pytest.raises(ValueError, match="outside expected range"):
        validate_snapshot(snapshot, (490, 510))
