import akshare as ak
import pandas as pd

from quant_backtest.data.providers import AkShareProvider


def test_akshare_volume_is_converted_from_lots_to_shares(monkeypatch):
    def fake_history(**kwargs):
        close = 9 if kwargs["adjust"] == "qfq" else 10
        return pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "开盘": [10],
                "最高": [11],
                "最低": [9],
                "收盘": [close],
                "成交量": [100],
            }
        )

    monkeypatch.setattr(ak, "stock_zh_a_hist", fake_history)

    result = AkShareProvider().fetch(
        "000001.SZ", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")
    )

    assert result.iloc[0]["volume"] == 10_000
    assert result.iloc[0]["adj_factor"] == 0.9
