import pandas as pd
import pytest

from quant_backtest.engine import BacktestEngine
from quant_backtest.factors import (
    CompositeFactor,
    MomentumFactor,
    ReversalFactor,
    VolatilityFactor,
)


def make_prices(periods=45):
    dates = pd.bdate_range("2024-01-01", periods=periods)
    records = []
    for index, date in enumerate(dates):
        records.extend(
            [
                {
                    "date": date,
                    "code": "AAA.US",
                    "close": 100 + index * 2,
                    "adj_factor": 1.0,
                },
                {
                    "date": date,
                    "code": "BBB.US",
                    "close": 100 + index,
                    "adj_factor": 1.0,
                },
            ]
        )
    return pd.DataFrame(records)


def test_momentum_uses_only_prior_rows():
    prices = make_prices(6)
    factor = MomentumFactor(lookback=2).compute(prices)
    first = factor[factor["code"] == "AAA.US"].iloc[0]

    assert first["date"] == pd.Timestamp("2024-01-04")
    assert first["factor_value"] == pytest.approx(0.04)


def test_engine_rebalances_monthly_and_applies_position_next_day():
    prices = make_prices()
    factors = MomentumFactor(5).compute(prices)

    result = BacktestEngine(top_n=1).run(prices, factors)

    assert set(result.rebalances["code"]) == {"AAA.US"}
    assert result.daily.iloc[0]["net_value"] == 1.0
    first_rebalance = result.rebalances.iloc[0]["date"]
    weight_on_rebalance = result.positions.query(
        "date == @first_rebalance and code == 'AAA.US'"
    )["weight"].iloc[0]
    assert weight_on_rebalance == 0.0
    assert result.daily.iloc[-1]["net_value"] > 1.0


def test_composite_factor_runs_with_engine():
    prices = make_prices()
    factors = CompositeFactor(
        [
            (MomentumFactor(5), 1),
            (ReversalFactor(3), 1),
            (VolatilityFactor(5), 1),
        ]
    ).compute(prices)

    result = BacktestEngine(top_n=1).run(prices, factors)

    assert not result.rebalances.empty
    assert result.daily["net_value"].notna().all()


def test_engine_rejects_empty_factors():
    empty = pd.DataFrame(columns=["date", "code", "factor_value"])

    with pytest.raises(ValueError, match="factors must not be empty"):
        BacktestEngine().run(make_prices(), empty)
