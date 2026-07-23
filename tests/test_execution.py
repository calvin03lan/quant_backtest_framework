import pandas as pd
import pytest

from quant_backtest.engine import BacktestEngine
from quant_backtest.execution import (
    ExecutionConfig,
    MarketFeeSchedule,
    execute_order,
)


def test_market_fees_and_sell_tax_are_applied_by_side():
    config = ExecutionConfig(
        cn_fees=MarketFeeSchedule(commission_rate=0.0003, sell_tax_rate=0.0005),
        slippage_bps=5,
    )

    buy = execute_order(
        desired_notional=500,
        holding_notional=0,
        available_cash=1_000,
        reference_price=10,
        volume=10_000,
        market="CN",
        config=config,
    )
    sell = execute_order(
        desired_notional=-500,
        holding_notional=500,
        available_cash=0,
        reference_price=10,
        volume=10_000,
        market="CN",
        config=config,
    )

    assert buy.commission == pytest.approx(0.15)
    assert buy.tax == 0
    assert buy.slippage == pytest.approx(0.25)
    assert sell.commission == pytest.approx(0.15)
    assert sell.tax == pytest.approx(0.25)
    assert sell.execution_price == pytest.approx(9.995)


def test_us_schedule_has_no_sell_tax():
    config = ExecutionConfig(
        us_fees=MarketFeeSchedule(commission_rate=0.0001)
    )

    fill = execute_order(
        desired_notional=-500,
        holding_notional=500,
        available_cash=0,
        reference_price=10,
        volume=10_000,
        market="US",
        config=config,
    )

    assert fill.commission == pytest.approx(0.05)
    assert fill.tax == 0


def test_volume_participation_limits_fill():
    config = ExecutionConfig(max_volume_participation=0.1)

    fill = execute_order(
        desired_notional=500,
        holding_notional=0,
        available_cash=1_000,
        reference_price=10,
        volume=20,
        market="US",
        config=config,
    )

    assert fill.executed_notional == pytest.approx(20)
    assert fill.fill_ratio == pytest.approx(0.04)
    assert fill.status == "VOLUME_LIMITED"


@pytest.mark.parametrize(
    ("volume", "status"),
    [(0, "SUSPENDED"), (None, "NO_VOLUME")],
)
def test_untradable_orders_are_rejected(volume, status):
    fill = execute_order(
        desired_notional=500,
        holding_notional=0,
        available_cash=1_000,
        reference_price=10,
        volume=volume,
        market="US",
        config=ExecutionConfig(max_volume_participation=0.1),
    )

    assert fill.executed_notional == 0
    assert fill.status == status


def execution_prices(volume=10_000):
    return pd.DataFrame(
        [
            {
                "date": date,
                "code": "AAA.US",
                "market": "US",
                "close": 10,
                "adj_factor": 1,
                "volume": volume,
            }
            for date in pd.bdate_range("2024-01-01", periods=3)
        ]
    )


def execution_factors():
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "code": "AAA.US",
                "factor_value": 1.0,
            }
        ]
    )


def test_engine_execution_cost_reduces_nav_and_records_trade():
    prices = execution_prices()
    factors = execution_factors()
    frictionless = BacktestEngine(top_n=1).run(prices, factors)
    constrained = BacktestEngine(
        top_n=1,
        execution_config=ExecutionConfig(
            initial_cash=1_000,
            us_fees=MarketFeeSchedule(commission_rate=0.001),
            slippage_bps=10,
            max_volume_participation=0.1,
        ),
    ).run(prices, factors)

    assert constrained.daily.iloc[-1]["net_value"] < frictionless.daily.iloc[-1][
        "net_value"
    ]
    assert constrained.daily["transaction_cost"].sum() > 0
    assert constrained.trades.iloc[0]["side"] == "BUY"
    assert constrained.trades.iloc[0]["fill_ratio"] > 0


def test_engine_keeps_cash_when_target_is_suspended():
    result = BacktestEngine(
        top_n=1,
        execution_config=ExecutionConfig(
            initial_cash=1_000, max_volume_participation=0.1
        ),
    ).run(execution_prices(volume=0), execution_factors())

    assert result.trades.iloc[0]["status"] == "SUSPENDED"
    assert result.daily.iloc[-1]["cash"] == 1_000
    assert result.positions["weight"].max() == 0
