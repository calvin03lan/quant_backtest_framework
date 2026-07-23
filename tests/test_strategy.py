import pandas as pd
import pytest

from quant_backtest.strategy import (
    MovingAverageStrategy,
    SingleAssetBacktestEngine,
)


def make_prices(values, code="AAPL.US"):
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=len(values)),
            "code": [code] * len(values),
            "close": values,
            "adj_factor": [1.0] * len(values),
        }
    )


def make_signals(prices, values):
    return pd.DataFrame(
        {
            "date": prices["date"],
            "code": prices["code"],
            "short_ma": prices["close"],
            "long_ma": prices["close"],
            "signal": values,
        }
    )


def test_moving_average_strategy_validates_windows():
    with pytest.raises(ValueError, match="short_window"):
        MovingAverageStrategy(short_window=0, long_window=3)
    with pytest.raises(ValueError, match="long_window"):
        MovingAverageStrategy(short_window=3, long_window=3)


def test_moving_average_strategy_generates_expected_signal():
    prices = make_prices([1, 2, 3, 4, 3, 2])

    signals = MovingAverageStrategy(
        short_window=2, long_window=3
    ).generate_signals(prices)

    assert signals["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    assert signals.loc[2, "short_ma"] == pytest.approx(2.5)
    assert signals.loc[2, "long_ma"] == pytest.approx(2.0)


def test_future_price_change_does_not_change_previous_signals():
    prices = make_prices([10, 11, 12, 13, 14, 15])
    changed = prices.copy()
    changed.loc[changed.index[-1], "close"] = 1_000
    strategy = MovingAverageStrategy(short_window=2, long_window=3)

    original = strategy.generate_signals(prices)
    revised = strategy.generate_signals(changed)

    pd.testing.assert_frame_equal(
        original.iloc[:-1].reset_index(drop=True),
        revised.iloc[:-1].reset_index(drop=True),
    )


def test_backtest_uses_next_day_position_and_charges_turnover_cost():
    prices = make_prices([100, 110, 121])
    signals = make_signals(prices, [0, 1, 1])

    result = SingleAssetBacktestEngine(transaction_cost_bps=100).run(
        prices, signals
    )

    assert result.daily["position"].tolist() == [0, 0, 1]
    assert result.daily["turnover"].tolist() == [0, 0, 1]
    assert result.daily.loc[2, "gross_return"] == pytest.approx(0.10)
    assert result.daily.loc[2, "transaction_cost"] == pytest.approx(0.01)
    assert result.daily.loc[2, "return"] == pytest.approx(0.09)
    assert result.daily.loc[2, "net_value"] == pytest.approx(1.09)


def test_backtest_rejects_multiple_or_duplicate_symbols():
    multiple = pd.concat(
        [make_prices([10, 11]), make_prices([20, 21], "MSFT.US")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="exactly one code"):
        MovingAverageStrategy(1, 2).generate_signals(multiple)

    duplicate = pd.concat(
        [make_prices([10, 11]), make_prices([10, 11]).iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate"):
        MovingAverageStrategy(1, 2).generate_signals(duplicate)
