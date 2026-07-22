import numpy as np
import pandas as pd
import pytest

from quant_backtest.factors import (
    CompositeFactor,
    CrossSectionalProcessor,
    MomentumFactor,
    ReversalFactor,
    VolatilityFactor,
)


def make_prices(periods=8, codes=("AAA.US",)):
    dates = pd.bdate_range("2024-01-01", periods=periods)
    return pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "close": 100 + index * (2 if code == "AAA.US" else 1),
                "adj_factor": 1.0,
            }
            for code in codes
            for index, date in enumerate(dates)
        ]
    )


@pytest.mark.parametrize(
    ("factor", "expected"),
    [
        (MomentumFactor(2), 0.04),
        (ReversalFactor(2), -0.04),
    ],
)
def test_return_factors_are_lagged(factor, expected):
    result = factor.compute(make_prices(6))

    assert result.iloc[0]["date"] == pd.Timestamp("2024-01-04")
    assert result.iloc[0]["factor_value"] == pytest.approx(expected)
    assert result.iloc[0]["factor_name"] == factor.name


def test_volatility_is_negative_and_lagged():
    result = VolatilityFactor(2).compute(make_prices(6))

    assert result.iloc[0]["date"] == pd.Timestamp("2024-01-04")
    assert result.iloc[0]["factor_value"] <= 0


def test_current_price_does_not_change_current_signal():
    prices = make_prices(6)
    original = MomentumFactor(2).compute(prices)
    target_date = prices["date"].max()
    prices.loc[prices["date"] == target_date, "close"] = 10_000
    changed = MomentumFactor(2).compute(prices)

    original_value = original.loc[
        original["date"] == target_date, "factor_value"
    ].iloc[0]
    changed_value = changed.loc[
        changed["date"] == target_date, "factor_value"
    ].iloc[0]
    assert changed_value == original_value


def test_adjustment_factor_prevents_false_split_signal():
    prices = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "code": ["AAA.US"] * 4,
            "close": [100, 102, 51, 52],
            "adj_factor": [1, 1, 2, 2],
        }
    )

    result = MomentumFactor(1).compute(prices)

    assert result.iloc[-1]["factor_value"] == pytest.approx(0.0)


def test_duplicate_price_rows_are_rejected():
    prices = make_prices(4)
    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        MomentumFactor(2).compute(prices)


def test_cross_section_winsorizes_and_standardizes():
    factors = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 5,
            "code": list("ABCDE"),
            "factor_name": ["test"] * 5,
            "factor_value": [1, 2, 3, 4, 100],
        }
    )

    result = CrossSectionalProcessor().transform(factors)

    assert result["factor_value"].mean() == pytest.approx(0.0)
    assert result["factor_value"].std(ddof=0) == pytest.approx(1.0)
    assert result["factor_value"].max() < 2


def test_zero_dispersion_cross_section_is_finite():
    factors = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "code": list("ABC"),
            "factor_value": [5, 5, 5],
        }
    )

    result = CrossSectionalProcessor().transform(factors)

    assert result["factor_value"].tolist() == [0.0, 0.0, 0.0]
    assert np.isfinite(result["factor_value"]).all()


class StaticFactor:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def compute(self, prices):
        return pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")] * len(self.values),
                "code": list(self.values),
                "factor_name": [self.name] * len(self.values),
                "factor_value": list(self.values.values()),
            }
        )


def test_composite_normalizes_weights_and_aligns_components():
    first = StaticFactor("first", {"A": 1, "B": 2, "C": 3})
    second = StaticFactor("second", {"A": 3, "B": 2, "C": 1})

    result = CompositeFactor([(first, 1), (second, 3)]).compute(pd.DataFrame())

    assert result["code"].tolist() == ["A", "B", "C"]
    assert result["factor_name"].unique().tolist() == ["composite"]
    assert result["factor_value"].tolist() == pytest.approx(
        [0.612372, 0.0, -0.612372]
    )


def test_composite_drops_rows_missing_a_component():
    first = StaticFactor("first", {"A": 1, "B": 2, "C": 3})
    second = StaticFactor("second", {"A": 3, "B": 2})

    result = CompositeFactor([(first, 1), (second, 1)]).compute(pd.DataFrame())

    assert result["code"].tolist() == ["A", "B"]
