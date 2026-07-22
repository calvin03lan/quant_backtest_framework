from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd


class Factor(Protocol):
    name: str

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return date, code, factor_name and factor_value columns."""


class MomentumFactor:
    name = "momentum"

    def __init__(self, lookback: int = 20) -> None:
        _validate_window(lookback, "lookback")
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        data = _prepare_prices(prices)
        data["factor_value"] = data.groupby("code")["adjusted_close"].transform(
            lambda values: values.pct_change(self.lookback).shift(1)
        )
        return _format_result(data, self.name)


class ReversalFactor:
    name = "reversal"

    def __init__(self, lookback: int = 5) -> None:
        _validate_window(lookback, "lookback")
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        data = _prepare_prices(prices)
        data["factor_value"] = data.groupby("code")["adjusted_close"].transform(
            lambda values: -values.pct_change(self.lookback).shift(1)
        )
        return _format_result(data, self.name)


class VolatilityFactor:
    name = "volatility"

    def __init__(self, window: int = 20) -> None:
        _validate_window(window, "window")
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        data = _prepare_prices(prices)
        daily_returns = data.groupby("code")["adjusted_close"].pct_change()
        data["factor_value"] = daily_returns.groupby(data["code"]).transform(
            lambda values: -values.rolling(
                self.window, min_periods=self.window
            ).std().shift(1)
        )
        return _format_result(data, self.name)


class CrossSectionalProcessor:
    def __init__(self, mad_threshold: float = 3.0) -> None:
        if mad_threshold <= 0:
            raise ValueError("mad_threshold must be positive")
        self.mad_threshold = mad_threshold

    def transform(self, factors: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "code", "factor_value"}
        missing = required - set(factors.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        result = factors.copy()
        result["date"] = pd.to_datetime(result["date"])
        result["factor_value"] = pd.to_numeric(
            result["factor_value"], errors="coerce"
        )
        result["factor_value"] = result.groupby("date")["factor_value"].transform(
            self._winsorize_and_standardize
        )
        return result.dropna(subset=["factor_value"]).reset_index(drop=True)

    def _winsorize_and_standardize(self, values: pd.Series) -> pd.Series:
        median = values.median()
        mad = (values - median).abs().median()
        clipped = values
        if pd.notna(mad) and mad > 0:
            limit = self.mad_threshold * mad
            clipped = values.clip(median - limit, median + limit)
        standard_deviation = clipped.std(ddof=0)
        if pd.isna(standard_deviation) or standard_deviation == 0:
            return pd.Series(0.0, index=values.index).where(values.notna())
        return (clipped - clipped.mean()) / standard_deviation


class CompositeFactor:
    name = "composite"

    def __init__(
        self,
        factors: Sequence[tuple[Factor, float]],
        processor: CrossSectionalProcessor | None = None,
    ) -> None:
        if not factors:
            raise ValueError("at least one factor is required")
        if any(weight < 0 for _, weight in factors):
            raise ValueError("factor weights must be non-negative")
        total_weight = sum(weight for _, weight in factors)
        if total_weight <= 0:
            raise ValueError("factor weights must have a positive sum")
        names = [factor.name for factor, _ in factors]
        if len(names) != len(set(names)):
            raise ValueError("factor names must be unique")
        self.factors = [
            (factor, weight / total_weight) for factor, weight in factors
        ]
        self.processor = processor or CrossSectionalProcessor()

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        components: list[pd.DataFrame] = []
        for factor, weight in self.factors:
            processed = self.processor.transform(factor.compute(prices))
            components.append(
                processed[["date", "code", "factor_value"]].rename(
                    columns={"factor_value": factor.name}
                )
            )

        combined = components[0]
        for component in components[1:]:
            combined = combined.merge(component, on=["date", "code"], how="inner")
        combined["factor_value"] = sum(
            combined[factor.name] * weight for factor, weight in self.factors
        )
        combined["factor_name"] = self.name
        return combined[
            ["date", "code", "factor_name", "factor_value"]
        ].reset_index(drop=True)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "code", "close", "adj_factor"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    data = prices[["date", "code", "close", "adj_factor"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    if data.duplicated(["code", "date"]).any():
        raise ValueError("duplicate (code, date) rows are not allowed")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["adj_factor"] = pd.to_numeric(data["adj_factor"], errors="coerce")
    data = data.sort_values(["code", "date"])
    data["adjusted_close"] = data["close"] * data["adj_factor"]
    data["adjusted_close"] = data["adjusted_close"].replace(
        [np.inf, -np.inf], np.nan
    )
    return data


def _format_result(data: pd.DataFrame, name: str) -> pd.DataFrame:
    result = data[["date", "code", "factor_value"]].dropna().copy()
    result["factor_name"] = name
    return result[
        ["date", "code", "factor_name", "factor_value"]
    ].reset_index(drop=True)


def _validate_window(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")
