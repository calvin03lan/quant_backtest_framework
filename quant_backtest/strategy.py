from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SingleAssetBacktestResult:
    daily: pd.DataFrame


@dataclass(frozen=True)
class MovingAverageStrategy:
    short_window: int = 20
    long_window: int = 60

    def __post_init__(self) -> None:
        if self.short_window < 1:
            raise ValueError("short_window must be positive")
        if self.long_window <= self.short_window:
            raise ValueError("long_window must be greater than short_window")

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        prepared = _prepare_prices(prices)
        prepared["short_ma"] = prepared["adjusted_close"].rolling(
            self.short_window, min_periods=self.short_window
        ).mean()
        prepared["long_ma"] = prepared["adjusted_close"].rolling(
            self.long_window, min_periods=self.long_window
        ).mean()
        prepared["signal"] = (
            (prepared["short_ma"] > prepared["long_ma"])
            & prepared["long_ma"].notna()
        ).astype(float)
        return prepared[
            [
                "date",
                "code",
                "adjusted_close",
                "short_ma",
                "long_ma",
                "signal",
            ]
        ]


class SingleAssetBacktestEngine:
    def __init__(self, transaction_cost_bps: float = 0.0) -> None:
        if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be finite and non-negative")
        self.transaction_cost_bps = float(transaction_cost_bps)

    def run(
        self, prices: pd.DataFrame, signals: pd.DataFrame
    ) -> SingleAssetBacktestResult:
        prepared = _prepare_prices(prices)
        prepared_signals = self._prepare_signals(signals, prepared["code"].iloc[0])
        daily = prepared.merge(
            prepared_signals,
            on=["date", "code"],
            how="left",
            validate="one_to_one",
        )
        daily["signal"] = daily["signal"].ffill().fillna(0.0)
        daily["position"] = daily["signal"].shift(1).fillna(0.0)
        daily["asset_return"] = daily["adjusted_close"].pct_change(fill_method=None)
        daily["turnover"] = daily["position"].diff().abs()
        daily.loc[daily.index[0], "turnover"] = abs(daily.loc[daily.index[0], "position"])
        daily["transaction_cost"] = (
            daily["turnover"] * self.transaction_cost_bps / 10_000.0
        )
        daily["gross_return"] = (
            daily["position"] * daily["asset_return"].fillna(0.0)
        )
        daily["return"] = daily["gross_return"] - daily["transaction_cost"]
        daily["net_value"] = (1.0 + daily["return"]).cumprod()
        daily["drawdown"] = (
            daily["net_value"] / daily["net_value"].cummax() - 1.0
        )
        return SingleAssetBacktestResult(
            daily=daily[
                [
                    "date",
                    "code",
                    "adjusted_close",
                    "short_ma",
                    "long_ma",
                    "signal",
                    "position",
                    "asset_return",
                    "turnover",
                    "transaction_cost",
                    "gross_return",
                    "return",
                    "net_value",
                    "drawdown",
                ]
            ]
        )

    @staticmethod
    def _prepare_signals(signals: pd.DataFrame, code: str) -> pd.DataFrame:
        required = {"date", "code", "short_ma", "long_ma", "signal"}
        missing = required - set(signals.columns)
        if missing:
            raise ValueError(f"signals missing columns: {sorted(missing)}")
        if signals.empty:
            raise ValueError("signals must not be empty")

        prepared = signals.copy()
        prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
        prepared["code"] = prepared["code"].astype(str).str.upper()
        if prepared["code"].nunique() != 1 or prepared["code"].iloc[0] != code:
            raise ValueError("signals must contain the same single code as prices")
        if prepared.duplicated(["code", "date"]).any():
            raise ValueError("signals contain duplicate code/date rows")
        prepared["signal"] = pd.to_numeric(prepared["signal"], errors="coerce")
        if (
            prepared["signal"].isna().any()
            or not prepared["signal"].isin([0.0, 1.0]).all()
        ):
            raise ValueError("signal must contain only 0 or 1")
        return prepared[
            ["date", "code", "short_ma", "long_ma", "signal"]
        ].sort_values("date")


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "code", "close", "adj_factor"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {sorted(missing)}")
    if prices.empty:
        raise ValueError("prices must not be empty")

    prepared = prices.copy()
    prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
    prepared["code"] = prepared["code"].astype(str).str.upper()
    if prepared["code"].nunique() != 1:
        raise ValueError("single-asset strategy supports exactly one code")
    if prepared.duplicated(["code", "date"]).any():
        raise ValueError("prices contain duplicate code/date rows")
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared["adj_factor"] = pd.to_numeric(
        prepared["adj_factor"], errors="coerce"
    )
    prepared["adjusted_close"] = prepared["close"] * prepared["adj_factor"]
    if (
        prepared["adjusted_close"].isna().any()
        or not np.isfinite(prepared["adjusted_close"]).all()
        or (prepared["adjusted_close"] <= 0).any()
    ):
        raise ValueError("prices contain invalid adjusted close values")
    return prepared.sort_values("date").reset_index(drop=True)
