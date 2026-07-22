from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame
    rebalances: pd.DataFrame


class BacktestEngine:
    def __init__(self, top_n: int = 5) -> None:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        self.top_n = top_n

    def run(self, prices: pd.DataFrame, factors: pd.DataFrame) -> BacktestResult:
        self._validate(prices, factors)
        adjusted = prices.assign(
            date=pd.to_datetime(prices["date"]),
            adjusted_close=prices["close"] * prices["adj_factor"],
        )
        close = (
            adjusted.pivot_table(
                index="date", columns="code", values="adjusted_close", aggfunc="last"
            )
            .sort_index()
            .sort_index(axis=1)
        )
        returns = close.pct_change(fill_method=None)
        signals = (
            factors.assign(date=pd.to_datetime(factors["date"]))
            .pivot_table(
                index="date", columns="code", values="factor_value", aggfunc="last"
            )
            .reindex(index=close.index, columns=close.columns)
        )

        targets = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        rebalance_dates = close.groupby(close.index.to_period("M")).apply(
            lambda frame: frame.index[0]
        )
        rebalance_records: list[dict] = []
        for date in rebalance_dates:
            targets.loc[date] = 0.0
            ranked = signals.loc[date].dropna().sort_values(ascending=False)
            selected = ranked.head(self.top_n).index
            if len(selected):
                weight = 1.0 / len(selected)
                targets.loc[date, selected] = weight
                rebalance_records.extend(
                    {
                        "date": date,
                        "code": code,
                        "factor_value": float(ranked.loc[code]),
                        "target_weight": weight,
                    }
                    for code in selected
                )

        target_positions = targets.ffill().fillna(0.0)
        # Targets are formed at each rebalance close and take effect next session.
        effective_positions = target_positions.shift(1).fillna(0.0)
        portfolio_returns = (effective_positions * returns.fillna(0.0)).sum(axis=1)
        daily = pd.DataFrame(
            {
                "date": close.index,
                "return": portfolio_returns.to_numpy(),
                "net_value": (1.0 + portfolio_returns).cumprod().to_numpy(),
            }
        )
        positions = (
            effective_positions.rename_axis(index="date", columns="code")
            .stack()
            .rename("weight")
            .reset_index()
        )
        rebalances = pd.DataFrame(
            rebalance_records,
            columns=["date", "code", "factor_value", "target_weight"],
        )
        return BacktestResult(daily, positions, rebalances)

    @staticmethod
    def _validate(prices: pd.DataFrame, factors: pd.DataFrame) -> None:
        price_missing = {"date", "code", "close", "adj_factor"} - set(prices.columns)
        factor_missing = {"date", "code", "factor_value"} - set(factors.columns)
        if price_missing:
            raise ValueError(f"prices missing columns: {sorted(price_missing)}")
        if factor_missing:
            raise ValueError(f"factors missing columns: {sorted(factor_missing)}")
        if prices.empty:
            raise ValueError("prices must not be empty")
        if factors.empty:
            raise ValueError("factors must not be empty")
