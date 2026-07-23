from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceComparison:
    daily: pd.DataFrame
    strategy_metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    relative_metrics: dict[str, float]


class PerformanceAnalyzer:
    def __init__(self, annualization: int = 252, risk_free_rate: float = 0.0) -> None:
        if annualization < 1:
            raise ValueError("annualization must be positive")
        self.annualization = annualization
        self.risk_free_rate = risk_free_rate

    def analyze(self, daily: pd.DataFrame) -> dict[str, float]:
        if "return" not in daily.columns:
            raise ValueError("daily data must contain return")
        returns = pd.to_numeric(daily["return"], errors="coerce").dropna()
        if returns.empty:
            raise ValueError("returns must not be empty")

        net_value = (1.0 + returns).cumprod()
        periods = len(returns)
        annual_return = (
            float(net_value.iloc[-1] ** (self.annualization / periods) - 1.0)
            if net_value.iloc[-1] > 0
            else -1.0
        )
        drawdown = net_value / net_value.cummax() - 1.0
        daily_risk_free = (1.0 + self.risk_free_rate) ** (
            1.0 / self.annualization
        ) - 1.0
        excess = returns - daily_risk_free
        volatility = excess.std(ddof=1)
        sharpe = (
            float(excess.mean() / volatility * math.sqrt(self.annualization))
            if volatility > 0 and np.isfinite(volatility)
            else 0.0
        )
        return {
            "net_value": float(net_value.iloc[-1]),
            "annual_return": annual_return,
            "max_drawdown": float(drawdown.min()),
            "sharpe_ratio": sharpe,
        }

    def compare(
        self,
        strategy_daily: pd.DataFrame,
        benchmark_daily: pd.DataFrame,
    ) -> PerformanceComparison:
        strategy = self._prepare_daily(strategy_daily, "strategy")
        benchmark = self._prepare_daily(benchmark_daily, "benchmark")
        comparison = strategy.merge(
            benchmark,
            on="date",
            how="inner",
            validate="one_to_one",
        )
        if comparison.empty:
            raise ValueError("strategy and benchmark have no overlapping dates")

        comparison["strategy_net_value"] = (
            1.0 + comparison["strategy_return"]
        ).cumprod()
        comparison["benchmark_net_value"] = (
            1.0 + comparison["benchmark_return"]
        ).cumprod()
        comparison["excess_net_value"] = (
            comparison["strategy_net_value"]
            / comparison["benchmark_net_value"]
        )
        comparison["strategy_drawdown"] = (
            comparison["strategy_net_value"]
            / comparison["strategy_net_value"].cummax()
            - 1.0
        )
        comparison["benchmark_drawdown"] = (
            comparison["benchmark_net_value"]
            / comparison["benchmark_net_value"].cummax()
            - 1.0
        )

        strategy_metrics = self.analyze(
            comparison[["strategy_return"]].rename(
                columns={"strategy_return": "return"}
            )
        )
        benchmark_metrics = self.analyze(
            comparison[["benchmark_return"]].rename(
                columns={"benchmark_return": "return"}
            )
        )
        active_return = (
            comparison["strategy_return"] - comparison["benchmark_return"]
        )
        active_volatility = active_return.std(ddof=1)
        tracking_error = (
            float(active_volatility * math.sqrt(self.annualization))
            if np.isfinite(active_volatility)
            else 0.0
        )
        information_ratio = (
            float(
                active_return.mean()
                / active_volatility
                * math.sqrt(self.annualization)
            )
            if active_volatility > 0 and np.isfinite(active_volatility)
            else 0.0
        )
        relative_metrics = {
            "cumulative_excess_return": float(
                comparison["excess_net_value"].iloc[-1] - 1.0
            ),
            "annual_excess_return": (
                strategy_metrics["annual_return"]
                - benchmark_metrics["annual_return"]
            ),
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
        }
        return PerformanceComparison(
            daily=comparison,
            strategy_metrics=strategy_metrics,
            benchmark_metrics=benchmark_metrics,
            relative_metrics=relative_metrics,
        )

    @staticmethod
    def _prepare_daily(data: pd.DataFrame, label: str) -> pd.DataFrame:
        missing = {"date", "return"} - set(data.columns)
        if missing:
            raise ValueError(
                f"{label} daily data missing columns: {sorted(missing)}"
            )
        if data.empty:
            raise ValueError(f"{label} daily data must not be empty")
        prepared = data[["date", "return"]].copy()
        prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
        if prepared["date"].duplicated().any():
            raise ValueError(f"{label} daily data contains duplicate dates")
        prepared["return"] = pd.to_numeric(
            prepared["return"], errors="coerce"
        )
        prepared = prepared.dropna(subset=["return"]).sort_values("date")
        if prepared.empty:
            raise ValueError(f"{label} returns must not be empty")
        return prepared.rename(columns={"return": f"{label}_return"})
