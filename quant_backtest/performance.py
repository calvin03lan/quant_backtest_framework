from __future__ import annotations

import math

import numpy as np
import pandas as pd


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
