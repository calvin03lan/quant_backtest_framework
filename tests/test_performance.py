import pandas as pd
import pytest

from quant_backtest.performance import PerformanceAnalyzer


def test_performance_metrics_for_known_returns():
    daily = pd.DataFrame({"return": [0.10, -0.10, 0.05]})

    metrics = PerformanceAnalyzer(annualization=3).analyze(daily)

    assert metrics["net_value"] == pytest.approx(1.0395)
    assert metrics["annual_return"] == pytest.approx(0.0395)
    assert metrics["max_drawdown"] == pytest.approx(-0.10)
    assert metrics["sharpe_ratio"] > 0


def test_constant_returns_have_zero_sharpe():
    daily = pd.DataFrame({"return": [0.01, 0.01, 0.01]})

    metrics = PerformanceAnalyzer().analyze(daily)

    assert metrics["sharpe_ratio"] == 0.0
