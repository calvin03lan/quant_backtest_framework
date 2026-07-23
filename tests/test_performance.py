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


def test_compare_calculates_benchmark_and_relative_metrics():
    dates = pd.bdate_range("2024-01-01", periods=2)
    strategy = pd.DataFrame({"date": dates, "return": [0.10, 0.0]})
    benchmark = pd.DataFrame({"date": dates, "return": [0.0, 0.0]})

    comparison = PerformanceAnalyzer(annualization=2).compare(
        strategy, benchmark
    )

    assert comparison.daily["strategy_net_value"].iloc[-1] == pytest.approx(
        1.10
    )
    assert comparison.relative_metrics[
        "cumulative_excess_return"
    ] == pytest.approx(0.10)
    assert comparison.relative_metrics["annual_excess_return"] == pytest.approx(
        0.10
    )
    assert comparison.relative_metrics["tracking_error"] == pytest.approx(0.10)
    assert comparison.relative_metrics["information_ratio"] == pytest.approx(
        1.0
    )


def test_compare_rejects_non_overlapping_dates():
    strategy = pd.DataFrame(
        {"date": ["2024-01-01"], "return": [0.01]}
    )
    benchmark = pd.DataFrame(
        {"date": ["2024-01-02"], "return": [0.01]}
    )

    with pytest.raises(ValueError, match="overlapping"):
        PerformanceAnalyzer().compare(strategy, benchmark)
