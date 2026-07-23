from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from quant_backtest.performance import PerformanceAnalyzer
from quant_backtest.reporting import (
    BacktestReportConfig,
    BacktestReportPlotter,
)
from quant_backtest.research import (
    MovingAverageResearchConfig,
    SingleAssetResearchResult,
)


def make_result(report=None):
    dates = pd.bdate_range("2024-01-01", periods=5)
    strategy_daily = pd.DataFrame(
        {
            "date": dates,
            "code": ["AAPL.US"] * 5,
            "adjusted_close": [10, 11, 12, 11, 13],
            "short_ma": [None, 10.5, 11.5, 11.5, 12],
            "long_ma": [None, None, 11, 11.33, 12],
            "position": [0, 0, 1, 1, 0],
            "return": [0.0, 0.0, 0.09, -0.08, 0.0],
        }
    )
    benchmark_daily = pd.DataFrame(
        {
            "date": dates,
            "code": ["SPY.US"] * 5,
            "adjusted_close": [20, 20.2, 20.4, 20.3, 20.5],
            "return": [0.0, 0.01, 0.01, -0.005, 0.01],
        }
    )
    performance = PerformanceAnalyzer().compare(
        strategy_daily[["date", "return"]],
        benchmark_daily[["date", "return"]],
    )
    config = MovingAverageResearchConfig(
        code="AAPL.US",
        benchmark_code="SPY.US",
        start=dates[0],
        end=dates[-1],
        short_window=2,
        long_window=3,
        report=report or BacktestReportConfig(),
    )
    return SingleAssetResearchResult(
        config=config,
        strategy_daily=strategy_daily,
        benchmark_daily=benchmark_daily,
        comparison_daily=performance.daily,
        performance=performance,
    )


def test_report_config_rejects_invalid_panels():
    with pytest.raises(ValueError, match="unsupported"):
        BacktestReportConfig(panels=("monthly_returns",))
    with pytest.raises(ValueError, match="duplicates"):
        BacktestReportConfig(panels=("equity", "equity"))


def test_build_respects_selected_panels_and_exposes_metrics(tmp_path: Path):
    output = tmp_path / "report.png"
    config = BacktestReportConfig(
        panels=("equity", "drawdown"),
        save_path=output,
    )

    artifacts = BacktestReportPlotter().build(
        make_result(report=config)
    )

    assert list(artifacts.axes) == ["equity", "drawdown"]
    assert artifacts.metrics.index.tolist() == ["Strategy", "Benchmark"]
    assert "Information Ratio" in artifacts.relative_metrics.index
    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(artifacts.figure)


def test_individual_signal_plot_marks_entries_and_exits():
    figure, axis = BacktestReportPlotter().plot_signals(make_result())

    assert "Moving Averages" in axis.get_title()
    assert len(axis.lines) == 3
    assert len(axis.collections) == 2
    plt.close(figure)
