import json
from pathlib import Path

import pandas as pd
import pytest

from quant_backtest.research import (
    MovingAverageResearchConfig,
    SingleAssetResearchRunner,
)
from quant_backtest.reporting import BacktestReportConfig


def make_prices(code, values, dates=None):
    resolved_dates = (
        pd.bdate_range("2024-01-01", periods=len(values))
        if dates is None
        else pd.DatetimeIndex(dates)
    )
    return pd.DataFrame(
        {
            "date": resolved_dates,
            "code": [code] * len(values),
            "close": values,
            "adj_factor": [1.0] * len(values),
        }
    )


class FakeRepository:
    def __init__(self, data):
        self.data = data.copy()

    def read_prices(self, *, codes, start, end):
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        return self.data[
            self.data["code"].isin(codes)
            & self.data["date"].between(start_date, end_date)
        ].copy()


class FakeDataService:
    def __init__(self, repository, downloadable):
        self.repository = repository
        self.downloadable = downloadable
        self.calls = []

    def load(self, codes_by_market, *, start, end, incremental):
        self.calls.append(
            {
                "codes_by_market": codes_by_market,
                "start": start,
                "end": end,
                "incremental": incremental,
            }
        )
        counts = {}
        for codes in codes_by_market.values():
            for code in codes:
                data = self.downloadable.get(code, pd.DataFrame())
                self.repository.data = pd.concat(
                    [self.repository.data, data], ignore_index=True
                )
                counts[code] = len(data)
        return counts


def make_config(**overrides):
    values = {
        "code": "AAPL.US",
        "benchmark_code": "SPY.US",
        "start": "2024-01-01",
        "end": "2024-01-31",
        "short_window": 2,
        "long_window": 3,
    }
    values.update(overrides)
    return MovingAverageResearchConfig(**values)


def test_research_config_normalizes_values_and_validates_dates():
    config = make_config(
        code=" aapl.us ",
        benchmark_code=" spy.us ",
        report=BacktestReportConfig(panels=("equity",)),
    )

    assert config.code == "AAPL.US"
    assert config.benchmark_code == "SPY.US"
    assert config.start == pd.Timestamp("2024-01-01")
    assert config.report.panels == ("equity",)
    with pytest.raises(ValueError, match="start date"):
        make_config(start="2024-02-01", end="2024-01-01")


def test_runner_builds_standard_result_and_aligns_benchmark_dates():
    strategy = make_prices("AAPL.US", [10, 11, 12, 13, 14, 15])
    benchmark = make_prices(
        "SPY.US",
        [20, 21, 22, 23, 24],
        dates=strategy["date"].iloc[1:],
    )
    repository = FakeRepository(pd.concat([strategy, benchmark]))
    service = FakeDataService(repository, {})

    result = SingleAssetResearchRunner(repository, service).run(make_config())

    assert service.calls == []
    assert len(result.strategy_daily) == 6
    assert len(result.benchmark_daily) == 5
    assert len(result.comparison_daily) == 5
    assert set(result.performance.relative_metrics) == {
        "cumulative_excess_return",
        "annual_excess_return",
        "tracking_error",
        "information_ratio",
    }


def test_runner_automatically_downloads_missing_benchmark():
    strategy = make_prices("AAPL.US", [10, 11, 12, 13, 14, 15])
    benchmark = make_prices("SPY.US", [20, 21, 22, 23, 24, 25])
    repository = FakeRepository(strategy)
    service = FakeDataService(repository, {"SPY.US": benchmark})

    result = SingleAssetResearchRunner(repository, service).run(make_config())

    assert service.calls[0]["codes_by_market"] == {"US": ["SPY.US"]}
    assert service.calls[0]["incremental"] is False
    assert len(result.benchmark_daily) == 6


def test_runner_can_disable_automatic_download():
    strategy = make_prices("AAPL.US", [10, 11, 12, 13, 14, 15])
    repository = FakeRepository(strategy)
    service = FakeDataService(repository, {})

    with pytest.raises(ValueError, match="auto_download is disabled"):
        SingleAssetResearchRunner(repository, service).run(
            make_config(auto_download=False)
        )
    assert service.calls == []


def test_backtest_notebook_is_a_thin_config_driven_template():
    notebook_path = (
        Path(__file__).parents[1] / "notebooks" / "backtest_template.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    combined = "\n".join(code_sources)

    assert len(code_sources) == 4
    for source in code_sources:
        compile(source, str(notebook_path), "exec")
    assert "MovingAverageResearchConfig" in combined
    assert "BacktestReportConfig" in combined
    assert "SingleAssetResearchRunner" in combined
    assert "BacktestReportPlotter" in combined
    for implementation_detail in [
        "plt.subplots",
        "MarketDataService",
        ".merge(",
        "pct_change",
        "ipywidgets",
    ]:
        assert implementation_detail not in combined
