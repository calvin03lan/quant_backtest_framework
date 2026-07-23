import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from quant_backtest.plotting import PriceChartEngine


def make_prices(periods=30, adj_factor=1.0):
    dates = pd.bdate_range("2024-01-01", periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "code": ["AAPL.US"] * periods,
            "market": ["US"] * periods,
            "open": [100 + index for index in range(periods)],
            "high": [102 + index for index in range(periods)],
            "low": [99 + index for index in range(periods)],
            "close": [101 + index for index in range(periods)],
            "adj_factor": [adj_factor] * periods,
            "volume": [1_000 + index for index in range(periods)],
        }
    )


class FakeRepository:
    def __init__(self, data):
        self.data = data

    def list_price_codes(self):
        return sorted(self.data["code"].unique())

    def read_prices(self, *, codes, start, end):
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        return self.data[
            self.data["code"].isin(codes)
            & self.data["date"].between(start_date, end_date)
        ].copy()


def make_engine(periods=30, adj_factor=1.0):
    return PriceChartEngine(FakeRepository(make_prices(periods, adj_factor)))


def test_line_chart_defaults_to_close():
    engine = make_engine(10)

    figure, axes = engine.plot(
        "AAPL.US", "2024-01-01", "2024-01-31", chart_type="line"
    )

    assert axes[0].lines[0].get_ydata().tolist() == list(range(101, 111))
    assert "Close Price" in axes[0].get_title()
    plt.close(figure)


def test_line_chart_can_use_adjusted_price():
    engine = make_engine(5, adj_factor=0.5)

    figure, axes = engine.plot(
        "AAPL.US",
        "2024-01-01",
        "2024-01-31",
        adjusted=True,
    )

    assert axes[0].lines[0].get_ydata()[0] == pytest.approx(50.5)
    plt.close(figure)


@pytest.mark.parametrize(
    ("periods", "expected_frequency"),
    [(160, "daily"), (161, "weekly"), (751, "monthly")],
)
def test_candlestick_auto_frequency(periods, expected_frequency):
    data = make_prices(periods)

    prepared, frequency = PriceChartEngine(
        FakeRepository(data)
    ).prepare_candlestick_data(data, frequency="auto")

    assert frequency == expected_frequency
    assert len(prepared) <= periods


def test_candlestick_volume_adds_subplot():
    engine = make_engine(40)

    figure_without, axes_without = engine.plot(
        "AAPL.US",
        "2024-01-01",
        "2024-03-31",
        chart_type="candlestick",
        show_volume=False,
    )
    figure_with, axes_with = engine.plot(
        "AAPL.US",
        "2024-01-01",
        "2024-03-31",
        chart_type="candlestick",
        show_volume=True,
    )

    assert len(axes_with) > len(axes_without)
    plt.close(figure_without)
    plt.close(figure_with)


def test_plot_saves_nonempty_png(tmp_path: Path):
    output = tmp_path / "nested" / "chart.png"

    figure, _ = make_engine(10).plot(
        "AAPL.US",
        "2024-01-01",
        "2024-01-31",
        save_path=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(figure)


def test_available_codes_and_invalid_inputs():
    engine = make_engine(10)

    assert engine.available_codes() == ["AAPL.US"]
    with pytest.raises(ValueError, match="start date"):
        engine.plot("AAPL.US", "2024-02-01", "2024-01-01")
    with pytest.raises(ValueError, match="chart_type"):
        engine.plot(
            "AAPL.US", "2024-01-01", "2024-01-31", chart_type="area"
        )
    with pytest.raises(ValueError, match="no price data"):
        engine.plot("MSFT.US", "2024-01-01", "2024-01-31")


def test_multiple_codes_are_rejected():
    data = pd.concat(
        [
            make_prices(2),
            make_prices(2).assign(code="MSFT.US"),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="exactly one code"):
        PriceChartEngine._prepare_single_symbol(data)


def test_notebook_parameters_compile_and_use_public_engine():
    notebook_path = (
        Path(__file__).parents[1] / "notebooks" / "price_chart_demo.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    combined = "\n".join(code_sources)

    for source in code_sources:
        compile(source, str(notebook_path), "exec")
    assert "PriceChartEngine" in combined
    assert 'CHART_TYPE = "line"' in combined
    assert 'PRICE_COLUMN = "close"' in combined
    assert "ipywidgets" not in combined
