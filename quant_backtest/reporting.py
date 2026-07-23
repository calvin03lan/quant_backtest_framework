from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .research import SingleAssetResearchResult


ReportPanel = Literal["signals", "equity", "drawdown"]
SUPPORTED_PANELS = {"signals", "equity", "drawdown"}


@dataclass(frozen=True)
class BacktestReportConfig:
    panels: tuple[ReportPanel, ...] = ("signals", "equity", "drawdown")
    figsize: tuple[float, float] = (14.0, 13.0)
    dpi: int = 120
    title: str | None = None
    save_path: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.panels:
            raise ValueError("panels must not be empty")
        invalid = set(self.panels) - SUPPORTED_PANELS
        if invalid:
            raise ValueError(f"unsupported report panels: {sorted(invalid)}")
        if len(set(self.panels)) != len(self.panels):
            raise ValueError("report panels must not contain duplicates")
        if len(self.figsize) != 2 or any(value <= 0 for value in self.figsize):
            raise ValueError("figsize must contain two positive values")
        if self.dpi < 1:
            raise ValueError("dpi must be positive")


@dataclass(frozen=True)
class BacktestReportArtifacts:
    figure: Figure
    axes: dict[str, Axes]
    metrics: pd.DataFrame
    relative_metrics: pd.Series


class BacktestReportPlotter:
    def build(
        self,
        result: SingleAssetResearchResult,
        config: BacktestReportConfig | None = None,
    ) -> BacktestReportArtifacts:
        resolved = config or result.config.report
        figure, axes_array = plt.subplots(
            len(resolved.panels),
            1,
            figsize=resolved.figsize,
            dpi=resolved.dpi,
            sharex=True,
            squeeze=False,
        )
        axes = {
            panel: axes_array[index, 0]
            for index, panel in enumerate(resolved.panels)
        }
        renderers = {
            "signals": self._draw_signals,
            "equity": self._draw_equity,
            "drawdown": self._draw_drawdown,
        }
        for panel, axis in axes.items():
            renderers[panel](result, axis)
            axis.grid(True, alpha=0.25)
            axis.margins(x=0.01)

        last_axis = axes[resolved.panels[-1]]
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        last_axis.xaxis.set_major_locator(locator)
        last_axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(locator)
        )
        last_axis.set_xlabel("Date")
        if resolved.title:
            figure.suptitle(resolved.title)
        figure.tight_layout()
        if resolved.save_path is not None:
            output = Path(resolved.save_path).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=resolved.dpi, bbox_inches="tight")

        metrics, relative = self.metrics_tables(result)
        return BacktestReportArtifacts(
            figure=figure,
            axes=axes,
            metrics=metrics,
            relative_metrics=relative,
        )

    def plot_signals(
        self,
        result: SingleAssetResearchResult,
    ) -> tuple[Figure, Axes]:
        figure, axis = plt.subplots(figsize=(14, 5.0), dpi=120)
        self._draw_signals(result, axis)
        self._finish_single_axis(figure, axis)
        return figure, axis

    def plot_equity(
        self,
        result: SingleAssetResearchResult,
    ) -> tuple[Figure, Axes]:
        figure, axis = plt.subplots(figsize=(14, 5.0), dpi=120)
        self._draw_equity(result, axis)
        self._finish_single_axis(figure, axis)
        return figure, axis

    def plot_drawdown(
        self,
        result: SingleAssetResearchResult,
    ) -> tuple[Figure, Axes]:
        figure, axis = plt.subplots(figsize=(14, 5.0), dpi=120)
        self._draw_drawdown(result, axis)
        self._finish_single_axis(figure, axis)
        return figure, axis

    @staticmethod
    def metrics_tables(
        result: SingleAssetResearchResult,
    ) -> tuple[pd.DataFrame, pd.Series]:
        performance = result.performance
        metrics = pd.DataFrame(
            {
                "Strategy": {
                    "Total Return": (
                        performance.strategy_metrics["net_value"] - 1.0
                    ),
                    "Annual Return": performance.strategy_metrics[
                        "annual_return"
                    ],
                    "Sharpe Ratio": performance.strategy_metrics[
                        "sharpe_ratio"
                    ],
                    "Max Drawdown": performance.strategy_metrics[
                        "max_drawdown"
                    ],
                },
                "Benchmark": {
                    "Total Return": (
                        performance.benchmark_metrics["net_value"] - 1.0
                    ),
                    "Annual Return": performance.benchmark_metrics[
                        "annual_return"
                    ],
                    "Sharpe Ratio": performance.benchmark_metrics[
                        "sharpe_ratio"
                    ],
                    "Max Drawdown": performance.benchmark_metrics[
                        "max_drawdown"
                    ],
                },
            }
        ).T
        relative = pd.Series(
            {
                "Cumulative Excess Return": performance.relative_metrics[
                    "cumulative_excess_return"
                ],
                "Annual Excess Return": performance.relative_metrics[
                    "annual_excess_return"
                ],
                "Tracking Error": performance.relative_metrics[
                    "tracking_error"
                ],
                "Information Ratio": performance.relative_metrics[
                    "information_ratio"
                ],
            },
            name="Relative Performance",
        )
        return metrics, relative

    @staticmethod
    def _draw_signals(
        result: SingleAssetResearchResult,
        axis: Axes,
    ) -> None:
        daily = result.strategy_daily
        required = {
            "date",
            "adjusted_close",
            "short_ma",
            "long_ma",
            "position",
        }
        BacktestReportPlotter._require_columns(daily, required, "strategy")
        position_change = daily["position"].diff().fillna(daily["position"])
        buys = daily[position_change > 0]
        sells = daily[position_change < 0]
        axis.plot(
            daily["date"],
            daily["adjusted_close"],
            label=result.config.code,
            linewidth=1.2,
        )
        axis.plot(
            daily["date"],
            daily["short_ma"],
            label=f"MA {result.config.short_window}",
            linewidth=1.0,
        )
        axis.plot(
            daily["date"],
            daily["long_ma"],
            label=f"MA {result.config.long_window}",
            linewidth=1.0,
        )
        axis.scatter(
            buys["date"],
            buys["adjusted_close"],
            marker="^",
            color="green",
            s=35,
            label="Buy",
        )
        axis.scatter(
            sells["date"],
            sells["adjusted_close"],
            marker="v",
            color="red",
            s=35,
            label="Sell",
        )
        axis.set_title("Price, Moving Averages and Trades")
        axis.set_ylabel("Adjusted Price")
        axis.legend(ncol=5)

    @staticmethod
    def _draw_equity(
        result: SingleAssetResearchResult,
        axis: Axes,
    ) -> None:
        daily = result.comparison_daily
        required = {
            "date",
            "strategy_net_value",
            "benchmark_net_value",
            "excess_net_value",
        }
        BacktestReportPlotter._require_columns(daily, required, "comparison")
        axis.plot(
            daily["date"],
            daily["strategy_net_value"],
            label="Strategy",
            linewidth=1.5,
        )
        axis.plot(
            daily["date"],
            daily["benchmark_net_value"],
            label=result.config.benchmark_code,
            linewidth=1.5,
        )
        axis.plot(
            daily["date"],
            daily["excess_net_value"],
            label="Strategy / Benchmark",
            linestyle="--",
            linewidth=1.2,
        )
        axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title("Net Value and Excess Performance")
        axis.set_ylabel("Net Value")
        axis.legend()

    @staticmethod
    def _draw_drawdown(
        result: SingleAssetResearchResult,
        axis: Axes,
    ) -> None:
        daily = result.comparison_daily
        required = {"date", "strategy_drawdown", "benchmark_drawdown"}
        BacktestReportPlotter._require_columns(daily, required, "comparison")
        axis.plot(
            daily["date"],
            daily["strategy_drawdown"],
            label="Strategy Drawdown",
        )
        axis.plot(
            daily["date"],
            daily["benchmark_drawdown"],
            label="Benchmark Drawdown",
        )
        axis.fill_between(
            daily["date"],
            daily["strategy_drawdown"],
            0,
            alpha=0.15,
        )
        axis.set_title("Drawdown")
        axis.set_ylabel("Drawdown")
        axis.legend()

    @staticmethod
    def _finish_single_axis(figure: Figure, axis: Axes) -> None:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.set_xlabel("Date")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.01)
        figure.tight_layout()

    @staticmethod
    def _require_columns(
        data: pd.DataFrame,
        required: set[str],
        label: str,
    ) -> None:
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"{label} data missing columns: {sorted(missing)}"
            )
