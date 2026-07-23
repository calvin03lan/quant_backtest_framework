from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from .data.repository import MongoMarketDataRepository

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


ChartType = Literal["line", "candlestick"]
Frequency = Literal["auto", "daily", "weekly", "monthly"]
PRICE_COLUMNS = {"open", "high", "low", "close"}
FREQUENCY_LABELS = {
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}


class PriceChartEngine:
    def __init__(
        self, repository: MongoMarketDataRepository | None = None
    ) -> None:
        self.repository = repository or MongoMarketDataRepository()

    def available_codes(self) -> list[str]:
        return self.repository.list_price_codes()

    def load_prices(
        self,
        code: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        start_date = pd.Timestamp(start).normalize()
        end_date = pd.Timestamp(end).normalize()
        if start_date > end_date:
            raise ValueError("start date must not be after end date")
        data = self.repository.read_prices(
            codes=[code.strip().upper()],
            start=start_date,
            end=end_date,
        )
        if data.empty:
            raise ValueError(
                f"no price data for {code} between "
                f"{start_date.date()} and {end_date.date()}"
            )
        return self._prepare_single_symbol(data)

    def plot(
        self,
        code: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        chart_type: ChartType = "line",
        price_column: str = "close",
        adjusted: bool = False,
        frequency: Frequency = "auto",
        show_volume: bool = True,
        save_path: str | Path | None = None,
    ) -> tuple[Figure, list[Axes]]:
        if chart_type not in {"line", "candlestick"}:
            raise ValueError("chart_type must be 'line' or 'candlestick'")
        if price_column not in PRICE_COLUMNS:
            raise ValueError(
                f"price_column must be one of {sorted(PRICE_COLUMNS)}"
            )
        if frequency not in {"auto", "daily", "weekly", "monthly"}:
            raise ValueError(
                "frequency must be 'auto', 'daily', 'weekly', or 'monthly'"
            )

        data = self.load_prices(code, start, end)
        if chart_type == "line":
            figure, axes = self._plot_line(
                data,
                code=code,
                price_column=price_column,
                adjusted=adjusted,
            )
        else:
            prepared, resolved_frequency = self.prepare_candlestick_data(
                data, adjusted=adjusted, frequency=frequency
            )
            figure, axes = self._plot_candlestick(
                prepared,
                code=code,
                frequency=resolved_frequency,
                adjusted=adjusted,
                show_volume=show_volume,
            )

        if save_path is not None:
            output = Path(save_path).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=160, bbox_inches="tight")
        return figure, list(axes)

    def prepare_candlestick_data(
        self,
        data: pd.DataFrame,
        *,
        adjusted: bool = False,
        frequency: Frequency = "auto",
    ) -> tuple[pd.DataFrame, str]:
        prepared = self._prepare_single_symbol(data)
        resolved = self._resolve_frequency(len(prepared), frequency)
        if adjusted:
            for column in PRICE_COLUMNS:
                prepared[column] = (
                    prepared[column] * prepared["adj_factor"]
                )
        prepared = prepared.set_index("date").sort_index()
        if resolved == "weekly":
            prepared = self._resample_ohlcv(prepared, "W-FRI")
        elif resolved == "monthly":
            prepared = self._resample_ohlcv(prepared, "ME")
        prepared = prepared.dropna(subset=["open", "high", "low", "close"])
        if prepared.empty:
            raise ValueError("no complete OHLC rows available for candlestick chart")
        return prepared, resolved

    @staticmethod
    def _prepare_single_symbol(data: pd.DataFrame) -> pd.DataFrame:
        required = {
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "adj_factor",
            "volume",
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"price data missing columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("price data must not be empty")

        prepared = data.copy()
        prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
        if prepared["code"].nunique() != 1:
            raise ValueError("price chart supports exactly one code")
        prepared = prepared.sort_values("date").drop_duplicates(
            ["code", "date"], keep="last"
        )
        numeric_columns = [*sorted(PRICE_COLUMNS), "adj_factor", "volume"]
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        finite_ohlc = np.isfinite(prepared[list(PRICE_COLUMNS)]).all(axis=1)
        prepared = prepared.loc[finite_ohlc].reset_index(drop=True)
        if prepared.empty:
            raise ValueError("price data contains no finite OHLC rows")
        return prepared

    @staticmethod
    def _resolve_frequency(size: int, frequency: Frequency) -> str:
        if frequency != "auto":
            return frequency
        if size <= 160:
            return "daily"
        if size <= 750:
            return "weekly"
        return "monthly"

    @staticmethod
    def _resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
        aggregation = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "adj_factor": "last",
            "code": "last",
        }
        if "market" in data.columns:
            aggregation["market"] = "last"
        return data.resample(rule).agg(aggregation)

    @staticmethod
    def _plot_line(
        data: pd.DataFrame,
        *,
        code: str,
        price_column: str,
        adjusted: bool,
    ) -> tuple[Figure, list[Axes]]:
        values = data[price_column]
        if adjusted:
            values = values * data["adj_factor"]
        width = min(18.0, max(10.0, 9.0 + len(data) / 350))
        figure, axis = plt.subplots(figsize=(width, 5.5), dpi=120)
        axis.plot(
            data["date"],
            values,
            color="#1f77b4",
            linewidth=1.5,
            label=price_column,
        )
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        mode = "Adjusted" if adjusted else "Raw"
        axis.set_title(f"{code.upper()} {mode} {price_column.title()} Price")
        axis.set_xlabel("Date")
        axis.set_ylabel("Price")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.01)
        figure.tight_layout()
        return figure, [axis]

    @staticmethod
    def _plot_candlestick(
        data: pd.DataFrame,
        *,
        code: str,
        frequency: str,
        adjusted: bool,
        show_volume: bool,
    ) -> tuple[Figure, list[Axes]]:
        plot_data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        width = min(18.0, max(10.0, 9.0 + len(plot_data) / 80))
        mode = "Adjusted" if adjusted else "Raw"
        figure, axes = mpf.plot(
            plot_data[["Open", "High", "Low", "Close", "Volume"]],
            type="candle",
            volume=show_volume,
            style="yahoo",
            figsize=(width, 7 if show_volume else 5.5),
            title=(
                f"{code.upper()} {mode} Candlestick "
                f"({FREQUENCY_LABELS[frequency]})"
            ),
            ylabel="Price",
            ylabel_lower="Volume",
            warn_too_much_data=10_000,
            returnfig=True,
        )
        return figure, list(axes)
