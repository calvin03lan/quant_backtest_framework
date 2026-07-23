from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import (
    AkShareProvider,
    MarketDataService,
    MongoMarketDataRepository,
    YFinanceProvider,
)
from .execution import infer_market
from .performance import PerformanceAnalyzer, PerformanceComparison
from .reporting import BacktestReportConfig
from .strategy import MovingAverageStrategy, SingleAssetBacktestEngine


@dataclass(frozen=True)
class MovingAverageResearchConfig:
    code: str = "AAPL.US"
    benchmark_code: str = "SPY.US"
    start: str | pd.Timestamp = "2018-01-01"
    end: str | pd.Timestamp = "2024-12-31"
    short_window: int = 20
    long_window: int = 60
    transaction_cost_bps: float = 5.0
    risk_free_rate: float = 0.0
    auto_download: bool = True
    report: BacktestReportConfig = field(default_factory=BacktestReportConfig)

    def __post_init__(self) -> None:
        code = self.code.strip().upper() if isinstance(self.code, str) else ""
        benchmark = (
            self.benchmark_code.strip().upper()
            if isinstance(self.benchmark_code, str)
            else ""
        )
        if not code or not benchmark:
            raise ValueError("code and benchmark_code must be non-empty strings")
        if code == benchmark:
            raise ValueError("code and benchmark_code must be different")
        start = pd.Timestamp(self.start).normalize()
        end = pd.Timestamp(self.end).normalize()
        if start > end:
            raise ValueError("start date must not be after end date")
        MovingAverageStrategy(self.short_window, self.long_window)
        SingleAssetBacktestEngine(self.transaction_cost_bps)
        if not np.isfinite(self.risk_free_rate) or self.risk_free_rate <= -1:
            raise ValueError("risk_free_rate must be finite and greater than -1")
        if not isinstance(self.auto_download, bool):
            raise ValueError("auto_download must be a boolean")

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "benchmark_code", benchmark)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class SingleAssetResearchResult:
    config: MovingAverageResearchConfig
    strategy_daily: pd.DataFrame
    benchmark_daily: pd.DataFrame
    comparison_daily: pd.DataFrame
    performance: PerformanceComparison


class SingleAssetResearchRunner:
    def __init__(
        self,
        repository: MongoMarketDataRepository | None = None,
        data_service: MarketDataService | None = None,
    ) -> None:
        self.repository = repository or MongoMarketDataRepository()
        self.data_service = data_service or MarketDataService(
            self.repository,
            {"CN": AkShareProvider(), "US": YFinanceProvider()},
        )

    def run(
        self,
        config: MovingAverageResearchConfig,
    ) -> SingleAssetResearchResult:
        prices = self._load_prices(config.code, config)
        benchmark_prices = self._load_prices(config.benchmark_code, config)

        signals = MovingAverageStrategy(
            short_window=config.short_window,
            long_window=config.long_window,
        ).generate_signals(prices)
        strategy_daily = SingleAssetBacktestEngine(
            transaction_cost_bps=config.transaction_cost_bps
        ).run(prices, signals).daily
        benchmark_daily = self._prepare_benchmark(benchmark_prices)
        performance = PerformanceAnalyzer(
            risk_free_rate=config.risk_free_rate
        ).compare(
            strategy_daily[["date", "return"]],
            benchmark_daily[["date", "return"]],
        )
        return SingleAssetResearchResult(
            config=config,
            strategy_daily=strategy_daily,
            benchmark_daily=benchmark_daily,
            comparison_daily=performance.daily,
            performance=performance,
        )

    def _load_prices(
        self,
        code: str,
        config: MovingAverageResearchConfig,
    ) -> pd.DataFrame:
        data = self.repository.read_prices(
            codes=[code],
            start=config.start,
            end=config.end,
        )
        if not data.empty:
            return data
        if not config.auto_download:
            raise ValueError(
                f"no local price data for {code} and auto_download is disabled"
            )

        load_result = self.data_service.load(
            {infer_market(code): [code]},
            start=config.start,
            end=config.end,
            incremental=False,
        )
        data = self.repository.read_prices(
            codes=[code],
            start=config.start,
            end=config.end,
        )
        if data.empty:
            raise ValueError(
                f"unable to load price data for {code}: {load_result.get(code)}"
            )
        return data

    @staticmethod
    def _prepare_benchmark(prices: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "code", "close", "adj_factor"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(
                f"benchmark prices missing columns: {sorted(missing)}"
            )
        prepared = prices.copy()
        prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
        if prepared["code"].nunique() != 1:
            raise ValueError("benchmark prices must contain exactly one code")
        if prepared.duplicated(["code", "date"]).any():
            raise ValueError("benchmark prices contain duplicate code/date rows")
        prepared["adjusted_close"] = (
            pd.to_numeric(prepared["close"], errors="coerce")
            * pd.to_numeric(prepared["adj_factor"], errors="coerce")
        )
        if (
            prepared["adjusted_close"].isna().any()
            or not np.isfinite(prepared["adjusted_close"]).all()
            or (prepared["adjusted_close"] <= 0).any()
        ):
            raise ValueError("benchmark prices contain invalid adjusted close")
        prepared = prepared.sort_values("date").reset_index(drop=True)
        prepared["return"] = prepared["adjusted_close"].pct_change(
            fill_method=None
        )
        prepared["return"] = prepared["return"].fillna(0.0)
        return prepared[
            ["date", "code", "adjusted_close", "return"]
        ]
