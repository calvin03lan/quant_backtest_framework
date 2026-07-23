"""Small factor backtesting framework."""

from .config import Settings
from .engine import BacktestEngine, BacktestResult
from .execution import ExecutionConfig, MarketFeeSchedule
from .factors import (
    CompositeFactor,
    CrossSectionalProcessor,
    MomentumFactor,
    ReversalFactor,
    VolatilityFactor,
)
from .performance import PerformanceAnalyzer, PerformanceComparison
from .plotting import PriceChartEngine
from .reporting import (
    BacktestReportArtifacts,
    BacktestReportConfig,
    BacktestReportPlotter,
)
from .research import (
    MovingAverageResearchConfig,
    SingleAssetResearchResult,
    SingleAssetResearchRunner,
)
from .strategy import (
    MovingAverageStrategy,
    SingleAssetBacktestEngine,
    SingleAssetBacktestResult,
)

__all__ = [
    "BacktestEngine",
    "BacktestReportArtifacts",
    "BacktestReportConfig",
    "BacktestReportPlotter",
    "BacktestResult",
    "CompositeFactor",
    "CrossSectionalProcessor",
    "ExecutionConfig",
    "MarketFeeSchedule",
    "MomentumFactor",
    "MovingAverageResearchConfig",
    "MovingAverageStrategy",
    "PerformanceAnalyzer",
    "PerformanceComparison",
    "PriceChartEngine",
    "ReversalFactor",
    "Settings",
    "SingleAssetBacktestEngine",
    "SingleAssetBacktestResult",
    "SingleAssetResearchResult",
    "SingleAssetResearchRunner",
    "VolatilityFactor",
]
