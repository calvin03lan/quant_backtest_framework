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
from .performance import PerformanceAnalyzer

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CompositeFactor",
    "CrossSectionalProcessor",
    "ExecutionConfig",
    "MarketFeeSchedule",
    "MomentumFactor",
    "PerformanceAnalyzer",
    "ReversalFactor",
    "Settings",
    "VolatilityFactor",
]
