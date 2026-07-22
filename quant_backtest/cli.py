from __future__ import annotations

import argparse
import json

from .data import (
    AkShareProvider,
    MarketDataService,
    MongoMarketDataRepository,
    YFinanceProvider,
)
from .engine import BacktestEngine
from .factors import (
    CompositeFactor,
    MomentumFactor,
    ReversalFactor,
    VolatilityFactor,
)
from .performance import PerformanceAnalyzer


SAMPLE_CODES = {
    "CN": ["000001.SZ", "000002.SZ", "600036.SH", "601318.SH", "600519.SH"],
    "US": ["AAPL.US", "MSFT.US", "AMZN.US", "GOOGL.US", "NVDA.US"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small factor backtest framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="create MongoDB indexes and sample pool")

    load = subparsers.add_parser("load-data", help="load or update sample market data")
    load.add_argument("--start", default="2015-01-01")
    load.add_argument("--end")
    load.add_argument("--full", action="store_true", help="disable incremental loading")

    backtest = subparsers.add_parser("backtest", help="run the factor sample")
    backtest.add_argument("--start", required=True)
    backtest.add_argument("--end", required=True)
    backtest.add_argument(
        "--factor", choices=["momentum", "composite"], default="composite"
    )
    backtest.add_argument("--lookback", type=int, default=20)
    backtest.add_argument("--reversal-lookback", type=int, default=5)
    backtest.add_argument("--volatility-window", type=int, default=20)
    backtest.add_argument("--top-n", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = MongoMarketDataRepository()
    repository.ping()
    repository.create_indexes()
    pool_codes = SAMPLE_CODES["CN"] + SAMPLE_CODES["US"]
    repository.upsert_pool("sample", pool_codes, "2015-01-01")

    if args.command == "init-db":
        print("MongoDB indexes and sample pool are ready.")
        return

    service = MarketDataService(
        repository,
        {"CN": AkShareProvider(), "US": YFinanceProvider()},
    )
    if args.command == "load-data":
        counts = service.load(
            SAMPLE_CODES,
            start=args.start,
            end=args.end,
            incremental=not args.full,
        )
        print(json.dumps(counts, indent=2, ensure_ascii=False))
        return

    prices = service.read(
        pool_id="sample",
        start=args.start,
        end=args.end,
        as_of=args.end,
    )
    if args.factor == "momentum":
        factor = MomentumFactor(args.lookback)
    else:
        factor = CompositeFactor(
            [
                (MomentumFactor(args.lookback), 1.0),
                (ReversalFactor(args.reversal_lookback), 1.0),
                (VolatilityFactor(args.volatility_window), 1.0),
            ]
        )
    factors = factor.compute(prices)
    result = BacktestEngine(args.top_n).run(prices, factors)
    metrics = PerformanceAnalyzer().analyze(result.daily)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
