from __future__ import annotations

import argparse
import json

import pandas as pd

from .data import (
    AkShareProvider,
    Csi300ConstituentProvider,
    MarketDataService,
    MongoMarketDataRepository,
    Sp500ConstituentProvider,
    YFinanceProvider,
)
from .engine import BacktestEngine
from .execution import ExecutionConfig, MarketFeeSchedule, infer_market
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

    sync = subparsers.add_parser(
        "sync-pools", help="sync current CSI 300 and S&P 500 constituents"
    )
    sync.add_argument(
        "--pool", choices=["csi300", "sp500", "all"], default="all"
    )

    load = subparsers.add_parser("load-data", help="load or update sample market data")
    load_source = load.add_mutually_exclusive_group()
    load_source.add_argument(
        "--pool", choices=["sample", "csi300", "sp500"], default="sample"
    )
    load_source.add_argument(
        "--codes",
        nargs="+",
        help="load explicit codes, for example SPY.US 510300.SH",
    )
    load.add_argument("--start", default="2015-01-01")
    load.add_argument("--end")
    load.add_argument("--full", action="store_true", help="disable incremental loading")
    load.add_argument("--batch-offset", type=int, default=0)
    load.add_argument("--batch-size", type=int)
    load.add_argument("--request-delay", type=float, default=0.0)

    backtest = subparsers.add_parser("backtest", help="run the factor sample")
    backtest.add_argument(
        "--pool", choices=["sample", "csi300", "sp500"], default="sample"
    )
    backtest.add_argument("--start", required=True)
    backtest.add_argument("--end", required=True)
    backtest.add_argument(
        "--factor", choices=["momentum", "composite"], default="composite"
    )
    backtest.add_argument("--lookback", type=int, default=20)
    backtest.add_argument("--reversal-lookback", type=int, default=5)
    backtest.add_argument("--volatility-window", type=int, default=20)
    backtest.add_argument("--top-n", type=int, default=5)
    backtest.add_argument("--initial-cash", type=float, default=1_000_000)
    backtest.add_argument("--cn-commission-bps", type=float, default=3)
    backtest.add_argument("--cn-sell-tax-bps", type=float, default=5)
    backtest.add_argument("--us-commission-bps", type=float, default=1)
    backtest.add_argument("--slippage-bps", type=float, default=5)
    backtest.add_argument("--max-volume-pct", type=float, default=10)
    backtest.add_argument(
        "--frictionless", action="store_true", help="disable all execution constraints"
    )
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
    if args.command == "sync-pools":
        providers = {
            "csi300": Csi300ConstituentProvider(),
            "sp500": Sp500ConstituentProvider(),
        }
        selected = providers if args.pool == "all" else {args.pool: providers[args.pool]}
        results = []
        for provider in selected.values():
            result = service.sync_pool(provider)
            result["snapshot_date"] = str(result["snapshot_date"].date())
            results.append(result)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(
            "WARNING: current constituent snapshots are not historical membership "
            "and must not be used for unbiased backtests before their snapshot dates."
        )
        return

    if args.command == "load-data":
        if args.codes:
            codes = [
                code.strip().upper()
                for value in args.codes
                for code in value.split(",")
                if code.strip()
            ]
            codes_by_market: dict[str, list[str]] = {}
            for code in codes:
                codes_by_market.setdefault(infer_market(code), []).append(code)
            counts = service.load(
                codes_by_market,
                start=args.start,
                end=args.end,
                incremental=not args.full,
                request_delay=args.request_delay,
            )
        elif args.pool == "sample":
            counts = service.load(
                SAMPLE_CODES,
                start=args.start,
                end=args.end,
                incremental=not args.full,
                request_delay=args.request_delay,
            )
        else:
            counts = service.load_pool(
                args.pool,
                start=args.start,
                end=args.end,
                incremental=not args.full,
                batch_offset=args.batch_offset,
                batch_size=args.batch_size,
                request_delay=args.request_delay,
            )
        print(json.dumps(counts, indent=2, ensure_ascii=False))
        return

    if args.pool != "sample":
        snapshot = repository.latest_pool_snapshot(args.pool)
        if snapshot is None:
            raise ValueError(f"pool not found; run sync-pools first: {args.pool}")
        if pd.Timestamp(args.end) < snapshot:
            raise ValueError(
                f"{args.pool} is a current-only snapshot from {snapshot.date()}; "
                "historical membership is unavailable"
            )
        print(
            f"WARNING: {args.pool} uses current constituents from {snapshot.date()}, "
            "so results are subject to survivorship bias."
        )
    prices = service.read(
        pool_id=args.pool,
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
    execution_config = None
    if not args.frictionless:
        execution_config = ExecutionConfig(
            initial_cash=args.initial_cash,
            cn_fees=MarketFeeSchedule(
                commission_rate=args.cn_commission_bps / 10_000,
                sell_tax_rate=args.cn_sell_tax_bps / 10_000,
            ),
            us_fees=MarketFeeSchedule(
                commission_rate=args.us_commission_bps / 10_000
            ),
            slippage_bps=args.slippage_bps,
            max_volume_participation=args.max_volume_pct / 100,
        )
    result = BacktestEngine(
        args.top_n, execution_config=execution_config
    ).run(prices, factors)
    metrics = PerformanceAnalyzer().analyze(result.daily)
    metrics.update(
        {
            "total_turnover": float(result.daily["turnover"].sum()),
            "total_transaction_cost": float(
                result.daily["transaction_cost"].sum()
            ),
        }
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
