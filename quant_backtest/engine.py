from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .execution import ExecutionConfig, execute_order, infer_market


TRADE_COLUMNS = [
    "date",
    "rebalance_date",
    "code",
    "market",
    "side",
    "target_weight",
    "desired_notional",
    "executed_notional",
    "fill_ratio",
    "reference_price",
    "execution_price",
    "volume",
    "commission",
    "tax",
    "slippage",
    "transaction_cost",
    "status",
]


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame
    rebalances: pd.DataFrame
    trades: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=TRADE_COLUMNS)
    )


class BacktestEngine:
    def __init__(
        self,
        top_n: int = 5,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        self.top_n = top_n
        self.execution_config = execution_config

    def run(self, prices: pd.DataFrame, factors: pd.DataFrame) -> BacktestResult:
        self._validate(prices, factors)
        adjusted = prices.assign(
            date=pd.to_datetime(prices["date"]),
            adjusted_close=prices["close"] * prices["adj_factor"],
        )
        adjusted = adjusted.sort_values(["date", "code"]).drop_duplicates(
            ["date", "code"], keep="last"
        )
        close = (
            adjusted.pivot_table(
                index="date", columns="code", values="adjusted_close", aggfunc="last"
            )
            .sort_index()
            .sort_index(axis=1)
        )
        returns = close.pct_change(fill_method=None)
        signals = (
            factors.assign(date=pd.to_datetime(factors["date"]))
            .pivot_table(
                index="date", columns="code", values="factor_value", aggfunc="last"
            )
            .reindex(index=close.index, columns=close.columns)
        )

        targets, rebalances = self._build_targets(close, signals)
        if self.execution_config is None or self.execution_config.is_frictionless:
            return self._run_frictionless(close, targets, rebalances)
        return self._run_with_execution(adjusted, close, targets, rebalances)

    def _build_targets(
        self, close: pd.DataFrame, signals: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        targets = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        rebalance_dates = (
            pd.Series(close.index, index=close.index)
            .groupby(close.index.to_period("M"))
            .first()
        )
        rebalance_records: list[dict] = []
        for date in rebalance_dates:
            targets.loc[date] = 0.0
            ranked = signals.loc[date].dropna().sort_values(ascending=False)
            selected = ranked.head(self.top_n).index
            if len(selected):
                weight = 1.0 / len(selected)
                targets.loc[date, selected] = weight
                rebalance_records.extend(
                    {
                        "date": date,
                        "code": code,
                        "factor_value": float(ranked.loc[code]),
                        "target_weight": weight,
                    }
                    for code in selected
                )
        rebalances = pd.DataFrame(
            rebalance_records,
            columns=["date", "code", "factor_value", "target_weight"],
        )
        return targets, rebalances

    @staticmethod
    def _run_frictionless(
        close: pd.DataFrame,
        targets: pd.DataFrame,
        rebalances: pd.DataFrame,
    ) -> BacktestResult:
        returns = close.pct_change(fill_method=None)
        target_positions = targets.ffill().fillna(0.0)
        # Targets are formed at each rebalance close and take effect next session.
        effective_positions = target_positions.shift(1).fillna(0.0)
        portfolio_returns = (effective_positions * returns.fillna(0.0)).sum(axis=1)
        net_value = (1.0 + portfolio_returns).cumprod()
        daily = pd.DataFrame(
            {
                "date": close.index,
                "return": portfolio_returns.to_numpy(),
                "net_value": net_value.to_numpy(),
                "turnover": 0.0,
                "commission": 0.0,
                "tax": 0.0,
                "slippage": 0.0,
                "transaction_cost": 0.0,
                "cash": (
                    (1.0 - effective_positions.sum(axis=1)).clip(lower=0) * net_value
                ).to_numpy(),
            }
        )
        positions = (
            effective_positions.rename_axis(index="date", columns="code")
            .stack()
            .rename("weight")
            .reset_index()
        )
        return BacktestResult(daily, positions, rebalances)

    def _run_with_execution(
        self,
        adjusted: pd.DataFrame,
        close: pd.DataFrame,
        targets: pd.DataFrame,
        rebalances: pd.DataFrame,
    ) -> BacktestResult:
        config = self.execution_config
        assert config is not None
        codes = close.columns.tolist()
        bars = adjusted.set_index(["date", "code"])
        market_by_code = (
            adjusted.dropna(subset=["code"])
            .drop_duplicates("code", keep="last")
            .set_index("code")
            .get("market", pd.Series(dtype=object))
            .to_dict()
        )
        execution_events: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Series]] = {}
        for rebalance_date in targets.dropna(how="all").index:
            location = close.index.get_loc(rebalance_date)
            if location + 1 < len(close.index):
                execution_events[close.index[location + 1]] = (
                    rebalance_date,
                    targets.loc[rebalance_date].fillna(0.0),
                )

        holdings = {code: 0.0 for code in codes}
        last_prices: dict[str, float] = {}
        cash = config.initial_cash
        previous_nav = config.initial_cash
        daily_records: list[dict] = []
        position_records: list[dict] = []
        trade_records: list[dict] = []

        for date in close.index:
            for code in codes:
                current_price = close.at[date, code]
                if pd.notna(current_price) and current_price > 0:
                    previous_price = last_prices.get(code)
                    if previous_price and holdings[code]:
                        holdings[code] *= float(current_price) / previous_price
                    last_prices[code] = float(current_price)

            nav_before_trade = cash + sum(holdings.values())
            day_costs = {
                "commission": 0.0,
                "tax": 0.0,
                "slippage": 0.0,
                "transaction_cost": 0.0,
            }
            gross_traded = 0.0
            event = execution_events.get(date)
            if event is not None:
                rebalance_date, target = event
                desired_orders = {
                    code: float(target.get(code, 0.0)) * nav_before_trade
                    - holdings[code]
                    for code in codes
                }
                ordered_codes = sorted(
                    codes, key=lambda code: desired_orders[code]
                )
                for code in ordered_codes:
                    desired = desired_orders[code]
                    if abs(desired) <= 1e-10:
                        continue
                    bar = (
                        bars.loc[(date, code)]
                        if (date, code) in bars.index
                        else pd.Series(dtype=float)
                    )
                    price = bar.get("adjusted_close", np.nan)
                    volume = bar.get("volume", np.nan)
                    market = infer_market(code, bar.get("market", market_by_code.get(code)))
                    fill = execute_order(
                        desired_notional=desired,
                        holding_notional=holdings[code],
                        available_cash=cash,
                        reference_price=price,
                        volume=volume,
                        market=market,
                        config=config,
                    )
                    holdings[code] += fill.executed_notional
                    cash -= fill.executed_notional + fill.transaction_cost
                    gross_traded += abs(fill.executed_notional)
                    for key in day_costs:
                        day_costs[key] += getattr(fill, key)
                    trade_records.append(
                        {
                            "date": date,
                            "rebalance_date": rebalance_date,
                            "code": code,
                            "market": market,
                            "side": fill.side,
                            "target_weight": float(target.get(code, 0.0)),
                            "desired_notional": fill.desired_notional,
                            "executed_notional": fill.executed_notional,
                            "fill_ratio": fill.fill_ratio,
                            "reference_price": fill.reference_price,
                            "execution_price": fill.execution_price,
                            "volume": volume,
                            "commission": fill.commission,
                            "tax": fill.tax,
                            "slippage": fill.slippage,
                            "transaction_cost": fill.transaction_cost,
                            "status": fill.status,
                        }
                    )

            nav = cash + sum(holdings.values())
            daily_return = nav / previous_nav - 1.0
            daily_records.append(
                {
                    "date": date,
                    "return": daily_return,
                    "net_value": nav / config.initial_cash,
                    "turnover": (
                        gross_traded / nav_before_trade if nav_before_trade > 0 else 0.0
                    ),
                    **day_costs,
                    "cash": cash,
                }
            )
            for code in codes:
                position_records.append(
                    {
                        "date": date,
                        "code": code,
                        "weight": holdings[code] / nav if nav > 0 else 0.0,
                    }
                )
            previous_nav = nav

        return BacktestResult(
            daily=pd.DataFrame(daily_records),
            positions=pd.DataFrame(position_records),
            rebalances=rebalances,
            trades=pd.DataFrame(trade_records, columns=TRADE_COLUMNS),
        )

    def _validate(self, prices: pd.DataFrame, factors: pd.DataFrame) -> None:
        price_missing = {"date", "code", "close", "adj_factor"} - set(prices.columns)
        factor_missing = {"date", "code", "factor_value"} - set(factors.columns)
        if price_missing:
            raise ValueError(f"prices missing columns: {sorted(price_missing)}")
        if factor_missing:
            raise ValueError(f"factors missing columns: {sorted(factor_missing)}")
        if prices.empty:
            raise ValueError("prices must not be empty")
        if factors.empty:
            raise ValueError("factors must not be empty")
        if (
            self.execution_config is not None
            and self.execution_config.max_volume_participation is not None
            and "volume" not in prices.columns
        ):
            raise ValueError("prices missing volume required by execution constraints")
