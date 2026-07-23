from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np


@dataclass(frozen=True)
class MarketFeeSchedule:
    commission_rate: float = 0.0
    sell_tax_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_rate < 0 or self.sell_tax_rate < 0:
            raise ValueError("fee rates must be non-negative")


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 1_000_000.0
    cn_fees: MarketFeeSchedule = MarketFeeSchedule()
    us_fees: MarketFeeSchedule = MarketFeeSchedule()
    slippage_bps: float = 0.0
    max_volume_participation: float | None = None

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if self.max_volume_participation is not None and not (
            0 < self.max_volume_participation <= 1
        ):
            raise ValueError("max_volume_participation must be in (0, 1]")

    @property
    def is_frictionless(self) -> bool:
        return (
            self.cn_fees == MarketFeeSchedule()
            and self.us_fees == MarketFeeSchedule()
            and self.slippage_bps == 0
            and self.max_volume_participation is None
        )

    def fees_for(self, market: str) -> MarketFeeSchedule:
        if market == "CN":
            return self.cn_fees
        if market == "US":
            return self.us_fees
        raise ValueError(f"unsupported market: {market}")


@dataclass(frozen=True)
class ExecutionFill:
    side: str
    desired_notional: float
    executed_notional: float
    reference_price: float
    execution_price: float
    commission: float
    tax: float
    slippage: float
    transaction_cost: float
    fill_ratio: float
    status: str


def execute_order(
    *,
    desired_notional: float,
    holding_notional: float,
    available_cash: float,
    reference_price: float | None,
    volume: float | None,
    market: str,
    config: ExecutionConfig,
) -> ExecutionFill:
    side = "BUY" if desired_notional > 0 else "SELL"
    price = float(reference_price) if reference_price is not None else np.nan
    if not np.isfinite(price) or price <= 0:
        return _empty_fill(side, desired_notional, price, "NO_PRICE")

    finite_volume = volume is not None and np.isfinite(volume)
    if finite_volume and float(volume) <= 0:
        return _empty_fill(side, desired_notional, price, "SUSPENDED")
    if config.max_volume_participation is not None and not finite_volume:
        return _empty_fill(side, desired_notional, price, "NO_VOLUME")

    requested = abs(float(desired_notional))
    volume_cap = (
        float(volume) * price * config.max_volume_participation
        if config.max_volume_participation is not None
        else inf
    )
    fees = config.fees_for(market)
    slippage_rate = config.slippage_bps / 10_000

    status = "FILLED"
    executable = min(requested, volume_cap)
    if executable < requested:
        status = "VOLUME_LIMITED"

    if side == "SELL":
        holding_cap = max(float(holding_notional), 0.0)
        if holding_cap < executable:
            executable = holding_cap
            status = "HOLDING_LIMITED"
        tax_rate = fees.sell_tax_rate
    else:
        tax_rate = 0.0
        all_in_rate = fees.commission_rate + slippage_rate
        cash_cap = max(float(available_cash), 0.0) / (1.0 + all_in_rate)
        if cash_cap < executable:
            executable = cash_cap
            status = "CASH_LIMITED"

    commission = executable * fees.commission_rate
    tax = executable * tax_rate
    slippage = executable * slippage_rate
    total_cost = commission + tax + slippage
    signed_notional = executable if side == "BUY" else -executable
    execution_price = price * (
        1.0 + slippage_rate if side == "BUY" else 1.0 - slippage_rate
    )
    if executable <= 1e-12:
        status = "NO_FILL"
    return ExecutionFill(
        side=side,
        desired_notional=float(desired_notional),
        executed_notional=signed_notional,
        reference_price=price,
        execution_price=execution_price,
        commission=commission,
        tax=tax,
        slippage=slippage,
        transaction_cost=total_cost,
        fill_ratio=executable / requested if requested else 1.0,
        status=status,
    )


def infer_market(code: str, market: str | None = None) -> str:
    if market in {"CN", "US"}:
        return market
    return "CN" if code.endswith((".SH", ".SZ", ".BJ")) else "US"


def _empty_fill(
    side: str, desired_notional: float, price: float, status: str
) -> ExecutionFill:
    return ExecutionFill(
        side=side,
        desired_notional=float(desired_notional),
        executed_notional=0.0,
        reference_price=price,
        execution_price=price,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        transaction_cost=0.0,
        fill_ratio=0.0,
        status=status,
    )
