"""Simple explicit execution-cost accounting for research backtests.

Costs are charged when the position changes and expressed in basis points.
The model intentionally excludes market impact, liquidity constraints, taxes,
borrow fees, and order-book execution.
"""

import pandas as pd


def cost_adjusted_returns(
    prices: pd.Series,
    positions: pd.Series,
    fee_bps: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Compute gross/net period returns using previous-period positions.

    positions[t] is the target position set after observing bar t.
    P&L on bar t uses positions[t-1], preventing same-bar signal execution.
    Turnover is abs(position[t-1] - position[t-2]) charged at the
    start of bar t; the final target change has no realized cost
    unless another price bar follows. The initial transition from
    flat to the first target is charged on the first investable bar.
    """
    if prices.empty or len(prices) < 2:
        raise ValueError("at least two prices are required")
    if not prices.index.equals(positions.index):
        raise ValueError("prices and positions must have identical indices")
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("prices must be positive and nonmissing")
    if positions.isna().any() or not positions.between(-1.0, 1.0).all():
        raise ValueError("positions must be within [-1, 1] and nonmissing")
    if any(not 0 <= x < float("inf") for x in (fee_bps, spread_bps, slippage_bps)):
        raise ValueError("cost basis points must be finite and nonnegative")

    held = positions.shift(1).fillna(0.0).astype(float)
    previous_held = held.shift(1).fillna(0.0)
    turnover = (held - previous_held).abs()
    market_returns = prices.pct_change().fillna(0.0)
    gross = market_returns * held
    cost = turnover * (fee_bps + spread_bps / 2.0 + slippage_bps) / 10_000.0
    net = gross - cost
    return pd.DataFrame(
        {"gross_return": gross, "turnover": turnover,
         "execution_cost": cost, "net_return": net},
        index=prices.index,
    )
