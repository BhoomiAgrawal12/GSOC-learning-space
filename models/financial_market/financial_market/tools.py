"""Custom trading tools for the Financial Market example.

Three tools are registered into `trader_tool_manager` (from agents.py) via the
@tool(tool_manager=...) decorator.  They are NOT added to the global tool registry,
so only LLMTraderAgents can call them.

Each tool validates input conservatively - if the LLM requests an impossible trade
(e.g. buying more than available cash) the quantity is silently clamped rather than
raising an exception, so the simulation never crashes on a bad LLM output.
"""

from typing import TYPE_CHECKING

from .agents import trader_tool_manager
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=trader_tool_manager)
def buy_shares(agent: "LLMAgent", quantity: int, rationale: str) -> str:
    """Purchase shares at the current market price.

    Args:
        quantity : Number of shares to buy (positive integer).
                   Automatically clamped to what the agent can afford.
        rationale: Your reasoning - which news signal or trend drove this decision.
        agent    : Provided automatically by the tool framework.

    Returns:
        Confirmation string with trade details, or an error if impossible.
    """
    # Robust type coercion: LLM may send floats or strings
    try:
        quantity = max(1, int(float(quantity)))
    except (TypeError, ValueError):
        return "buy_shares: invalid quantity - must be a positive integer.  No trade executed."

    price = agent.model.price
    max_affordable = int(agent.cash // price) if price > 0 else 0

    if max_affordable <= 0:
        agent.last_action = "hold"
        return (
            f"buy_shares: insufficient cash (${agent.cash:.2f}) to buy at "
            f"${price:.4f}/share.  Holding instead."
        )

    if quantity > max_affordable:
        quantity = max_affordable  # clamp to affordable amount

    cost = quantity * price
    agent.cash -= cost
    agent.shares += quantity
    agent.model.net_orders += quantity
    agent.model.total_volume += quantity
    agent.trades_made += 1
    agent.last_action = f"buy_{quantity}"

    return (
        f"BUY {quantity} shares @ ${price:.4f}  |  cost ${cost:.2f}  |  "
        f"cash remaining ${agent.cash:.2f}  |  total shares {agent.shares}  |  "
        f"rationale: {rationale}"
    )


@tool(tool_manager=trader_tool_manager)
def sell_shares(agent: "LLMAgent", quantity: int, rationale: str) -> str:
    """Sell shares at the current market price.

    Args:
        quantity : Number of shares to sell (positive integer).
                   Automatically clamped to what the agent holds.
        rationale: Your reasoning - which news signal or trend drove this decision.
        agent    : Provided automatically by the tool framework.

    Returns:
        Confirmation string with trade details, or an error if impossible.
    """
    try:
        quantity = max(1, int(float(quantity)))
    except (TypeError, ValueError):
        return "sell_shares: invalid quantity - must be a positive integer.  No trade executed."

    if agent.shares <= 0:
        agent.last_action = "hold"
        return "sell_shares: no shares to sell.  Holding instead."

    if quantity > agent.shares:
        quantity = agent.shares  # clamp to holdings

    price = agent.model.price
    proceeds = quantity * price
    agent.shares -= quantity
    agent.cash += proceeds
    agent.model.net_orders -= quantity
    agent.model.total_volume += quantity
    agent.trades_made += 1
    agent.last_action = f"sell_{quantity}"

    return (
        f"SELL {quantity} shares @ ${price:.4f}  |  proceeds ${proceeds:.2f}  |  "
        f"cash now ${agent.cash:.2f}  |  shares remaining {agent.shares}  |  "
        f"rationale: {rationale}"
    )


@tool(tool_manager=trader_tool_manager)
def hold_position(agent: "LLMAgent", rationale: str) -> str:
    """Hold the current position - take no trade this round.

    Args:
        rationale: Your reasoning - what uncertainty or risk suggests waiting.
        agent    : Provided automatically by the tool framework.

    Returns:
        Confirmation that no trade was executed this step.
    """
    agent.last_action = "hold"
    return (
        f"HOLD  |  shares {agent.shares}  |  cash ${agent.cash:.2f}  |  "
        f"portfolio ${agent.portfolio_value:.2f}  |  rationale: {rationale}"
    )
