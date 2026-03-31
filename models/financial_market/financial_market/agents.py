"""Agent definitions for the Financial Market example.

Two agent types compete in the same market:
  - LLMTraderAgent   : uses STLTMemory + configurable reasoning to read news
                       headlines and price history, then calls buy/sell/hold tools.
  - RuleBasedTraderAgent : deterministic momentum trader (baseline for comparison).

Both agents expose the same attributes (cash, shares, portfolio_value, last_action)
so the DataCollector and visualisation can treat them uniformly.
"""

from mesa.agent import Agent
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

# One shared ToolManager for all LLMTraderAgents.
# buy_shares / sell_shares / hold_position are registered here by tools.py at import time.
# Global inbuilt tools (speak_to, …) are automatically copied in at construction time.
trader_tool_manager = ToolManager()


def _get_recent_messages(agent, max_messages: int = 4) -> str:
    messages = []
    memory_source = None
    if hasattr(agent.memory, "short_term_memory"):
        memory_source = agent.memory.short_term_memory
    elif hasattr(agent.memory, "memory_entries"):
        memory_source = agent.memory.memory_entries

    if memory_source:
        for entry in reversed(list(memory_source)[-max_messages * 2:]):
            if len(messages) >= max_messages:
                break
            if isinstance(entry.content, dict) and "message" in entry.content:
                msg = entry.content.get("message", "")
                messages.append(f"  • {msg}")

    messages.reverse()
    return "\n".join(messages) if messages else "  No prior messages."


class LLMTraderAgent(LLMAgent):
    """Sentiment-driven trader that uses LLM multi-step reasoning to trade.

    Each step the agent receives the current price, recent price history,
    a market news headline, and its own portfolio state.  It then calls
    exactly one of: buy_shares, sell_shares, or hold_position.

    Mesa-LLM features demonstrated:
      - STLTMemory  : consolidates past trade decisions into long-term memory
                      so later steps reflect on earlier performance.
      - Configurable reasoning (ReActReasoning / ReWOOReasoning): multi-step
        planning allows the agent to think "what happens next round?" before acting.
      - vision=None : no spatial grid - context delivered entirely via prompt.
      - Custom tools : buy_shares, sell_shares, hold_position.

    Attributes:
        trader_id     : human-readable index (0 … N_LLM_TRADERS-1).
        cash          : USD cash balance.
        shares        : number of shares currently held.
        portfolio_value: cash + shares current_price (updated each step end).
        trades_made   : total number of non-hold actions taken.
        last_action   : string summary of last action ("buy_10", "sell_5", "hold").
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model: str,
        trader_id: int,
        initial_cash: float,
        initial_shares: int,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=(
                "You are a sophisticated quantitative trader with deep expertise in "
                "market microstructure and sentiment analysis.  Each round you see the "
                "current asset price, recent price history, and a single market news "
                "headline.  Your goal is to maximise portfolio value over the full "
                "simulation.  Think ahead: if news is positive, others will likely buy "
                "next round too, pushing prices higher - buy now.  If negative, sell "
                "before the crowd.  Be decisive: hold only when genuinely uncertain."
            ),
            vision=None,
            internal_state=[
                f"trader_id:{trader_id}",
                "strategy:sentiment_momentum",
                "agent_type:llm_trader",
            ],
        )
        self.tool_manager = trader_tool_manager

        self.trader_id = trader_id
        self.cash = float(initial_cash)
        self.shares = int(initial_shares)
        self.portfolio_value = float(initial_cash) + int(initial_shares) * model.price
        self.trades_made: int = 0
        self.last_action: str = "hold"

    def step(self):
        observation = self.generate_obs()

        # Build price-history string (last 6 data points)
        hist = self.model.price_history[-6:]
        hist_str = "  ->  ".join(f"${p:.2f}" for p in hist)

        if len(hist) >= 2:
            delta = hist[-1] - hist[-2]
            pct = delta / hist[-2] * 100
            trend = f"{'▲' if delta > 0 else '▼' if delta < 0 else '—'} {pct:+.2f}% last step"
        else:
            trend = "insufficient history"

        max_buy = int(self.cash // self.model.price) if self.model.price > 0 else 0
        recent_msgs = _get_recent_messages(self)

        prompt = (
            f"MARKET STEP {self.model.steps}\n\n"
            f"MARKET STATE:\n"
            f"  Current price  : ${self.model.price:.4f}\n"
            f"  Price history  : {hist_str}\n"
            f"  Price trend    : {trend}\n"
            f"  Today's news   : {self.model.current_news}\n\n"
            f"YOUR PORTFOLIO:\n"
            f"  Cash available : ${self.cash:.2f}\n"
            f"  Shares held    : {self.shares}\n"
            f"  Portfolio value: ${self.portfolio_value:.2f}\n"
            f"  Max you can buy : {max_buy} shares\n"
            f"  Max you can sell: {self.shares} shares\n\n"
            f"RECENT MEMORY:\n{recent_msgs}\n\n"
            f"INSTRUCTIONS:\n"
            f"Analyse the news headline and price momentum.  Choose exactly ONE action:\n"
            f"  buy_shares(quantity, rationale)  - purchase shares at current price\n"
            f"  sell_shares(quantity, rationale) - sell shares at current price\n"
            f"  hold_position(rationale)         - take no trade this round\n\n"
            f"Hard constraints:\n"
            f"  • quantity must be a positive integer\n"
            f"  • cannot buy more than {max_buy} shares (cash limit)\n"
            f"  • cannot sell more than {self.shares} shares (holdings limit)\n\n"
            f"Think multi-step: what will the price do next round given this news and "
            f"how other traders are likely to react?"
        )

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["buy_shares", "sell_shares", "hold_position"],
        )
        self.apply_plan(plan)


class RuleBasedTraderAgent(Agent):
    """Deterministic momentum trader used as a rule-based baseline.

    Strategy (simple single-step momentum):
      - If price rose last step  -> buy up to TRADE_SIZE shares (if affordable).
      - If price fell last step  -> sell up to TRADE_SIZE shares (if held).
      - Otherwise               -> hold.

    Exposes the same interface as LLMTraderAgent so visualisation treats both uniformly.

    Attributes:
        trader_id     : index (N_LLM_TRADERS … N_LLM_TRADERS+N_RULE_TRADERS-1).
        cash          : USD cash balance.
        shares        : number of shares currently held.
        portfolio_value: cash + shares price (updated end of each model step).
        trades_made   : total non-hold actions.
        last_action   : string summary of last action.
    """

    TRADE_SIZE = 10

    def __init__(self, model, trader_id: int, initial_cash: float, initial_shares: int):
        super().__init__(model=model)
        self.trader_id = trader_id
        self.cash = float(initial_cash)
        self.shares = int(initial_shares)
        self.portfolio_value = float(initial_cash) + int(initial_shares) * model.price
        self.trades_made: int = 0
        self.last_action: str = "hold"

    def step(self):
        hist = self.model.price_history
        if len(hist) < 2:
            self.last_action = "hold"
            return

        price_rose = hist[-1] > hist[-2]
        price_fell = hist[-1] < hist[-2]
        price = self.model.price

        if price_rose:
            qty = min(self.TRADE_SIZE, int(self.cash // price))
            if qty > 0:
                cost = qty * price
                self.cash -= cost
                self.shares += qty
                self.model.net_orders += qty
                self.model.total_volume += qty
                self.trades_made += 1
                self.last_action = f"buy_{qty}"
                return

        if price_fell:
            qty = min(self.TRADE_SIZE, self.shares)
            if qty > 0:
                proceeds = qty * price
                self.shares -= qty
                self.cash += proceeds
                self.model.net_orders -= qty
                self.model.total_volume += qty
                self.trades_made += 1
                self.last_action = f"sell_{qty}"
                return

        self.last_action = "hold"
