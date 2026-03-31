from mesa.datacollection import DataCollector
from mesa.model import Model
from rich import print as rprint

from .agents import LLMTraderAgent, RuleBasedTraderAgent
from mesa_llm.reasoning.reasoning import Reasoning

NEWS_CYCLE: list[str] = [
    "Q3 earnings beat expectations: major sectors report 12% revenue growth above analyst forecasts.",
    "Central bank signals potential rate pause - borrowing cost pressures may ease next quarter.",
    "Inflation prints below expectations for the third consecutive month - soft-landing narrative gains.",
    "Geopolitical tensions escalate in a key oil-producing region; supply disruption fears mount.",
    "Regulatory body launches sweeping antitrust probe into leading technology companies.",
    "Consumer confidence falls to 20-month low; household spending contracts for second consecutive month.",
    "GDP data: economy enters technical recession after two consecutive quarters of contraction.",
    "Credit default swap spreads widen sharply - institutional risk appetite retreats to 2-year low.",
    "Historic trade agreement signed: tariffs cut significantly between two of the largest global economies.",
    "Central bank holds rates steady; forward guidance hints at cuts within six months if data cooperates.",
    "Jobs report: unemployment at 3.1%, wage growth at 4.2% year-over-year - labour market remains robust.",
    "Breakthrough AI productivity study: 18% efficiency gains across manufacturing; tech stocks surge pre-market.",
]

INITIAL_PRICE: float = 100.0
N_LLM_TRADERS: int = 5
N_RULE_TRADERS: int = 5
INITIAL_CASH: float = 10_000.0
INITIAL_SHARES: int = 50

# Each net share of aggregate demand moves price by this fraction of current price.
# At 5 LLM + 5 rule traders each trading up to 10-50 shares, a fully coordinated
# buy will shift price by ~2–4%, which is realistic for a thin market.
PRICE_IMPACT: float = 0.003


class FinancialMarketModel(Model):
    """
    Ten traders (5 LLM, 5 rule-based) all trade a single risky asset.  Each
    simulation step:

      1. A new news headline from NEWS_CYCLE is announced.
      2. All traders act simultaneously (shuffle_do).
         - LLM traders call buy_shares / sell_shares / hold_position based on
           LLM reasoning over the headline and price history.
         - Rule-based traders apply a simple momentum rule.
      3. The market price updates based on aggregate net orders.
      4. Portfolio values are recalculated for all agents.

    DataCollector tracks price, volume, net orders, and average portfolio values
    for both groups, enabling direct LLM vs rule-based performance comparison.

    Args:
        reasoning : Reasoning strategy class (e.g. ReActReasoning, ReWOOReasoning).
        llm_model : LiteLLM model string, e.g. ``"gemini/gemini-2.0-flash"``.
        rng       : Random seed for reproducibility.
    """

    def __init__(
        self,
        reasoning: type[Reasoning],
        llm_model: str = "gemini/gemini-2.0-flash",
        rng: int = 42,
    ):
        super().__init__(rng=rng)

        self.price: float = INITIAL_PRICE
        self.price_history: list[float] = [INITIAL_PRICE]
        self.current_news: str = NEWS_CYCLE[0]
        self.net_orders: int = 0    # reset each step; positive = net buy
        self.total_volume: int = 0  # reset each step; cumulative shares traded

        for i in range(N_LLM_TRADERS):
            LLMTraderAgent(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                trader_id=i,
                initial_cash=INITIAL_CASH,
                initial_shares=INITIAL_SHARES,
            )

        for i in range(N_RULE_TRADERS):
            RuleBasedTraderAgent(
                model=self,
                trader_id=N_LLM_TRADERS + i,
                initial_cash=INITIAL_CASH,
                initial_shares=INITIAL_SHARES,
            )

        self.datacollector = DataCollector(
            model_reporters={
                "Price": "price",
                "NetOrders": "net_orders",
                "TotalVolume": "total_volume",
                "LLMAvgPortfolio": lambda m: m._avg_portfolio(use_llm=True),
                "RuleAvgPortfolio": lambda m: m._avg_portfolio(use_llm=False),
            },
            agent_reporters={
                "PortfolioValue": "portfolio_value",
                "Cash": "cash",
                "SharesHeld": "shares",
                "TradesMade": "trades_made",
            },
        )


    def _avg_portfolio(self, use_llm: bool) -> float:
        cls = LLMTraderAgent if use_llm else RuleBasedTraderAgent
        group = [a for a in self.agents if isinstance(a, cls)]
        if not group:
            return 0.0
        return sum(a.portfolio_value for a in group) / len(group)

    def _update_price(self) -> None:
        change = self.net_orders * PRICE_IMPACT * self.price
        self.price = max(1.0, round(self.price + change, 4))
        self.price_history.append(self.price)


    def step(self) -> None:
        self.datacollector.collect(self)

        # Rotate news headline
        self.current_news = NEWS_CYCLE[self.steps % len(NEWS_CYCLE)]
        self.net_orders = 0
        self.total_volume = 0

        rprint(f"\n[bold cyan] Market Step {self.steps} [/bold cyan]")
        rprint(f"[yellow]NEWS: {self.current_news}[/yellow]")
        rprint(f"[white]Opening price: ${self.price:.4f}[/white]")

        self.agents.shuffle_do("step")
        self._update_price()

        # Refresh portfolio values after price move
        for a in self.agents:
            a.portfolio_value = a.cash + a.shares * self.price

        llm_avg = self._avg_portfolio(use_llm=True)
        rule_avg = self._avg_portfolio(use_llm=False)
        rprint(
            f"[bold green]  Close: ${self.price:.4f}  "
            f"net_orders={self.net_orders:+d}  vol={self.total_volume}  "
            f"LLM avg=${llm_avg:,.0f}  Rule avg=${rule_avg:,.0f}[/bold green]"
        )


if __name__ == "__main__":
    
    from mesa_llm.reasoning.react import ReActReasoning

    m = FinancialMarketModel(
        reasoning=ReActReasoning,
        llm_model="gemini/gemini-2.0-flash",
        rng=42,
    )
    for _ in range(12):
        m.step()

    rprint("\n[bold magenta]=== Final Results ===[/bold magenta]")
    rprint(f"Final price : ${m.price:.4f}  (started at ${INITIAL_PRICE:.2f})")
    rprint(f"LLM avg portfolio  : ${m._avg_portfolio(use_llm=True):,.2f}")
    rprint(f"Rule avg portfolio : ${m._avg_portfolio(use_llm=False):,.2f}")
