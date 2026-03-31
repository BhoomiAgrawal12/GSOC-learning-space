import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import solara
from dotenv import load_dotenv
from mesa.visualization import SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter

from financial_market.agents import LLMTraderAgent, RuleBasedTraderAgent
from financial_market.model import INITIAL_PRICE, FinancialMarketModel
from mesa_llm.reasoning.react import ReActReasoning

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
logging.getLogger("pydantic").setLevel(logging.ERROR)

load_dotenv()

model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "llm_model": {
        "type": "Select",
        "value": "gemini/gemini-2.0-flash",
        "values": [
            "gemini/gemini-2.0-flash",
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-haiku-4-5-20251001",
            "ollama/llama3.2",
        ],
        "label": "LLM Model",
    },
    "reasoning": ReActReasoning,
}

model = FinancialMarketModel(
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"]["value"],
    rng=model_params["rng"]["value"],
)

def PriceHistoryChart(model):
    """Line chart of asset price across all simulation steps."""
    update_counter.get()

    fig, ax = plt.subplots(figsize=(9, 3.5))

    hist = model.price_history
    if len(hist) < 2:
        ax.set_title("No price data yet — click Step to begin")
        ax.axhline(y=INITIAL_PRICE, color="#95a5a6", linestyle="--", linewidth=1)
        ax.set_ylim(INITIAL_PRICE * 0.8, INITIAL_PRICE * 1.2)
        return solara.FigureMatplotlib(fig)

    steps = list(range(len(hist)))
    ax.plot(steps, hist, color="#2980b9", linewidth=2, marker="o", markersize=3)
    ax.axhline(y=INITIAL_PRICE, color="#95a5a6", linestyle="--",
               linewidth=1, label=f"Start ${INITIAL_PRICE:.0f}")
    ax.fill_between(steps, INITIAL_PRICE, hist,
                    where=[p >= INITIAL_PRICE for p in hist],
                    alpha=0.15, color="#27ae60", label="Above start")
    ax.fill_between(steps, INITIAL_PRICE, hist,
                    where=[p < INITIAL_PRICE for p in hist],
                    alpha=0.15, color="#e74c3c", label="Below start")

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Price ($)", fontsize=10)
    ax.set_title(f"Asset Price — Step {model.steps}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    return solara.FigureMatplotlib(fig)


def PortfolioComparisonChart(model):
    """Line chart comparing LLM vs rule-based average portfolio value over time."""
    update_counter.get()

    fig, ax = plt.subplots(figsize=(9, 3.5))

    try:
        df = model.datacollector.get_model_vars_dataframe()
    except Exception:
        ax.set_title("No data yet — click Step")
        return solara.FigureMatplotlib(fig)

    if df.empty or "LLMAvgPortfolio" not in df.columns:
        ax.set_title("Run a few steps to see portfolio comparison")
        return solara.FigureMatplotlib(fig)

    ax.plot(df.index, df["LLMAvgPortfolio"], color="#8e44ad", linewidth=2,
            marker="o", markersize=4, label="LLM traders (avg)")
    ax.plot(df.index, df["RuleAvgPortfolio"], color="#e67e22", linewidth=2,
            marker="s", markersize=4, linestyle="--", label="Rule-based (avg)")
    ax.axhline(y=df["LLMAvgPortfolio"].iloc[0] if not df.empty else 11_000,
               color="#95a5a6", linestyle=":", linewidth=1, label="Initial value")

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax.set_title("LLM vs Rule-Based Trader Performance", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return solara.FigureMatplotlib(fig)


def AgentPortfolioBar(model):
    """Bar chart of each trader's current portfolio value."""
    update_counter.get()

    llm_agents = [a for a in model.agents if isinstance(a, LLMTraderAgent)]
    rule_agents = [a for a in model.agents if isinstance(a, RuleBasedTraderAgent)]

    fig, ax = plt.subplots(figsize=(9, 3.5))

    if not llm_agents and not rule_agents:
        ax.set_title("No agents yet")
        return solara.FigureMatplotlib(fig)

    labels = (
        [f"LLM-{a.trader_id}" for a in llm_agents]
        + [f"Rule-{a.trader_id - len(llm_agents)}" for a in rule_agents]
    )
    values = [a.portfolio_value for a in llm_agents + rule_agents]
    colors = ["#8e44ad"] * len(llm_agents) + ["#e67e22"] * len(rule_agents)

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)

    initial_val = 10_000.0 + 50 * INITIAL_PRICE
    ax.axhline(y=initial_val, color="#95a5a6", linestyle="--",
               linewidth=1.2, label=f"Initial ${initial_val:,.0f}")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"${val:,.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax.set_title(f"Individual Portfolios — Step {model.steps}  (purple=LLM, orange=rule)", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return solara.FigureMatplotlib(fig)


@solara.component
def MarketStatusPanel(model):
    """Summary table showing each trader's cash, shares, last action, and portfolio."""
    update_counter.get()

    llm_agents = [a for a in model.agents if isinstance(a, LLMTraderAgent)]
    rule_agents = [a for a in model.agents if isinstance(a, RuleBasedTraderAgent)]

    latest_price = model.price
    solara.Text(
        f"Step {model.steps}  ·  Price ${latest_price:.4f}  ·  "
        f"Net orders {model.net_orders:+d}  ·  Volume {model.total_volume}  ·  "
        f"News: {model.current_news[:60]}…"
    )

    rows = []
    for a in llm_agents:
        rows.append({
            "Agent": f"LLM-{a.trader_id}",
            "Type": "LLM",
            "Cash ($)": f"{a.cash:,.2f}",
            "Shares": a.shares,
            "Portfolio ($)": f"{a.portfolio_value:,.2f}",
            "Last Action": a.last_action,
            "Trades": a.trades_made,
        })
    for a in rule_agents:
        rows.append({
            "Agent": f"Rule-{a.trader_id - len(llm_agents)}",
            "Type": "Rule",
            "Cash ($)": f"{a.cash:,.2f}",
            "Shares": a.shares,
            "Portfolio ($)": f"{a.portfolio_value:,.2f}",
            "Last Action": a.last_action,
            "Trades": a.trades_made,
        })

    solara.DataFrame(pd.DataFrame(rows))


PricePlot = make_plot_component("Price")
VolumePlot = make_plot_component("TotalVolume")
NetOrdersPlot = make_plot_component("NetOrders")


page = SolaraViz(
    model,
    renderer=None,
    components=[
        PriceHistoryChart,
        PortfolioComparisonChart,
        AgentPortfolioBar,
        MarketStatusPanel,
        VolumePlot,
        NetOrdersPlot,
    ],
    model_params=model_params,
    name="Financial Market — Mesa-LLM",
)
