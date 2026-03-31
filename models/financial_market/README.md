# Financial Market - Sentiment-Driven Trading with LLM vs Rule-Based Comparison

An agent-based financial market where **LLM-powered traders** read news headlines,
reason about sentiment and price momentum, and place buy/sell/hold orders alongside
a **deterministic rule-based baseline** - making LLM contribution directly measurable.

## What This Model Demonstrates

| Mesa-LLM Feature | How It's Used |
|---|---|
| `STLTMemory` (default) | Each LLM trader consolidates past trade decisions into long-term memory, enabling reflection like "I bought on positive news three steps ago and price rose - reinforce that pattern." |
| Configurable reasoning (`ReActReasoning` / `ReWOOReasoning`) | Multi-step planning lets agents reason: *"News is positive → others will likely buy next step too → buy now before the crowd pushes price up."* |
| Custom tools | `buy_shares`, `sell_shares`, `hold_position` with quantity-aware execution and clamping. |
| Rule-based baseline | `RuleBasedTraderAgent` applies a simple momentum rule (buy on uptick, sell on downtick). DataCollector shows LLM vs rule-based portfolio value side-by-side. |
| No spatial grid | `vision=None` - all market context delivered through the step prompt. |

## Model Design

```
10 traders total
├── 5 × LLMTraderAgent   - sentiment + momentum analysis via LLM
└── 5 × RuleBasedTraderAgent - single-step momentum rule (buy↑ / sell↓)

Each step:
  1. New news headline announced (12-headline cycle)
  2. All traders act simultaneously (shuffle_do)
  3. Price updated: Δprice = net_orders × 0.003 × current_price
  4. Portfolio values recalculated for all agents
```

**Price mechanics:** aggregate net buy/sell orders shift the price proportionally,
so coordinated LLM sentiment creates visible price swings that the rule-based
traders then mechanically follow - producing emergent momentum cascades.

## Setup

```bash
cd models/financial_market

# Install dependencies
pip install mesa mesa-llm solara python-dotenv matplotlib pandas rich

# Set your LLM API key (e.g. Google Gemini)
echo "GEMINI_API_KEY=your-key-here" > .env

# Run with Solara visualisation
solara run app.py

# Run headless (terminal output only)
python -m financial_market.model
```

## Files

```
financial_market/
├── app.py                          # Solara entry-point
└── financial_market/
    ├── __init__.py                 # registers tools into trader_tool_manager
    ├── agents.py                   # LLMTraderAgent + RuleBasedTraderAgent
    ├── model.py                    # FinancialMarketModel + NEWS_CYCLE
    └── tools.py                    # buy_shares, sell_shares, hold_position
```

## Visualisation Panels

1. **Asset Price History** - line chart with green/red fill above/below starting price
2. **LLM vs Rule-Based Portfolio Comparison** - two lines over time; gap = LLM edge
3. **Individual Portfolio Bar** - current snapshot per agent (purple=LLM, orange=rule)
4. **Market Status Table** - cash, shares, last action, trades made per agent
5. **Volume / Net Orders** - standard make_plot_component series

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `llm_model` | `gemini/gemini-2.0-flash` | LiteLLM model string |
| `reasoning` | `ReActReasoning` | Reasoning strategy (can swap to `ReWOOReasoning`) |
| `rng` | `42` | Random seed |

Constants in `model.py`:

| Constant | Value | Description |
|---|---|---|
| `N_LLM_TRADERS` | 5 | Number of LLM traders |
| `N_RULE_TRADERS` | 5 | Number of rule-based traders |
| `INITIAL_PRICE` | $100 | Starting asset price |
| `INITIAL_CASH` | $10,000 | Starting cash per trader |
| `INITIAL_SHARES` | 50 | Starting shares per trader |
| `PRICE_IMPACT` | 0.003 | Price movement per net share of demand |

