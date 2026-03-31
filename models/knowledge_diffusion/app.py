import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import solara
from dotenv import load_dotenv
from mesa.visualization import SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter

from knowledge_diffusion.agents import ResearcherAgent
from knowledge_diffusion.model import KnowledgeDiffusionModel
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

model = KnowledgeDiffusionModel(
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"]["value"],
    rng=model_params["rng"]["value"],
)


def KnowledgeCoverageChart(model):
    """Line chart showing unique papers covered vs papers published, over steps."""
    update_counter.get()

    fig, ax = plt.subplots(figsize=(9, 3.5))

    try:
        df = model.datacollector.get_model_vars_dataframe()
    except Exception:
        ax.set_title("No data yet — click Step to begin")
        return solara.FigureMatplotlib(fig)

    if df.empty or "PapersPublished" not in df.columns:
        ax.set_title("Run a few steps to see knowledge coverage")
        return solara.FigureMatplotlib(fig)

    steps = df.index
    ax.plot(steps, df["PapersPublished"], color="#95a5a6", linestyle="--",
            linewidth=1.5, label="Papers published")
    ax.plot(steps, df["UniquePapersCovered"], color="#27ae60", linewidth=2,
            marker="o", markersize=4, label="Unique papers covered (≥1 reader)")

    ax.fill_between(steps, df["UniquePapersCovered"], df["PapersPublished"],
                    alpha=0.12, color="#e74c3c", label="Gap (uncovered)")

    ax.set_xlabel("Lab Day (Step)", fontsize=10)
    ax.set_ylabel("Paper Count", fontsize=10)
    ax.set_title("Knowledge Coverage vs Publication Rate", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return solara.FigureMatplotlib(fig)


def ResearcherPapersReadBar(model):
    """Bar chart of papers read by each researcher."""
    update_counter.get()

    researchers = [a for a in model.agents if isinstance(a, ResearcherAgent)]
    fig, ax = plt.subplots(figsize=(9, 3.5))

    if not researchers:
        ax.set_title("No agents yet")
        return solara.FigureMatplotlib(fig)

    names = [a.name.replace("Dr. ", "") for a in researchers]
    papers_read = [len(a.papers_read) for a in researchers]
    findings_shared = [a.findings_shared for a in researchers]

    x = range(len(names))
    width = 0.38

    bars1 = ax.bar([i - width / 2 for i in x], papers_read, width,
                   color="#2980b9", label="Papers read", edgecolor="white")
    bars2 = ax.bar([i + width / 2 for i in x], findings_shared, width,
                   color="#e67e22", label="Findings shared", edgecolor="white")

    for bar in bars1:
        v = bar.get_height()
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    str(int(v)), ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        v = bar.get_height()
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    str(int(v)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Per-Researcher Activity — Day {model.steps}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return solara.FigureMatplotlib(fig)


def SpreadActivityChart(model):
    """Line chart of cumulative sharing and challenge events over time."""
    update_counter.get()

    fig, ax = plt.subplots(figsize=(9, 3.5))

    try:
        df = model.datacollector.get_model_vars_dataframe()
    except Exception:
        ax.set_title("No data yet")
        return solara.FigureMatplotlib(fig)

    if df.empty or "TotalShares" not in df.columns:
        ax.set_title("Run a few steps to see spread activity")
        return solara.FigureMatplotlib(fig)

    ax.plot(df.index, df["TotalShares"], color="#8e44ad", linewidth=2,
            marker="o", markersize=4, label="Cumulative findings shared")
    ax.plot(df.index, df["TotalChallenges"], color="#e74c3c", linewidth=2,
            marker="^", markersize=4, linestyle="--", label="Cumulative challenges")
    ax.plot(df.index, df["TotalFocusUpdates"], color="#16a085", linewidth=1.5,
            marker="s", markersize=3, linestyle=":", label="Focus updates")

    ax.set_xlabel("Lab Day (Step)", fontsize=10)
    ax.set_ylabel("Cumulative Events", fontsize=10)
    ax.set_title("Knowledge Spread Activity Over Time", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return solara.FigureMatplotlib(fig)


@solara.component
def ResearcherStatusPanel(model):
    """Table showing each researcher's current state."""
    update_counter.get()

    researchers = [a for a in model.agents if isinstance(a, ResearcherAgent)]
    covered = model._unique_papers_covered()
    published = len(model.available_papers)

    solara.Text(
        f"Day {model.steps}  ·  Published {published} papers  ·  "
        f"Covered {covered} / {published}  ·  "
        f"Total shares {sum(a.findings_shared for a in researchers)}  ·  "
        f"Challenges {model.total_challenges}"
    )

    if model.available_papers:
        latest = model.available_papers[-1]
        solara.Text(f"Latest paper: \"{latest['title']}\" ({latest['field']})")

    rows = []
    for a in sorted(researchers, key=lambda x: x.name):
        read_ids = sorted(a.papers_read)
        rows.append({
            "Researcher": a.name,
            "Specialization": a.specialization[:45] + "…" if len(a.specialization) > 45 else a.specialization,
            "Papers Read": len(a.papers_read),
            "Paper IDs": str(read_ids) if read_ids else "—",
            "Shared": a.findings_shared,
            "Challenges": a.challenges_raised,
            "Focus Updates": a.focus_updates,
        })

    solara.DataFrame(pd.DataFrame(rows))



PapersPublishedPlot = make_plot_component("PapersPublished")
UniqueCoveredPlot = make_plot_component("UniquePapersCovered")
TotalSharesPlot = make_plot_component("TotalShares")


page = SolaraViz(
    model,
    renderer=None,
    components=[
        KnowledgeCoverageChart,
        ResearcherPapersReadBar,
        SpreadActivityChart,
        ResearcherStatusPanel,
        TotalSharesPlot,
    ],
    model_params=model_params,
    name="Knowledge Diffusion — Mesa-LLM",
)
