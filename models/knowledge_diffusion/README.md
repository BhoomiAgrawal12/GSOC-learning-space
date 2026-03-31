# Knowledge Diffusion - Academic Research Spread Through a Researcher Network

An agent-based model of **how scientific knowledge propagates, transforms, and is
challenged** as it moves through a small multi-disciplinary lab network.  Six
LLM-powered researchers read newly published papers, share findings with colleagues,
challenge incorrect claims, and occasionally pivot their research focus - with
**STLTMemory consolidation** as the core mechanism that determines whether knowledge
accumulates accurately or degrades into noise.

## What This Model Demonstrates

| Mesa-LLM Feature | How It's Used |
|---|---|
| `STLTMemory` (short_term=6, consolidation=3) | When a researcher's memory exceeds 6 entries, the oldest 3 are summarised into long-term memory by a second LLM call. This mirrors how researchers retain the *gist* of many papers without perfect recall - and means early findings can drift in representation as they get consolidated. |
| `speak_to` (inbuilt, via `share_finding`) | Findings route into recipients' STLTMemory inbox; recipients read them in the next step. This is the primary diffusion mechanic. |
| `vision=-1` | All researchers observe each other's internal state - simulating a weekly lab meeting where everyone can see who is working on what. |
| Custom tools | `read_paper`, `share_finding`, `challenge_claim`, `update_research_focus`. |

## Model Design

```
6 ResearcherAgents in a sparse undirected network
  Dr. Chen (NLP)        ---- Dr. Wei (Vision)
      │                           │
  Dr. Patel (RL)         Dr. Okafor (Systems)
      │                           |
  Dr. Tanaka (Theory) ---------------------Dr. Rivera (Healthcare AI)

Each step:
  1. One new paper published into the available pool (12-paper cycle)
  2. All researchers act simultaneously (shuffle_do)
     Each chooses ONE of: read_paper / share_finding / challenge_claim / update_research_focus
  3. Shared findings route through send_message → STLTMemory of recipients
  4. DataCollector captures coverage, shares, challenges per step
```

**12 pre-seeded papers** span NLP, CV, RL, Systems, Theory, and Healthcare AI -
matching the researchers' specializations so cross-disciplinary decisions are
non-trivial (e.g. does Dr. Rivera read the NLP paper because it mentions clinical
decision support?).

## Key Observables

| Metric | What it shows |
|---|---|
| `UniquePapersCovered` | How many distinct papers ≥1 researcher has read - measures spread speed. |
| `TotalShares` | Secondary dissemination events - how much researchers relay information. |
| `TotalChallenges` | Epistemic rigour - how often incorrect/imprecise claims get pushed back. |
| Per-agent `PapersRead` | Whether knowledge is concentrated in 1-2 researchers or distributed. |

## Setup

```bash
cd models/knowledge_diffusion

# Install dependencies
pip install mesa mesa-llm solara python-dotenv matplotlib pandas rich

# Set your LLM API key
echo "GEMINI_API_KEY=your-key-here" > .env

# Run with Solara visualisation
solara run app.py

# Run headless
python -m knowledge_diffusion.model
```

## Files

```
knowledge_diffusion/
├── app.py                              # Solara entry-point
└── knowledge_diffusion/
    ├── __init__.py                     # registers tools into researcher_tool_manager
    ├── agents.py                       # ResearcherAgent (LLM)
    ├── model.py                        # KnowledgeDiffusionModel + PAPER_POOL + RESEARCHER_PROFILES
    └── tools.py                        # read_paper, share_finding, challenge_claim, update_research_focus
```

## Visualisation Panels

1. **Knowledge Coverage Chart** - published vs covered papers over time (gap = uncovered)
2. **Per-Researcher Activity Bar** - papers read + findings shared per researcher
3. **Spread Activity Chart** - cumulative shares, challenges, focus updates over time
4. **Researcher Status Table** - current specialization, papers read (IDs), share/challenge counts

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `llm_model` | `gemini/gemini-2.0-flash` | LiteLLM model string |
| `reasoning` | `ReActReasoning` | Reasoning strategy |
| `rng` | `42` | Random seed |

