from mesa.datacollection import DataCollector
from mesa.model import Model
from rich import print as rprint

from .agents import ResearcherAgent
from mesa_llm.reasoning.reasoning import Reasoning

RESEARCHER_PROFILES: list[dict] = [
    {
        "name": "Dr. Chen",
        "specialization": "natural language processing and transformer architectures",
        "colleague_names": ["Dr. Wei", "Dr. Patel"],
    },
    {
        "name": "Dr. Wei",
        "specialization": "computer vision and convolutional neural networks",
        "colleague_names": ["Dr. Chen", "Dr. Okafor"],
    },
    {
        "name": "Dr. Patel",
        "specialization": "reinforcement learning and multi-agent systems",
        "colleague_names": ["Dr. Chen", "Dr. Tanaka"],
    },
    {
        "name": "Dr. Okafor",
        "specialization": "ML systems, scalable training infrastructure, and MLOps",
        "colleague_names": ["Dr. Wei", "Dr. Tanaka"],
    },
    {
        "name": "Dr. Tanaka",
        "specialization": "statistical learning theory and generalisation bounds",
        "colleague_names": ["Dr. Patel", "Dr. Okafor", "Dr. Rivera"],
    },
    {
        "name": "Dr. Rivera",
        "specialization": "AI applications in healthcare and clinical decision support",
        "colleague_names": ["Dr. Tanaka"],
    },
]

PAPER_POOL: list[dict] = [
    {
        "paper_id": 0,
        "title": "Attention Is All You Need: Revisited at Scale",
        "field": "natural language processing",
        "abstract": (
            "We train transformer models up to 540B parameters and find that "
            "scaling laws hold across four orders of magnitude.  Emergent abilities "
            "appear at 62B parameters for multi-step arithmetic and 137B for "
            "chain-of-thought reasoning."
        ),
        "key_claim": "Emergent abilities in transformers appear at 62B parameters.",
    },
    {
        "paper_id": 1,
        "title": "Sparse Mixture-of-Experts Reduces Inference Costs by 4×",
        "field": "ML systems",
        "abstract": (
            "Routing tokens to 2 of 64 expert FFN layers reduces FLOPs by 75% "
            "while matching dense model quality on standard benchmarks.  Experts "
            "specialise into syntactic, semantic, and world-knowledge subgroups."
        ),
        "key_claim": "MoE routing to 2/64 experts cuts FLOPs by 75% with no quality loss.",
    },
    {
        "paper_id": 2,
        "title": "Self-Supervised Vision Transformers Match Supervised CNNs on ImageNet",
        "field": "computer vision",
        "abstract": (
            "DINO-style self-supervised ViT-L achieves 86.1% top-1 accuracy on "
            "ImageNet without any labels, matching ResNet-152 trained with full "
            "supervision.  Patch attention maps produce interpretable segmentations."
        ),
        "key_claim": "Self-supervised ViT-L matches supervised CNNs on ImageNet at 86.1% top-1.",
    },
    {
        "paper_id": 3,
        "title": "RLHF Beyond Reward Hacking: Conservative Policy Optimisation",
        "field": "reinforcement learning",
        "abstract": (
            "We show that standard PPO with a learned reward model leads to "
            "reward hacking in 73% of runs beyond 10k steps.  A conservative "
            "KL-constrained update reduces hacking to 8% while preserving 94% "
            "of peak reward."
        ),
        "key_claim": "KL-constrained PPO reduces reward hacking from 73% to 8% in RLHF.",
    },
    {
        "paper_id": 4,
        "title": "Double Descent in Deep Networks: A Unified View",
        "field": "statistical learning theory",
        "abstract": (
            "We unify bias-variance tradeoff and double descent under a single "
            "interpolation threshold framework.  Models past the interpolation "
            "threshold generalise despite zero training loss when implicit "
            "regularisation via SGD selects minimum-norm solutions."
        ),
        "key_claim": "Double descent unifies with bias-variance tradeoff via interpolation threshold theory.",
    },
    {
        "paper_id": 5,
        "title": "Large Language Models as Clinical Decision Support: A Safety Audit",
        "field": "AI in healthcare",
        "abstract": (
            "We evaluate GPT-4 and Claude-3 on 2,400 clinical vignettes.  Both "
            "models achieve 89% accuracy on diagnosis but hallucinate drug dosages "
            "in 14% of cases.  A retrieval-augmented pipeline reduces hallucination "
            "to 2.1%."
        ),
        "key_claim": "RAG reduces LLM drug dosage hallucination from 14% to 2.1% in clinical settings.",
    },
    {
        "paper_id": 6,
        "title": "Mechanistic Interpretability: Circuits in Transformers",
        "field": "natural language processing",
        "abstract": (
            "We identify 'induction heads' — a two-attention-layer circuit — "
            "responsible for in-context learning in transformer LMs.  Ablating "
            "these heads reduces few-shot accuracy by 37% on average."
        ),
        "key_claim": "Induction head circuits are responsible for in-context learning in transformers.",
    },
    {
        "paper_id": 7,
        "title": "Flash Attention 3: Hardware-Aware Exact Attention in O(N) Memory",
        "field": "ML systems",
        "abstract": (
            "Tiling and kernel fusion reduce attention memory from O(N²) to O(N) "
            "while maintaining mathematical exactness.  Throughput improves 2.7× "
            "over standard attention on A100 GPUs at sequence length 32k."
        ),
        "key_claim": "Flash Attention 3 achieves O(N) memory for exact attention with 2.7× speedup.",
    },
    {
        "paper_id": 8,
        "title": "Foundation Models for Retinal Disease Screening: A Prospective Study",
        "field": "AI in healthcare",
        "abstract": (
            "A ViT-B fine-tuned on 3.2M fundus images detects diabetic retinopathy "
            "with 94.3% sensitivity and 96.1% specificity — outperforming the median "
            "ophthalmologist (91.2% / 93.4%) in a prospective 1,200-patient trial."
        ),
        "key_claim": "Foundation model for retinal screening outperforms median ophthalmologist.",
    },
    {
        "paper_id": 9,
        "title": "Multi-Agent Reinforcement Learning Under Partial Observability",
        "field": "reinforcement learning",
        "abstract": (
            "We prove that decentralised-POMDP with communication is NEXP-complete "
            "in general but tractable for sparse-reward cooperative tasks when agents "
            "share a common prior.  A new QMIX variant achieves SOTA on StarCraft II."
        ),
        "key_claim": "Dec-POMDP with communication is NEXP-complete; tractable under sparse reward + common prior.",
    },
    {
        "paper_id": 10,
        "title": "Contrastive Learning Improves Zero-Shot Transfer by 22%",
        "field": "computer vision",
        "abstract": (
            "CLIP-style image-text contrastive pretraining on 400M pairs achieves "
            "76.2% zero-shot accuracy on ImageNet, 22% above the previous best.  "
            "Retrieval-augmented generation with frozen CLIP features matches "
            "fine-tuned ResNets on 10 downstream tasks."
        ),
        "key_claim": "CLIP-style pretraining achieves 76.2% zero-shot ImageNet accuracy, 22% above prior best.",
    },
    {
        "paper_id": 11,
        "title": "Generalisation Bounds for Overparameterised Neural Networks via PAC-Bayes",
        "field": "statistical learning theory",
        "abstract": (
            "PAC-Bayes bounds applied to the flat minima found by SGD explain "
            "why overparameterised networks generalise.  We derive tight bounds that "
            "scale as O(sqrt(sharpness / n)) and validate on CIFAR-10 and Penn Treebank."
        ),
        "key_claim": "PAC-Bayes bounds O(sqrt(sharpness/n)) explain generalisation of overparameterised networks.",
    },
]


class KnowledgeDiffusionModel(Model):
    """Academic knowledge diffusion through a small multi-disciplinary researcher network.

    Six LLM-powered researchers form a sparse social network.  Each simulation step:

      1. One new paper is published into the available paper pool.
      2. All researchers act simultaneously (shuffle_do):
           - read a newly available paper, or
           - share a finding with a colleague, or
           - challenge an incorrect claim, or
           - update their research focus.
      3. Messages sent via share_finding are routed into recipients' STLTMemory.
         STLTMemory consolidation then summarises old entries so researchers
         retain the *gist* without unbounded context growth.

    Key observables:
      - Unique papers covered (how many distinct papers at least one researcher read).
      - Total secondary shares (how many times a finding was forwarded).
      - Total challenges raised (proxy for epistemic rigour / misinformation detection).
      - Per-researcher papers read (diversity of knowledge acquisition).

    Args:
        reasoning : Reasoning class (e.g. ReActReasoning).
        llm_model : LiteLLM model string, e.g. ``"gemini/gemini-2.0-flash"``.
        rng       : Random seed.
    """

    def __init__(
        self,
        reasoning: type[Reasoning],
        llm_model: str = "gemini/gemini-2.0-flash",
        rng: int = 42,
    ):
        super().__init__(rng=rng)

        # Paper state
        self.available_papers: list[dict] = []   # grows by 1 per step
        self._paper_queue = list(PAPER_POOL)      # papers not yet published
        self.paper_spread: dict[int, dict] = {}

        # Global event counters (updated by tools)
        self.total_challenges: int = 0
        self.total_focus_updates: int = 0

        # Create researcher agents (colleague_ids populated below after creation)
        for profile in RESEARCHER_PROFILES:
            ResearcherAgent(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                name=profile["name"],
                specialization=profile["specialization"],
                colleague_ids=[],  # patched below
            )

        # Patch colleague_ids: map name -> unique_id and assign
        self._patch_colleagues()

        self.datacollector = DataCollector(
            model_reporters={
                "PapersPublished": lambda m: len(m.available_papers),
                "UniquePapersCovered": lambda m: m._unique_papers_covered(),
                "TotalShares": lambda m: sum(
                    a.findings_shared for a in m.agents
                ),
                "TotalChallenges": "total_challenges",
                "TotalFocusUpdates": "total_focus_updates",
            },
            agent_reporters={
                "PapersRead": lambda a: len(a.papers_read),
                "FindingsShared": "findings_shared",
                "ChallengesRaised": "challenges_raised",
                "Specialization": "specialization",
            },
        )


    def _patch_colleagues(self) -> None:
        """Resolve colleague names -> unique_ids after all agents are created."""
        name_to_id = {a.name: a.unique_id for a in self.agents}
        profile_by_name = {p["name"]: p for p in RESEARCHER_PROFILES}

        for agent in self.agents:
            profile = profile_by_name.get(agent.name, {})
            agent.colleague_ids = [
                name_to_id[n]
                for n in profile.get("colleague_names", [])
                if n in name_to_id
            ]


    def _unique_papers_covered(self) -> int:
        """Number of distinct papers read by at least one researcher."""
        all_read: set[int] = set()
        for a in self.agents:
            all_read.update(a.papers_read)
        return len(all_read)

    def _coverage_percent(self) -> float:
        """Fraction of published papers covered by at least one researcher."""
        published = len(self.available_papers)
        if published == 0:
            return 0.0
        return self._unique_papers_covered() / published * 100


    def step(self) -> None:
        self.datacollector.collect(self)

        if self._paper_queue:
            new_paper = self._paper_queue.pop(0)
        else:
            self._paper_queue = [
                p for p in PAPER_POOL
                if p["paper_id"] not in self._unique_read_paper_ids()
            ]
            new_paper = self._paper_queue.pop(0) if self._paper_queue else None

        if new_paper:
            self.available_papers.append(new_paper)
            rprint(
                f"\n[bold cyan] Lab Day {self.steps} [/bold cyan]  "
                f"[yellow]New paper: \"{new_paper['title']}\" "
                f"(field: {new_paper['field']})[/yellow]"
            )
        else:
            rprint(f"\n[bold cyan] Lab Day {self.steps} [/bold cyan]  "
                   f"[yellow]No new paper today — all papers published.[/yellow]")

        self.agents.shuffle_do("step")

        covered = self._unique_papers_covered()
        total_shares = sum(a.findings_shared for a in self.agents)
        rprint(
            f"[bold green]  Day {self.steps} end: "
            f"published={len(self.available_papers)}  "
            f"covered={covered} ({self._coverage_percent():.0f}%)  "
            f"shares={total_shares}  "
            f"challenges={self.total_challenges}[/bold green]"
        )

    def _unique_read_paper_ids(self) -> set[int]:
        all_read: set[int] = set()
        for a in self.agents:
            all_read.update(a.papers_read)
        return all_read


if __name__ == "__main__":

    from mesa_llm.reasoning.react import ReActReasoning

    m = KnowledgeDiffusionModel(
        reasoning=ReActReasoning,
        llm_model="gemini/gemini-2.0-flash",
        rng=42,
    )
    for _ in range(8):
        m.step()

    rprint("\n[bold magenta]=== Final Knowledge State ===[/bold magenta]")
    for a in m.agents:
        rprint(
            f"  {a.name:<14} specialization: {a.specialization:<50} "
            f"papers_read={len(a.papers_read)}  "
            f"shared={a.findings_shared}  challenges={a.challenges_raised}"
        )
    rprint(f"\nUnique papers covered: {m._unique_papers_covered()} / {len(m.available_papers)}")
    rprint(f"Total challenges raised: {m.total_challenges}")
