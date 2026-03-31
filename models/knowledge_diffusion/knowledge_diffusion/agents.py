"""Agent definitions for the Knowledge Diffusion example.

A single agent type - ResearcherAgent - populates a small academic network.
Researchers read newly published papers, share findings with lab colleagues,
challenge potentially incorrect claims, and periodically pivot their research
focus when a new subdomain becomes compelling.

The model intentionally has no rule-based baseline agent: the interesting
observable here is not LLM vs rule-based performance, but rather how knowledge
consolidation (STLTMemory) shapes whether insights spread accurately or degrade
into noise as they pass through multiple researchers.
"""

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

researcher_tool_manager = ToolManager()


def _get_recent_messages(agent, max_messages: int = 5) -> str:
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
                sender_id = entry.content.get("sender", "?")
                msg = entry.content.get("message", "")
                # Try to resolve sender id -> name
                try:
                    sender = next(a for a in agent.model.agents if a.unique_id == sender_id)
                    sender_label = getattr(sender, "name", f"Researcher-{sender_id}")
                except StopIteration:
                    sender_label = f"Researcher-{sender_id}"
                messages.append(f"  [{sender_label}]: {msg}")

    messages.reverse()
    return "\n".join(messages) if messages else "  No messages received yet."


class ResearcherAgent(LLMAgent):
    """LLM-powered academic researcher in a knowledge diffusion simulation.

    Each step the researcher:
      1. Sees all papers currently available in the model's paper pool.
      2. Reads messages in their inbox from colleagues (via STLTMemory).
      3. Chooses ONE action:
           - read_paper        : formally read a paper and store the key takeaway.
           - share_finding     : forward an insight to one or more lab colleagues.
           - challenge_claim   : push back on an incorrect claim received from a peer.
           - update_research_focus : pivot their declared specialization.

    Mesa-LLM features demonstrated:
      - STLTMemory (short_term_capacity=6, consolidation_capacity=3):
          Each step adds a new memory entry (paper read / message received).
          When capacity is exceeded, the oldest 3 entries are consolidated by the
          LLM into a long-term summary - so the agent retains the *gist* of many
          papers without flooding the context window.
      - speak_to (inbuilt): used internally by share_finding to route messages.
      - vision=-1 : all researchers observe each other's internal state (simulating
          a lab meeting where everyone can see who is working on what).
      - Custom tools: read_paper, share_finding, challenge_claim, update_research_focus.

    Attributes:
        name             : human-readable label (e.g. "Dr. Chen").
        specialization   : current research focus area.
        colleague_ids    : list[int] of unique_ids this researcher is connected to.
        papers_read      : set of paper_ids the researcher has formally processed.
        findings_shared  : cumulative count of share_finding calls made.
        challenges_raised: cumulative count of challenge_claim calls made.
        focus_updates    : cumulative count of update_research_focus calls made.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model: str,
        name: str,
        specialization: str,
        colleague_ids: list[int],
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=(
                f"You are {name}, a researcher specialising in {specialization}.  "
                "You work in a small interdisciplinary lab.  Each day (simulation step) "
                "you can read newly available papers in your feed, share important "
                "findings with colleagues, challenge claims that seem incorrect based on "
                "your expertise, or update your research focus when a new direction looks "
                "compelling.  Be selective: you can only take one substantive action per "
                "step.  Prioritise papers closest to your specialization, but stay open "
                "to cross-disciplinary insights.  When sharing, be precise - distorted "
                "information degrades the lab's collective knowledge."
            ),
            vision=-1,
            internal_state=[
                f"name:{name}",
                f"specialization:{specialization}",
            ],
        )
        self.tool_manager = researcher_tool_manager

        self.name = name
        self.specialization = specialization
        self.colleague_ids: list[int] = list(colleague_ids)

        self.papers_read: set[int] = set()
        self.findings_shared: int = 0
        self.challenges_raised: int = 0
        self.focus_updates: int = 0

    def step(self):
        observation = self.generate_obs()
        recent_msgs = _get_recent_messages(self)

        available = [
            p for p in self.model.available_papers
            if p["paper_id"] not in self.papers_read
        ]
        already_read = [
            p for p in self.model.available_papers
            if p["paper_id"] in self.papers_read
        ]

        if available:
            unread_block = "\n".join(
                f"  [{p['paper_id']}] \"{p['title']}\"  "
                f"(field: {p['field']})  "
                f"Abstract: {p['abstract']}"
                for p in available
            )
        else:
            unread_block = "  All available papers have already been read."

        if already_read:
            read_block = ", ".join(
                f"[{p['paper_id']}] \"{p['title']}\""
                for p in already_read
            )
        else:
            read_block = "  None yet."

        # Resolve colleague names
        colleague_info = []
        for cid in self.colleague_ids:
            try:
                c = next(a for a in self.model.agents if a.unique_id == cid)
                colleague_info.append(
                    f"  ID {cid}: {getattr(c, 'name', '?')} "
                    f"(specialization: {getattr(c, 'specialization', '?')}, "
                    f"papers read: {len(getattr(c, 'papers_read', set()))})"
                )
            except StopIteration:
                colleague_info.append(f"  ID {cid}: unknown")

        prompt = (
            f"LAB DAY {self.model.steps}\n\n"
            f"YOU ARE: {self.name}  (specialization: {self.specialization})\n"
            f"Papers you have read: {read_block}\n\n"
            f"NEWLY AVAILABLE PAPERS (unread by you):\n{unread_block}\n\n"
            f"YOUR COLLEAGUES:\n" + "\n".join(colleague_info) + "\n\n"
            f"MESSAGES FROM COLLEAGUES:\n{recent_msgs}\n\n"
            f"INSTRUCTIONS:\n"
            f"Choose exactly ONE action this step:\n\n"
            f"  read_paper(paper_id, key_takeaway, is_relevant)\n"
            f"    -> Formally process a paper from the list above.\n"
            f"      key_takeaway: 1-2 sentence summary of the core finding.\n"
            f"      is_relevant: true if it connects to your specialization.\n\n"
            f"  share_finding(colleague_ids, paper_id, summary, confidence)\n"
            f"    -> Forward a key insight to one or more colleagues (use their IDs).\n"
            f"      summary: what you want them to know (be precise!).\n"
            f"      confidence: float 0-1 reflecting how certain you are.\n\n"
            f"  challenge_claim(source_id, paper_title, concern)\n"
            f"    -> Push back on something a colleague shared that seems incorrect.\n"
            f"      concern: specific reason the claim is questionable.\n\n"
            f"  update_research_focus(new_focus, justification)\n"
            f"    -> Pivot your declared specialization to a new subdomain.\n"
            f"      justification: why the new direction is more compelling.\n\n"
            f"Prioritise: read papers in your specialization first.  If you have read "
            f"a paper with high-value findings, share with the most relevant colleague "
            f"next step.  Challenge incorrect claims to prevent misinformation spread."
        )

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[
                "read_paper",
                "share_finding",
                "challenge_claim",
                "update_research_focus",
            ],
        )
        self.apply_plan(plan)
