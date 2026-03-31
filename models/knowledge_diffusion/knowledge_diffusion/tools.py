"""Custom tools for the Knowledge Diffusion example.

Four tools are registered into `researcher_tool_manager` (agents.py) at import
time via @tool(tool_manager=...).  The inbuilt speak_to tool is also available
to all ResearcherAgents (copied in at ToolManager construction time).

Design principles:
  - read_paper   marks a paper as read and records the takeaway in model state,
                 so the DataCollector can track unique coverage per step.
  - share_finding uses agent.send_message() to route the summary into the
                 recipient's STLTMemory inbox - the core diffusion mechanic.
  - challenge_claim sends a targeted correction message and logs the event,
                 providing a proxy metric for epistemic rigour.
  - update_research_focus mutates agent.specialization and logs the pivot,
                 which the visualisation exposes in the status panel.
"""

import json
from typing import TYPE_CHECKING

from .agents import researcher_tool_manager
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=researcher_tool_manager)
def read_paper(
    agent: "LLMAgent",
    paper_id: int,
    key_takeaway: str,
    is_relevant: bool,
) -> str:
    """Formally read and process a paper from the available pool.

    Args:
        paper_id    : Integer ID of the paper to read (from the available list).
        key_takeaway: Your 1-2 sentence summary of the paper's core finding.
        is_relevant : True if the paper connects to your current specialization.
        agent       : Provided automatically.

    Returns:
        Confirmation string with paper details, or an error if paper_id is invalid.
    """
    try:
        paper_id = int(paper_id)
    except (TypeError, ValueError):
        return "read_paper: paper_id must be an integer.  No paper read."

    paper = next(
        (p for p in agent.model.available_papers if p["paper_id"] == paper_id), None
    )
    if paper is None:
        available_ids = [p["paper_id"] for p in agent.model.available_papers]
        return (
            f"read_paper: paper_id {paper_id} not found in available pool "
            f"(available: {available_ids}).  No paper read."
        )

    if paper_id in agent.papers_read:
        return f"read_paper: paper [{paper_id}] '{paper['title']}' already read.  Choose another action."

    agent.papers_read.add(paper_id)

    # Register this read in the model's spread tracker
    if paper_id not in agent.model.paper_spread:
        agent.model.paper_spread[paper_id] = {"direct_reads": [], "shared_summaries": []}
    agent.model.paper_spread[paper_id]["direct_reads"].append(agent.unique_id)

    relevance_tag = "RELEVANT to your specialization" if is_relevant else "adjacent field"

    return (
        f"READ [{paper_id}] \"{paper['title']}\"  ({relevance_tag})\n"
        f"Field: {paper['field']}\n"
        f"Your takeaway: {key_takeaway}\n"
        f"Total papers you have read: {len(agent.papers_read)}"
    )


@tool(tool_manager=researcher_tool_manager)
def share_finding(
    agent: "LLMAgent",
    colleague_ids: list[int],
    paper_id: int,
    summary: str,
    confidence: float,
) -> str:
    """Share a paper finding with one or more lab colleagues.

    Args:
        colleague_ids: List of colleague unique_ids to notify (use their integer IDs).
        paper_id     : ID of the paper this finding comes from.
        summary      : The insight you want to share - be precise and accurate.
        confidence   : Your confidence in this finding (0.0 = uncertain, 1.0 = certain).
        agent        : Provided automatically.

    Returns:
        Confirmation of who was notified, or an error if no valid recipients found.
    """
    # Normalise colleague_ids - LLM may pass a JSON string or a single int
    if isinstance(colleague_ids, str):
        try:
            colleague_ids = json.loads(colleague_ids)
        except (json.JSONDecodeError, ValueError):
            colleague_ids = [
                int(x.strip())
                for x in colleague_ids.strip("[]").split(",")
                if x.strip().lstrip("-").isdigit()
            ]
    if isinstance(colleague_ids, int):
        colleague_ids = [colleague_ids]
    colleague_ids = [int(cid) for cid in (colleague_ids or [])]

    try:
        paper_id = int(paper_id)
    except (TypeError, ValueError):
        paper_id = -1

    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # Resolve recipient agents
    recipients = [a for a in agent.model.agents if a.unique_id in colleague_ids]
    if not recipients:
        return (
            f"share_finding: no valid recipients found for IDs {colleague_ids}.  "
            f"Your colleagues have IDs: {agent.colleague_ids}"
        )

    paper = next((p for p in agent.model.available_papers if p["paper_id"] == paper_id), None)
    paper_title = paper["title"] if paper else f"Paper {paper_id}"

    message = (
        f"[SHARED FINDING from {agent.name}]  "
        f"Paper: \"{paper_title}\" (ID {paper_id})  |  "
        f"Confidence: {confidence:.0%}  |  "
        f"Summary: {summary}"
    )

    agent.send_message(message, recipients)
    agent.findings_shared += 1

    # Record the share in the model's spread tracker
    if paper_id not in agent.model.paper_spread:
        agent.model.paper_spread[paper_id] = {"direct_reads": [], "shared_summaries": []}
    agent.model.paper_spread[paper_id]["shared_summaries"].append(
        {"from": agent.unique_id, "to": [r.unique_id for r in recipients], "summary": summary}
    )

    recipient_names = [getattr(r, "name", str(r.unique_id)) for r in recipients]
    return (
        f"Shared finding about \"{paper_title}\" with {recipient_names}.  "
        f"Confidence: {confidence:.0%}.  "
        f"Total findings shared by you: {agent.findings_shared}"
    )


@tool(tool_manager=researcher_tool_manager)
def challenge_claim(
    agent: "LLMAgent",
    source_id: int,
    paper_title: str,
    concern: str,
) -> str:
    """Push back on an incorrect or imprecise claim shared by a colleague.

    Args:
        source_id  : unique_id of the colleague who made the claim.
        paper_title: Title or short identifier of the paper the claim concerns.
        concern    : Specific reason the claim is questionable or incorrect.
        agent      : Provided automatically.

    Returns:
        Confirmation that a challenge message was sent.
    """
    try:
        source_id = int(source_id)
    except (TypeError, ValueError):
        return "challenge_claim: source_id must be an integer."

    source = next((a for a in agent.model.agents if a.unique_id == source_id), None)
    if source is None:
        return f"challenge_claim: no agent with ID {source_id} found."

    source_name = getattr(source, "name", f"Researcher-{source_id}")
    message = (
        f"[CHALLENGE from {agent.name}]  "
        f"Re: \"{paper_title}\" - "
        f"I question this claim: {concern}"
    )
    agent.send_message(message, [source])
    agent.challenges_raised += 1
    agent.model.total_challenges += 1

    return (
        f"Challenge sent to {source_name} regarding \"{paper_title}\".  "
        f"Concern: {concern}  |  "
        f"Your total challenges raised: {agent.challenges_raised}"
    )


@tool(tool_manager=researcher_tool_manager)
def update_research_focus(
    agent: "LLMAgent",
    new_focus: str,
    justification: str,
) -> str:
    """Pivot your declared research specialization to a new subdomain.

    Args:
        new_focus    : The new research area or specialization string.
        justification: Why the new direction is more compelling or important now.
        agent        : Provided automatically.

    Returns:
        Confirmation of the focus change.
    """
    if not new_focus or not new_focus.strip():
        return "update_research_focus: new_focus cannot be empty."

    old_focus = agent.specialization
    agent.specialization = new_focus.strip()
    agent.focus_updates += 1
    agent.model.total_focus_updates += 1

    # Broadcast the pivot to colleagues so they can update their mental models
    colleagues = [a for a in agent.model.agents if a.unique_id in agent.colleague_ids]
    if colleagues:
        agent.send_message(
            f"[FOCUS UPDATE] {agent.name} has pivoted from '{old_focus}' to "
            f"'{agent.specialization}'.  Reason: {justification}",
            colleagues,
        )

    return (
        f"Research focus updated: '{old_focus}' -> '{agent.specialization}'.  "
        f"Colleagues notified.  "
        f"Justification: {justification}"
    )
