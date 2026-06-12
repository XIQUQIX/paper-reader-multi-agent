from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        """Return a model completion for chat-style messages."""


@dataclass(frozen=True)
class AgentRole:
    name: str
    mission: str
    output_focus: str


@dataclass(frozen=True)
class AgentResponse:
    role: str
    content: str


class ScientistAgent:
    def __init__(self, role: AgentRole, llm: LLMClient):
        self.role = role
        self.llm = llm

    def respond(self, paper: str, question: str, meeting_id: int, temperature: float) -> AgentResponse:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the {self.role.name} in a multi-agent paper reading lab. "
                    f"Mission: {self.role.mission} "
                    f"Focus your output on: {self.role.output_focus}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Meeting group: {meeting_id}\n"
                    f"Research question: {question}\n\n"
                    "Paper text:\n"
                    f"{paper}\n\n"
                    "Give concise, concrete notes. Prefer claims that can be checked against the paper."
                ),
            },
        ]
        return AgentResponse(
            role=self.role.name,
            content=self.llm.complete(messages, temperature=temperature),
        )


class PIAgent:
    def frame_question(self, paper: str, question: str | None) -> str:
        if question:
            return question

        first_line = next((line.strip("# ").strip() for line in paper.splitlines() if line.strip()), "")
        if first_line:
            return f"What are the core contributions, assumptions, risks, and reusable ideas in '{first_line}'?"
        return "What are the core contributions, assumptions, risks, and reusable ideas in this paper?"


class SynthesisAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def synthesize(self, question: str, meeting_notes: list[str], temperature: float = 0.2) -> str:
        joined_notes = "\n\n".join(meeting_notes)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the synthesis agent. Merge multiple independent agent meetings into "
                    "one precise reading note. Preserve disagreements and uncertainty. Avoid filler."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {question}\n\n"
                    f"Meeting notes:\n{joined_notes}\n\n"
                    "Write a Markdown reading note with these sections: TL;DR, Paper Map, "
                    "Key Mechanism, Strongest Insights, Critiques, Related Work To Chase, "
                    "Reusable Ideas For My Own Multi-Agent Project, Next Actions."
                ),
            },
        ]
        return self.llm.complete(messages, temperature=temperature)


DEFAULT_ROLES = [
    AgentRole(
        name="Structure Analyst",
        mission="Decompose the paper into problem, method, evidence, assumptions, and conclusion.",
        output_focus="paper structure, causal chain, and what each section contributes",
    ),
    AgentRole(
        name="Critic",
        mission="Challenge the argument and identify weak evidence, ambiguity, and overclaims.",
        output_focus="risks, missing controls, fragile assumptions, and questions to ask",
    ),
    AgentRole(
        name="Literature Connector",
        mission="Connect the paper to neighboring ideas and suggest follow-up literature searches.",
        output_focus="keywords, adjacent papers, intellectual lineage, and search directions",
    ),
    AgentRole(
        name="Application Strategist",
        mission="Translate the paper into implementation ideas and reusable project patterns.",
        output_focus="features, architecture patterns, experiments, and implementation milestones",
    ),
]
