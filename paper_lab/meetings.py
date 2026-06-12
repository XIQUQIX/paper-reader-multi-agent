from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .agents import AgentRole, LLMClient, PIAgent, ScientistAgent, SynthesisAgent


@dataclass(frozen=True)
class MeetingTranscript:
    meeting_id: int
    responses: list[tuple[str, str]]

    def to_markdown(self) -> str:
        chunks = [f"## Meeting {self.meeting_id}"]
        for role, content in self.responses:
            chunks.append(f"### {role}\n\n{content}")
        return "\n\n".join(chunks)


@dataclass(frozen=True)
class LabResult:
    question: str
    meetings: list[MeetingTranscript]
    note: str

    def transcript_markdown(self) -> str:
        chunks = [f"# Transcript\n\nQuestion: {self.question}"]
        chunks.extend(meeting.to_markdown() for meeting in self.meetings)
        return "\n\n".join(chunks)


class PaperReadingLab:
    def __init__(self, llm: LLMClient, roles: list[AgentRole]):
        self.llm = llm
        self.pi = PIAgent()
        self.synthesis_agent = SynthesisAgent(llm)
        self.scientists = [ScientistAgent(role, llm) for role in roles]

    def run(self, paper: str, question: str | None, parallel_meetings: int) -> LabResult:
        framed_question = self.pi.frame_question(paper=paper, question=question)
        meeting_count = max(1, parallel_meetings)

        with ThreadPoolExecutor(max_workers=meeting_count) as executor:
            meetings = list(
                executor.map(
                    lambda meeting_id: self._run_meeting(
                        paper=paper,
                        question=framed_question,
                        meeting_id=meeting_id,
                    ),
                    range(1, meeting_count + 1),
                )
            )

        note = self.synthesis_agent.synthesize(
            question=framed_question,
            meeting_notes=[meeting.to_markdown() for meeting in meetings],
            temperature=0.2,
        )
        return LabResult(question=framed_question, meetings=meetings, note=note)

    def _run_meeting(self, paper: str, question: str, meeting_id: int) -> MeetingTranscript:
        temperature = 0.85 + (meeting_id - 1) * 0.05
        responses = []
        for scientist in self.scientists:
            response = scientist.respond(
                paper=paper,
                question=question,
                meeting_id=meeting_id,
                temperature=temperature,
            )
            responses.append((response.role, response.content))
        return MeetingTranscript(meeting_id=meeting_id, responses=responses)
