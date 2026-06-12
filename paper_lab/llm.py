from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import textwrap
import urllib.error
import urllib.request


class MockLLM:
    """Deterministic local substitute for seeing the agent workflow without an API key."""

    def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        paper_excerpt = self._extract_paper_excerpt(user)

        if "synthesis agent" in system:
            return self._synthesize(user)
        if "Structure Analyst" in system:
            return self._structure_notes(paper_excerpt)
        if "Critic" in system:
            return self._critic_notes(paper_excerpt, temperature)
        if "Literature Connector" in system:
            return self._literature_notes(paper_excerpt)
        if "Application Strategist" in system:
            return self._application_notes(paper_excerpt)
        return "- Mock response: no specialized behavior matched."

    def _extract_paper_excerpt(self, user: str) -> str:
        marker = "Paper text:\n"
        if marker not in user:
            return user[:1200]
        return user.split(marker, 1)[1][:1600]

    def _structure_notes(self, excerpt: str) -> str:
        title = next((line.strip("# ").strip() for line in excerpt.splitlines() if line.strip()), "the paper")
        return textwrap.dedent(
            f"""
            - Main object: {title}.
            - Problem: complex research work needs expertise across multiple fields.
            - Proposed mechanism: a PI agent coordinates specialized scientist agents through meetings.
            - Evidence to inspect: whether role separation and meeting orchestration improve decisions over a single agent.
            - Important assumption: expert personas elicit meaningfully different reasoning paths.
            """
        ).strip()

    def _critic_notes(self, excerpt: str, temperature: float) -> str:
        diversity_note = "Parallel high-temperature meetings" if "parallel" in excerpt.lower() else "Multiple discussion rounds"
        return textwrap.dedent(
            f"""
            - The architecture may look scientific while still depending on model priors rather than grounded evidence.
            - {diversity_note} improves diversity, but it does not automatically improve truthfulness.
            - The human checkpoint is high leverage; unclear checkpoints could let weak conclusions propagate.
            - For a practice project, log disagreements explicitly instead of only preserving the final polished answer.
            - Temperature setting used here: {temperature:.2f}; higher diversity needs stronger synthesis discipline.
            """
        ).strip()

    def _literature_notes(self, excerpt: str) -> str:
        return textwrap.dedent(
            """
            - Search terms: multi-agent LLM, debate agents, LLM scientific discovery, critique agent, self-consistency.
            - Adjacent patterns: mixture-of-agents, reviewer-writer loops, tree-of-thought, and self-consistency sampling.
            - Compare against: single-agent paper summarizers and retrieval-augmented literature review systems.
            - Track which ideas require external retrieval versus which can be reasoned from the paper alone.
            """
        ).strip()

    def _application_notes(self, excerpt: str) -> str:
        return textwrap.dedent(
            """
            - Build a PI orchestrator, role-specific scientist agents, parallel meeting groups, and a final synthesizer.
            - Start with text or Markdown input; add PDF parsing only after the agent loop feels useful.
            - Save transcripts so you can study when debate helps and when it only creates extra noise.
            - Useful next feature: a related-work tool that each Literature Connector can call independently.
            """
        ).strip()

    def _synthesize(self, user: str) -> str:
        return textwrap.dedent(
            """
            # Multi-Agent Paper Reading Note

            ## TL;DR

            The paper's most reusable idea is the separation between a coordinating PI agent, specialized scientist agents, parallel high-diversity meetings, and a conservative final synthesis step.

            ## Paper Map

            - Problem: real research often requires more expert breadth than one person or one model pass can provide.
            - System: the PI agent frames tasks, convenes meetings, gathers opinions, and moves the project forward.
            - Expert layer: scientist agents contribute from assigned disciplinary viewpoints.
            - Human role: intervene at high-level checkpoints instead of micromanaging details.

            ## Key Mechanism

            Multiple independent meetings explore the same question with higher randomness. A later synthesis pass merges the strongest points with lower randomness. This uses diversity for exploration and synthesis for control.

            ## Strongest Insights

            - Role separation makes the system easier to inspect than a monolithic prompt.
            - Parallel meetings are a practical way to reduce single-agent conservatism.
            - The transcript is as important as the final answer because it shows disagreement and reasoning paths.

            ## Critiques

            - Diversity is not the same as correctness.
            - Expert personas can create a false sense of authority.
            - Human checkpoints need clear criteria or the system may simply produce polished uncertainty.

            ## Related Work To Chase

            - Multi-agent debate
            - Mixture-of-agents
            - Self-consistency sampling
            - Critic/reviewer agent loops
            - Retrieval-augmented literature review

            ## Reusable Ideas For My Own Multi-Agent Project

            - Use one PI orchestrator and four paper-reading roles: structure analyst, critic, literature connector, and application strategist.
            - Run 2-4 independent meetings per paper.
            - Preserve raw meeting notes before final synthesis.
            - Add tools gradually: PDF extraction first, related-paper retrieval second.

            ## Next Actions

            - Run this lab on one paper you already know well.
            - Compare mock mode, one real LLM meeting, and three real LLM meetings.
            - Add a transcript JSON export once the note format feels right.
            """
        ).strip()


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class OpenAIResponsesLLM:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PAPER_LAB_API_KEY")
        self.model = os.getenv("PAPER_LAB_MODEL", "gpt-4.1")
        self.base_url = os.getenv("PAPER_LAB_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.store = env_bool("PAPER_LAB_STORE", False)
        self.max_retries = int(os.getenv("PAPER_LAB_MAX_RETRIES", "5"))
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai provider.")

    def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        url = f"{self.base_url}/responses"
        payload = {
            "model": self.model,
            "input": messages,
            "temperature": temperature,
            "store": self.store,
        }
        data = self._post_with_retries(url, payload)

        return extract_response_text(data)

    def _post_with_retries(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code != 429 or attempt >= self.max_retries:
                    raise RuntimeError(f"LLM request failed: {exc.code} {details}") from exc

                wait_seconds = retry_wait_seconds(exc, details, attempt)
                print(
                    f"Rate limit hit; waiting {wait_seconds:.1f}s before retry "
                    f"{attempt + 1}/{self.max_retries}."
                )
                time.sleep(wait_seconds)

        raise RuntimeError("LLM request failed after retries.")


class OpenAICompatibleLLM(OpenAIResponsesLLM):
    """Backward-compatible provider name for older README commands."""


def extract_response_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and "text" in content:
                texts.append(content["text"])

    if texts:
        return "\n".join(texts)
    raise RuntimeError(f"Could not find text in Responses API payload: {json.dumps(data)[:800]}")


def retry_wait_seconds(exc: urllib.error.HTTPError, details: str, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after) + 1.0
        except ValueError:
            pass

    match = re.search(r"try again in ([0-9.]+)s", details, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0

    return min(60.0, 2.0 ** attempt + 2.0)


def build_llm(provider: str | None = None):
    load_dotenv()
    selected = provider or os.getenv("PAPER_LAB_PROVIDER", "mock")
    if selected == "mock":
        return MockLLM()
    if selected in {"openai", "openai-compatible"}:
        return OpenAIResponsesLLM()
    raise ValueError(f"Unknown provider: {selected}")
