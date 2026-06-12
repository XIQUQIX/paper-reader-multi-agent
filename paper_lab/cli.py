from __future__ import annotations

import argparse
from pathlib import Path

from .agents import DEFAULT_ROLES
from .llm import build_llm
from .meetings import PaperReadingLab
from .paper_loader import load_paper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Virtual Lab style paper reading session.")
    parser.add_argument("--paper", required=True, help="Path to a .pdf, .txt, or .md paper/note file.")
    parser.add_argument("--question", default=None, help="Optional research question for the PI agent.")
    parser.add_argument("--parallel-meetings", type=int, default=3, help="Number of independent meetings.")
    parser.add_argument("--output", default=None, help="Optional Markdown output path.")
    parser.add_argument(
        "--max-paper-chars",
        type=int,
        default=60000,
        help="Maximum paper characters sent to each agent. Use 0 to send the full extracted text.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["mock", "openai-compatible"],
        help="LLM provider. Defaults to PAPER_LAB_PROVIDER or mock.",
    )
    parser.add_argument("--show-transcript", action="store_true", help="Print all meeting notes before synthesis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_path = Path(args.paper)
    paper = load_paper(paper_path)
    if args.max_paper_chars > 0 and len(paper) > args.max_paper_chars:
        print(f"Loaded {len(paper)} characters; sending the first {args.max_paper_chars}.")
        paper = paper[: args.max_paper_chars]

    llm = build_llm(provider=args.provider)
    lab = PaperReadingLab(llm=llm, roles=DEFAULT_ROLES)
    result = lab.run(
        paper=paper,
        question=args.question,
        parallel_meetings=args.parallel_meetings,
    )

    if args.show_transcript:
        print(result.transcript_markdown())
        print("\n---\n")

    print(result.note)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.note, encoding="utf-8")
        print(f"\nSaved reading note to {output_path}")
