import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from paper_lab.agents import DEFAULT_ROLES
from paper_lab.llm import MockLLM, extract_response_text, retry_wait_seconds
from paper_lab.meetings import PaperReadingLab
from paper_lab.paper_loader import load_paper


class PaperReadingLabTest(unittest.TestCase):
    def test_lab_runs_parallel_meetings_and_synthesizes_note(self):
        lab = PaperReadingLab(llm=MockLLM(), roles=DEFAULT_ROLES)

        result = lab.run(
            paper="# Test Paper\n\nA PI coordinates several scientist agents.",
            question=None,
            parallel_meetings=2,
        )

        self.assertIn("Test Paper", result.question)
        self.assertEqual(len(result.meetings), 2)
        self.assertIn("Multi-Agent Paper Reading Note", result.note)
        self.assertIn("Structure Analyst", result.transcript_markdown())

    def test_extract_response_text_from_responses_payload(self):
        payload = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "hello from responses",
                        }
                    ]
                }
            ]
        }

        self.assertEqual(extract_response_text(payload), "hello from responses")

    def test_load_text_paper(self):
        with TemporaryDirectory() as temp_dir:
            paper_path = Path(temp_dir) / "paper.md"
            paper_path.write_text("# Local Paper\n\nBody", encoding="utf-8")

            self.assertEqual(load_paper(paper_path), "# Local Paper\n\nBody")

    def test_load_unsupported_format_fails(self):
        with TemporaryDirectory() as temp_dir:
            paper_path = Path(temp_dir) / "paper.doc"
            paper_path.write_text("Body", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_paper(paper_path)

    def test_retry_wait_seconds_from_rate_limit_message(self):
        class Error:
            headers = {}

        details = "Rate limit reached. Please try again in 20.956s."

        self.assertAlmostEqual(retry_wait_seconds(Error(), details, attempt=0), 22.956)


if __name__ == "__main__":
    unittest.main()
