from __future__ import annotations

from pathlib import Path


def load_paper(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported paper format: {path.suffix}. Use .pdf, .txt, or .md.")


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF input requires pypdf. Install it or convert the PDF to .txt first.") from exc

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"\n\n--- Page {index} ---\n\n{text}")

    if not pages:
        raise RuntimeError(f"No extractable text found in PDF: {path}")
    return "".join(pages).strip()
