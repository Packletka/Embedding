from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Dict, Any

import pdfplumber


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = "\n".join(line.strip() for line in text.split("\n"))

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def extract_pages_from_pdf(pdf_path: Path) -> Iterator[Dict[str, Any]]:
    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            norm = normalize_text(raw)
            yield {
                "page": idx,
                "raw_text": raw,
                "text": norm,
                "char_count": len(norm),
            }
