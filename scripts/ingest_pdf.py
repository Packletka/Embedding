from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from src.ingest.pdf_ingest import extract_pages_from_pdf

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/pages.jsonl")

CATEGORY_BY_FILENAME = {
    "general_changes.pdf": "general",
    "heroes_changes.pdf": "heroes",
    "items_changes.pdf": "items",
    "neutral_creeps_changes.pdf": "neutral_creeps",
    "neutral_items_changes.pdf": "neutral_items",
}


def make_doc_id(source_file: str, page: int) -> str:
    base = source_file.replace(".pdf", "")
    return f"{base}_p{page:03d}"


def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([p for p in RAW_DIR.glob("*.pdf") if p.name in CATEGORY_BY_FILENAME])

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {RAW_DIR.resolve()} matching CATEGORY_BY_FILENAME keys."
        )

    total_written = 0

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for pdf_path in pdf_files:
            category = CATEGORY_BY_FILENAME[pdf_path.name]
            print(f"[ingest] {pdf_path.name} -> category={category}")

            pages_total = 0
            pages_written = 0
            empty_pages = 0

            for page_obj in extract_pages_from_pdf(pdf_path):
                pages_total += 1
                text = page_obj["text"]

                if len(text) < 30:
                    empty_pages += 1
                    continue

                record = {
                    "doc_id": make_doc_id(pdf_path.name, page_obj["page"]),
                    "category": category,
                    "source_type": "pdf",
                    "source_file": pdf_path.name,
                    "page": page_obj["page"],
                    "text": text,
                    "char_count": page_obj["char_count"],
                }

                write_jsonl_line(out, record)
                pages_written += 1
                total_written += 1

            print(
                f"    pages_total={pages_total}, pages_written={pages_written}, skipped_empty={empty_pages}"
            )

    print(f"[done] wrote {total_written} records -> {OUT_PATH}")


if __name__ == "__main__":
    main()
