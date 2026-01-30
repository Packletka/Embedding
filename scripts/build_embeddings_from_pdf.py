from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.ingest.pdf_ingest import extract_pages_from_pdf
from src.chunking.chunker import ChunkerConfig, chunk_text
from src.embed.embedder import Embedder, EmbedderConfig
from src.ingest.heroes_html_ingest import extract_hero_docs_from_html
from src.chunking.chunker import _split_long_text  # или сделай публичную split-функцию
from src.chunking.chunker import chunk_hero_card

from src.ingest.html_ingest import extract_docs_from_html, HtmlDoc
from src.chunking.chunker import chunk_html_doc, split_long_text_by_chars  # chunk_html_doc уже добавлен ранее

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("index")

OUT_EMB = INDEX_DIR / "embeddings.npy"
OUT_PTR = INDEX_DIR / "pointers.jsonl"

CATEGORY_BY_FILENAME = {
    "general_changes.pdf": "general",
    "heroes_changes.pdf": "heroes",
    "items_changes.pdf": "items",
    "neutral_creeps_changes.pdf": "neutral_creeps",
    "neutral_items_changes.pdf": "neutral_items",
}


def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([p for p in RAW_DIR.glob("*.pdf") if p.name in CATEGORY_BY_FILENAME])
    if not pdf_files:
        raise FileNotFoundError(f"No known PDFs in {RAW_DIR}. Found: {[p.name for p in RAW_DIR.glob('*.pdf')]}")

    ch_cfg = ChunkerConfig(max_chars=1200, overlap_chars=200, min_chars=30)

    e_cfg = EmbedderConfig(
        model_name="google/embeddinggemma-300m",
        max_length=512,
        batch_size=16,
        normalize=True,
        device=None,
    )
    embedder = Embedder(e_cfg)

    all_texts: List[str] = []
    pointers: List[Dict[str, Any]] = []

    print("[build] extracting -> chunking -> collecting texts ...")

    for pdf_path in pdf_files:
        category = CATEGORY_BY_FILENAME[pdf_path.name]
        print(f"  - {pdf_path.name} ({category})")

        # --- HTML-first: если есть соответствующий HTML, используем его ---
        html_name = f"{category}_changes.html"
        html_path = RAW_DIR / html_name
        if html_path.exists():
            print(f"    using HTML source: {html_name}")
            docs = extract_docs_from_html(html_path, category=category)
            print(f"    found {len(docs)} docs in HTML")

            for doc in docs:
                # chunk_html_doc возвращает список чанков для одного логического документа
                chunks = chunk_html_doc(doc.title, doc.text, ch_cfg)
                for ci, ch in enumerate(chunks):
                    all_texts.append(ch)
                    pointers.append({
                        "category": category,
                        "source": "html",
                        "source_file": html_path.name,
                        "doc_slug": doc.slug,
                        "title": doc.title,
                        "chunk_index": ci,
                        "chunker": {
                            "max_chars": ch_cfg.max_chars,
                            "overlap_chars": ch_cfg.overlap_chars,
                            "min_chars": ch_cfg.min_chars,
                            "chunker_mode": "html_by_containers",
                        }
                    })
            # не обрабатываем PDF для этой категории
            continue

        # --- fallback: PDF processing (как раньше) ---
        for page_obj in extract_pages_from_pdf(pdf_path):
            page_num = page_obj["page"]
            text = page_obj["text"]

            if len(text) < 30:
                continue

            # Попытка PDF-чанкинга по заголовкам для героев/предметов уже реализована в chunk_text
            chunks = chunk_text(text, ch_cfg)

            if not chunks:
                continue

            for ci, ch in enumerate(chunks):
                all_texts.append(ch)

                pointers.append({
                    "category": category,
                    "source": "pdf",
                    "source_file": pdf_path.name,
                    "page": page_num,
                    "chunk_index": ci,
                    "chunker": {
                        "max_chars": ch_cfg.max_chars,
                        "overlap_chars": ch_cfg.overlap_chars,
                        "min_chars": ch_cfg.min_chars,
                        "chunker_mode": "default",
                    }
                })

    if not all_texts:
        raise RuntimeError("No texts extracted from PDFs. Check pdf parsing.")

    print(f"[build] total chunks: {len(all_texts)}")
    print("[embed] encoding chunks ...")
    embeddings = embedder.encode(all_texts).astype(np.float32)
    print(f"[embed] embeddings shape: {embeddings.shape}")

    print("[save] writing index ...")
    np.save(str(OUT_EMB), embeddings)

    with OUT_PTR.open("w", encoding="utf-8") as f:
        for p in pointers:
            write_jsonl_line(f, p)

    print(f"[done] {OUT_EMB}")
    print(f"[done] {OUT_PTR}")
    print("[note] pointers.jsonl does NOT contain text (as requested).")


if __name__ == "__main__":
    main()
