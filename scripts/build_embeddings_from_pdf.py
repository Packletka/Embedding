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

        # --- HEROES: берём из HTML, а не из PDF ---
        if category == "heroes":
            html_path = RAW_DIR / "heroes_changes.html"
            if not html_path.exists():
                raise FileNotFoundError(f"Missing {html_path}. Put heroes_changes.html into data/raw/")

            hero_docs = extract_hero_docs_from_html(html_path)
            # В блоке обработки героев, после извлечения docs:
            print(f"  - Found {len(hero_docs)} heroes in HTML")

            # В блоке обработки героев:
            for doc in hero_docs:
                # Используем специальную функцию чанкинга для героев
                # chunk_hero_card сохранит героя как один чанк целиком
                chunks = chunk_hero_card(doc.text)

                for ci, ch in enumerate(chunks):
                    all_texts.append(ch)

                    pointers.append({
                        "category": "heroes",
                        "source": "html",
                        "source_file": html_path.name,
                        "hero_slug": doc.hero_slug,
                        "hero_name": doc.hero_name,
                        "attribute": doc.attribute,
                        "chunk_index": ci,
                        "chunker": {
                            "max_chars": ch_cfg.max_chars,
                            "overlap_chars": ch_cfg.overlap_chars,
                            "min_chars": ch_cfg.min_chars,
                            "chunker_mode": "heroes_full_card",
                        }
                    })

            # ВАЖНО: не обрабатываем heroes PDF ниже
            continue

        for page_obj in extract_pages_from_pdf(pdf_path):
            page_num = page_obj["page"]
            text = page_obj["text"]

            if len(text) < 30:
                continue

            chunks = chunk_text(text, ch_cfg)

            if not chunks:
                continue

            for ci, ch in enumerate(chunks):
                all_texts.append(ch)

                pointers.append({
                    "category": category,
                    "source_file": pdf_path.name,
                    "page": page_num,
                    "chunk_index": ci,
                    "chunker": {
                        "max_chars": ch_cfg.max_chars,
                        "overlap_chars": ch_cfg.overlap_chars,
                        "min_chars": ch_cfg.min_chars,
                        "chunker_mode": "heroes_by_headers" if category == "heroes" else "default",
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
