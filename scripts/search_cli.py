from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.ingest.pdf_ingest import extract_pages_from_pdf
from src.chunking.chunker import ChunkerConfig, chunk_text
from src.embed.embedder import Embedder, EmbedderConfig
from src.ingest.heroes_html_ingest import extract_hero_docs_from_html

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("index")

EMB_PATH = INDEX_DIR / "embeddings.npy"
PTR_PATH = INDEX_DIR / "pointers.jsonl"

_HERO_DOCS_CACHE = None


def _get_hero_docs():
    global _HERO_DOCS_CACHE
    if _HERO_DOCS_CACHE is None:
        from pathlib import Path
        html_path = Path("data/raw/heroes_changes.html")
        # Временно перенаправляем stdout чтобы не видеть отладочный вывод
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            _HERO_DOCS_CACHE = {d.hero_slug: d for d in extract_hero_docs_from_html(html_path)}
        finally:
            sys.stdout = old_stdout
    return _HERO_DOCS_CACHE


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


@lru_cache(maxsize=16)
def get_page_text(pdf_name: str, page_num: int) -> str:
    pdf_path = RAW_DIR / pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    for page_obj in extract_pages_from_pdf(pdf_path):
        if page_obj["page"] == page_num:
            return page_obj["text"]
    return ""


def cosine_topk(scores: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(top_k, scores.shape[0])
    idx = np.argpartition(-scores, kth=top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    idx = idx.astype(int)
    return idx


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.search_cli "query" [top_k] [min_score]')
        print('Example: python -m scripts.search_cli "Pudge" 5')
        print('Example: python -m scripts.search_cli "Blink Dagger" 10 0.55')
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    min_score = float(sys.argv[3]) if len(sys.argv) >= 4 else None

    if not EMB_PATH.exists() or not PTR_PATH.exists():
        raise FileNotFoundError("Index not found. Run: python -m scripts.build_embeddings_from_pdf")

    embeddings = np.load(str(EMB_PATH)).astype(np.float32)
    pointers = load_jsonl(PTR_PATH)

    if len(pointers) != embeddings.shape[0]:
        raise RuntimeError("Pointers count != embeddings count. Rebuild index.")

    e_cfg = EmbedderConfig(
        model_name="google/embeddinggemma-300m",
        max_length=256,
        batch_size=1,
        normalize=True,
        device=None,
    )
    embedder = Embedder(e_cfg)

    q_emb = embedder.encode([query])[0]
    scores = embeddings @ q_emb

    if min_score is not None:
        idx = np.where(scores >= min_score)[0]
        idx = idx[np.argsort(-scores[idx])]
        idx = idx[:top_k].astype(int)
    else:
        idx = cosine_topk(scores, top_k)

    print(f'Query: "{query}" | top_k={top_k}' + (f" | min_score={min_score}" if min_score is not None else ""))
    print("-" * 100)

    for rank, i in enumerate(idx, start=1):
        i = int(i)
        p = pointers[i]
        score = float(scores[i])

        pdf_name = p["source_file"]
        chunk_index = int(p["chunk_index"])

        # Инициализируем переменную для текста чанка
        chunk_text_value = ""

        if p.get("category") == "heroes" and p.get("source") == "html":
            hero_docs = _get_hero_docs()
            doc = hero_docs.get(p.get("hero_slug"))
            if doc is None:
                chunk_text_value = "[hero not found in html]"
            else:
                # Для героев используем chunk_hero_card для получения чанков
                from src.chunking.chunker import chunk_hero_card
                chunks = chunk_hero_card(doc.text)
                chunk_index = p.get("chunk_index", 0)

                if 0 <= chunk_index < len(chunks):
                    chunk_text_value = chunks[chunk_index]
                else:
                    chunk_text_value = f"[chunk not found for hero: {doc.hero_name}]"
        else:
            # старый путь (PDF)
            if "page" in p:
                page_num = p["page"]
                page_text = get_page_text(pdf_name, page_num)

                ch = p.get("chunker", {})
                ch_cfg = ChunkerConfig(
                    max_chars=int(ch.get("max_chars", 1200)),
                    overlap_chars=int(ch.get("overlap_chars", 200)),
                    min_chars=int(ch.get("min_chars", 30)),
                )

                chunks = chunk_text(page_text, ch_cfg)

                if 0 <= chunk_index < len(chunks):
                    chunk_text_value = chunks[chunk_index]
                else:
                    chunk_text_value = f"[chunk not found - chunking mismatch: got {chunk_index}, chunks={len(chunks)}]"
            else:
                chunk_text_value = "[page number not found]"

        preview = chunk_text_value.replace("\n", " ")
        if len(preview) > 320:
            preview = preview[:320] + "..."

        # Для красивого вывода информации о странице
        page_info = ""
        if "page" in p:
            page_info = f" | page={p['page']}"

        print(f"{rank}) score={score:.4f} | {p['category']} | {pdf_name}{page_info} | chunk={chunk_index}")
        print(preview)
        print("-" * 100)


if __name__ == "__main__":
    main()
