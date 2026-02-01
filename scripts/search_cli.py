from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.chunking.chunker import ChunkerConfig, chunk_text, chunk_html_doc, chunk_hero_card
from src.embed.embedder import Embedder, EmbedderConfig
from src.ingest.html_ingest import extract_docs_from_html, HtmlDoc
from src.ingest.pdf_ingest import extract_pages_from_pdf

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("index")

EMB_PATH = INDEX_DIR / "embeddings.npy"
PTR_PATH = INDEX_DIR / "pointers.jsonl"


# ----------------------------
# Helpers
# ----------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
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
    if top_k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, kth=top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(int)


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _tokens(s: str) -> List[str]:
    s = _norm_text(s)
    toks = re.findall(r"[0-9a-zA-Zа-яА-Я_]+", s)
    out: List[str] = []
    for t in toks:
        out.extend([x for x in t.split("_") if x])
    return out


@contextmanager
def _silence_stdout() -> Iterable[None]:
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old


@lru_cache(maxsize=32)
def _load_html_docs_cached(category: str, source_file: str) -> Dict[str, HtmlDoc]:
    """Load and cache docs for (category, html_file). Silences parser debug output."""
    html_path = RAW_DIR / source_file
    if not html_path.exists():
        return {}
    with _silence_stdout():
        docs = extract_docs_from_html(html_path, category)
    return {d.slug: d for d in docs}


@lru_cache(maxsize=512)
def _chunk_doc_cached(
        category: str,
        doc_slug: str,
        title: str,
        text: str,
        max_chars: int,
        overlap_chars: int,
        min_chars: int,
) -> List[str]:
    cfg = ChunkerConfig(max_chars=max_chars, overlap_chars=overlap_chars, min_chars=min_chars)
    # ВАЖНО: всегда chunk_html_doc, чтобы совпадало с индексацией
    return chunk_html_doc(title, text, cfg)


def _get_chunk_preview(pointer: Dict[str, Any]) -> str:
    """Return the chunk text for a pointer (HTML or PDF)."""
    source = pointer.get("source")
    chunk_index = int(pointer.get("chunk_index", 0))

    if source == "html":
        category = pointer.get("category", "")
        source_file = pointer.get("source_file", "")
        doc_slug = pointer.get("doc_slug")
        if not doc_slug:
            return "[doc_slug not found in pointer]"

        docs_map = _load_html_docs_cached(category, source_file)
        doc = docs_map.get(doc_slug)
        if doc is None:
            return f"[doc not found in html: {doc_slug}]"

        ch = pointer.get("chunker", {}) or {}
        max_chars = int(ch.get("max_chars", 1200))
        overlap_chars = int(ch.get("overlap_chars", 200))
        min_chars = int(ch.get("min_chars", 30))

        chunks = _chunk_doc_cached(
            category,
            doc.slug,
            doc.title,
            doc.text,
            max_chars,
            overlap_chars,
            min_chars,
        )
        if 0 <= chunk_index < len(chunks):
            return chunks[chunk_index]
        return f"[chunk not found for doc_slug={doc_slug} idx={chunk_index} total={len(chunks)}]"

    # PDF fallback
    pdf_name = pointer.get("source_file", "")
    if "page" not in pointer:
        return "[page number not found]"
    page_num = int(pointer["page"])
    page_text = get_page_text(pdf_name, page_num)

    ch = pointer.get("chunker", {}) or {}
    cfg = ChunkerConfig(
        max_chars=int(ch.get("max_chars", 1200)),
        overlap_chars=int(ch.get("overlap_chars", 200)),
        min_chars=int(ch.get("min_chars", 30)),
    )
    chunks = chunk_text(page_text, cfg)
    if 0 <= chunk_index < len(chunks):
        return chunks[chunk_index]
    return f"[chunk not found - chunking mismatch: got {chunk_index}, chunks={len(chunks)}]"


# ----------------------------
# Router + Rerank
# ----------------------------


@dataclass(frozen=True)
class PointerMeta:
    category: str
    title_norm: str
    slug_norm: str
    title_tokens: Tuple[str, ...]
    slug_tokens: Tuple[str, ...]


def _build_pointer_meta(pointers: Sequence[Dict[str, Any]]) -> List[PointerMeta]:
    meta: List[PointerMeta] = []
    for p in pointers:
        title = str(p.get("title", ""))
        slug = str(p.get("doc_slug", ""))
        meta.append(
            PointerMeta(
                category=str(p.get("category", "")),
                title_norm=_norm_text(title),
                slug_norm=_norm_text(slug),
                title_tokens=tuple(_tokens(title)),
                slug_tokens=tuple(_tokens(slug)),
            )
        )
    return meta


def _build_category_centroids(
        embeddings: np.ndarray,
        pointers: Sequence[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    cat_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(pointers):
        cat_to_rows[str(p.get("category", ""))].append(i)

    centroids: Dict[str, np.ndarray] = {}
    for cat, rows in cat_to_rows.items():
        if not rows:
            continue
        v = embeddings[np.array(rows, dtype=int)].mean(axis=0)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        centroids[cat] = v.astype(np.float32)
    return centroids


def _lexical_route_scores(
        query: str,
        meta: Sequence[PointerMeta],
) -> Dict[str, float]:
    qn = _norm_text(query)
    qt = set(_tokens(query))
    if not qn:
        return {}

    scores: Dict[str, float] = defaultdict(float)
    for m in meta:
        cat = m.category
        if qn == m.title_norm:
            scores[cat] = max(scores[cat], 10.0)
            continue
        if qn == m.slug_norm:
            scores[cat] = max(scores[cat], 10.0)
            continue

        if len(qn) >= 4:
            if qn in m.title_norm:
                scores[cat] = max(scores[cat], 6.0)
            if qn in m.slug_norm:
                scores[cat] = max(scores[cat], 5.0)

        if qt:
            overlap = len(qt.intersection(m.title_tokens))
            overlap += 0.6 * len(qt.intersection(m.slug_tokens))
            if overlap > 0:
                scores[cat] = max(scores[cat], float(overlap))
    return dict(scores)


def route_categories(
        query: str,
        q_emb: np.ndarray,
        pointers: Sequence[Dict[str, Any]],
        meta: Sequence[PointerMeta],
        centroids: Dict[str, np.ndarray],
        top_categories: int = 2,
        debug: bool = False,
) -> List[str]:
    lex = _lexical_route_scores(query, meta)
    if lex:
        best_cat, best_val = max(lex.items(), key=lambda x: x[1])
        if best_val >= 6.0:
            if debug:
                print(f"[router] lexical strong -> {best_cat} (score={best_val:.2f})")
            return [best_cat]

    if not centroids:
        cats = sorted({str(p.get("category", "")) for p in pointers if p.get("category")})
        return cats

    cat_list = list(centroids.keys())
    sims = np.array([float(centroids[c] @ q_emb) for c in cat_list], dtype=np.float32)
    order = np.argsort(-sims)
    ordered = [(cat_list[int(i)], float(sims[int(i)])) for i in order]

    if debug:
        top_show = ", ".join([f"{c}:{s:.3f}" for c, s in ordered[: min(5, len(ordered))]])
        print(f"[router] centroid top: {top_show}")

    if len(ordered) == 1:
        return [ordered[0][0]]

    c1, s1 = ordered[0]
    c2, s2 = ordered[1]
    if (s1 - s2) >= 0.06:
        return [c1]
    return [c for c, _ in ordered[: max(1, top_categories)]]


def _rerank_score(
        base_score: float,
        query: str,
        m: PointerMeta,
        preferred_cat: Optional[str] = None,
) -> float:
    qn = _norm_text(query)
    qt = set(_tokens(query))

    bonus = 0.0
    if preferred_cat and m.category == preferred_cat:
        bonus += 0.04

    if qn:
        if qn == m.title_norm:
            bonus += 0.35
        if qn == m.slug_norm:
            bonus += 0.35
        if len(qn) >= 4:
            if qn in m.title_norm:
                bonus += 0.18
            if qn in m.slug_norm:
                bonus += 0.12

    if qt:
        bonus += 0.06 * len(qt.intersection(m.title_tokens))
        bonus += 0.04 * len(qt.intersection(m.slug_tokens))

    if ("-" in qn) and ("-" in m.title_norm):
        q_parts = [x.strip() for x in qn.split("-") if x.strip()]
        t_parts = [x.strip() for x in m.title_norm.split("-") if x.strip()]
        if q_parts and t_parts and q_parts[0] == t_parts[0]:
            bonus += 0.06

    return float(base_score + bonus)


# ----------------------------
# Main
# ----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.search_cli",
        description="Semantic search over index/embeddings.npy + index/pointers.jsonl",
    )
    p.add_argument("query", help="Search query")
    p.add_argument("top_k", nargs="?", type=int, default=5, help="Number of results")
    p.add_argument("min_score_pos", nargs="?", type=float, default=None)

    p.add_argument("--min-score", type=float, default=None, help="Filter results by minimum cosine score")
    p.add_argument("--category", type=str, default=None, help='Force category (e.g. "heroes", "items", "general")')
    p.add_argument("--no-router", action="store_true", help="Disable query router and search all categories")
    p.add_argument("--top-categories", type=int, default=2, help="How many categories to search when router is unsure")
    p.add_argument("--candidate-mult", type=int, default=10, help="Retrieve top_k*candidate_mult before rerank")
    p.add_argument("--rerank", choices=["lexical", "off"], default="lexical", help="Rerank method")
    p.add_argument("--debug", action="store_true", help="Print router/rerank debug")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    query: str = args.query
    top_k: int = int(args.top_k)
    min_score = args.min_score if args.min_score is not None else args.min_score_pos

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

    meta = _build_pointer_meta(pointers)
    centroids = _build_category_centroids(embeddings, pointers)

    if args.category:
        selected_cats = [args.category]
        if args.debug:
            print(f"[router] forced category -> {selected_cats}")
    elif args.no_router:
        selected_cats = sorted({m.category for m in meta if m.category})
        if args.debug:
            print(f"[router] disabled -> searching all categories: {selected_cats}")
    else:
        selected_cats = route_categories(
            query,
            q_emb,
            pointers,
            meta,
            centroids,
            top_categories=max(1, int(args.top_categories)),
            debug=args.debug,
        )

    candidate_idxs = np.array([i for i, p in enumerate(pointers) if str(p.get("category", "")) in set(selected_cats)],
                              dtype=int)
    if candidate_idxs.size == 0:
        candidate_idxs = np.arange(len(pointers), dtype=int)
        selected_cats = sorted({m.category for m in meta if m.category})

    cand_emb = embeddings[candidate_idxs]
    cand_scores = cand_emb @ q_emb

    pool_k = max(top_k * int(args.candidate_mult), 50)
    pool_k = min(pool_k, cand_scores.shape[0])

    if min_score is not None:
        keep = np.where(cand_scores >= float(min_score))[0]
        keep = keep[np.argsort(-cand_scores[keep])]
        keep = keep[:pool_k]
        pool_local = keep.astype(int)
    else:
        pool_local = cosine_topk(cand_scores, pool_k)

    pool_global = candidate_idxs[pool_local]

    if args.rerank != "off":
        preferred = selected_cats[0] if selected_cats else None
        final = []
        for gi, li in zip(pool_global.tolist(), pool_local.tolist()):
            base = float(cand_scores[int(li)])
            final_score = _rerank_score(base, query, meta[int(gi)], preferred_cat=preferred)
            final.append((int(gi), base, final_score))
        final.sort(key=lambda x: x[2], reverse=True)
        final = final[:top_k]
    else:
        final = [(int(gi), float(cand_scores[int(li)]), float(cand_scores[int(li)])) for gi, li in
                 zip(pool_global, pool_local)][:top_k]

    header = f'Query: "{query}" | top_k={top_k}'
    if min_score is not None:
        header += f" | min_score={min_score}"
    header += f" | cats={','.join(selected_cats)}"
    print(header)
    print("-" * 100)

    for rank, (gi, base, final_score) in enumerate(final, start=1):
        p = pointers[gi]
        chunk_index = int(p.get("chunk_index", 0))

        chunk_text_value = _get_chunk_preview(p)
        preview = chunk_text_value.replace("\n", " ")
        if len(preview) > 320:
            preview = preview[:320] + "..."

        page_info = f" | page={p['page']}" if "page" in p else ""

        if args.debug and args.rerank != "off":
            score_str = f"score={final_score:.4f} (base={base:.4f})"
        else:
            score_str = f"score={final_score:.4f}"

        print(f"{rank}) {score_str} | {p.get('category')} | {p.get('source_file')}{page_info} | chunk={chunk_index}")
        print(preview)
        print("-" * 100)


if __name__ == "__main__":
    main()
