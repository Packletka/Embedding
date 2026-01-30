from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import re
import unicodedata

from bs4 import BeautifulSoup, Tag


# Doc модель
@dataclass(frozen=True)
class HtmlDoc:
    slug: str
    title: str
    text: str
    category: str
    source_file: str
    icon_url: Optional[str] = None
    position: Optional[int] = None
    extra: Dict[str, object] = None


# --- Утилиты -----------------------------------------------------------------

def _normalize_whitespace(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # strip each line and remove repeated empty lines
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _slugify(title: str) -> str:
    if not title:
        return ""
    # Normalize unicode
    t = unicodedata.normalize("NFKD", title)
    # Lowercase
    t = t.lower()
    # Replace apostrophes and similar with dash
    t = re.sub(r"[’'`]", "-", t)
    # Replace non-alnum with dash
    t = re.sub(r"[^a-z0-9а-яё\-]+", "-", t)
    # Collapse dashes
    t = re.sub(r"-{2,}", "-", t).strip("-")
    if not t:
        # fallback to ascii hex
        t = re.sub(r"[^a-z0-9\-]+", "", title.lower())
    return t


def _extract_icon_url_from_style(style: str) -> Optional[str]:
    if not style:
        return None
    m = re.search(r'url\(["\']?(https?://[^"\')]+)["\']?\)', style)
    if m:
        return m.group(1)
    return None


def _get_text_from_container(container: Tag) -> str:
    # remove scripts/styles
    for bad in container.find_all(["script", "style"]):
        bad.decompose()
    # get text with separator
    text = container.get_text(separator="\n", strip=True)
    return _normalize_whitespace(text)


# --- Селекторы и эвристики -----------------------------------------------

def _looks_like_item_container(el: Tag) -> bool:
    # Primary signal: style background-image with /items/ or img src with /items/
    if not isinstance(el, Tag):
        return False
    # check style attribute for background-image
    style = el.get("style", "")
    if "/items/" in style or "/units/" in style:
        return True
    # check any img inside
    for img in el.find_all("img"):
        src = img.get("src", "")
        if "/items/" in src or "/units/" in src:
            return True
    # fallback: has a short title child and several descriptive children
    # short title candidate: child with short text (1-5 words)
    title_found = False
    desc_count = 0
    for child in el.find_all(recursive=False):
        txt = (child.get_text(strip=True) or "")
        if txt and 1 <= len(txt.split()) <= 6:
            title_found = True
        if txt and len(txt.split()) > 6:
            desc_count += 1
    return title_found and desc_count >= 1


def _find_title_in_container(el: Tag) -> Optional[str]:
    # Common pattern in your HTML: a div with short text (class _1MKhq... etc)
    # Try several heuristics in order
    # 1) anchor to hero (/hero/)
    a = el.find("a", href=lambda h: h and "/hero/" in h)
    if a and a.get_text(strip=True):
        return a.get_text(strip=True)

    # 2) element with many short words (likely title)
    candidates = []
    for child in el.find_all():
        txt = (child.get_text(strip=True) or "")
        if not txt:
            continue
        words = txt.split()
        if 1 <= len(words) <= 6 and len(txt) <= 60:
            candidates.append((len(words), txt, child))
    if candidates:
        # prefer smallest word count then first occurrence
        candidates.sort(key=lambda x: (x[0], len(x[1])))
        return candidates[0][1]

    # 3) look for specific classes that often contain titles (best-effort)
    for cls in ["_1MKhqwpq2QdCgvldsg6djR", "_2SxiJz8zGSkJQerssvs-oZ", "Bold"]:
        el2 = el.find(class_=lambda c: c and cls in c)
        if el2 and el2.get_text(strip=True):
            return el2.get_text(strip=True)

    return None


# --- Основная функция извлечения -----------------------------------------

def extract_docs_from_html(path: str | Path, category: str) -> List[HtmlDoc]:
    """
    Универсальный HTML-ингест для страниц патчноутов.
    Возвращает список HtmlDoc для заданной категории.
    category: 'heroes', 'items', 'neutral_items', 'neutral_creeps', 'general'
    """
    path = Path(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    docs: List[HtmlDoc] = []
    seen_slugs = set()

    # Strategy:
    # 1) Найти все candidate containers: элементы, которые повторяются и выглядят как карточки
    # 2) Для каждого контейнера извлечь title, icon_url, text
    # 3) Нормализовать slug, избегать коллизий

    # Collect candidate containers: search for divs that contain background-image or img with /items/ or /units/
    candidates = []

    # fast pass: find elements with style containing /items/ or /units/
    for el in soup.find_all(style=True):
        style = el.get("style", "")
        if "/items/" in style or "/units/" in style:
            candidates.append(el)

    # also consider elements that contain img with /items/ or /units/
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if "/items/" in src or "/units/" in src:
            # prefer the nearest ancestor that looks like a card (div)
            parent = img.find_parent("div")
            if parent is not None:
                candidates.append(parent)

    # fallback: find repeating blocks that look like item containers
    if not candidates:
        for div in soup.find_all("div"):
            if _looks_like_item_container(div):
                candidates.append(div)

    # Deduplicate by identity
    unique_candidates = []
    seen_ids = set()
    for el in candidates:
        if not isinstance(el, Tag):
            continue
        ident = id(el)
        if ident in seen_ids:
            continue
        seen_ids.add(ident)
        unique_candidates.append(el)
    print(f"  debug: unique_candidates={len(unique_candidates)}")

    # If category == general, prefer section-like headings
    if category == "general":
        # find section headers (h1-h4 or divs with short bold text)
        # sections = []
        # h2/h3/h4
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            txt = h.get_text(strip=True)
            if txt and len(txt.split()) <= 6:
                # collect following siblings until next header
                block = []
                for sib in h.find_next_siblings():
                    # stop if next header
                    if sib.name in ["h1", "h2", "h3", "h4"]:
                        break
                    block.append(sib)
                container = Tag(name="div")
                for b in block:
                    container.append(b)
                title = txt
                text = _normalize_whitespace(
                    h.get_text(separator="\n", strip=True) + "\n" + container.get_text(separator="\n", strip=True))
                slug = _slugify(title)
                if slug in seen_slugs:
                    # add position suffix
                    suffix = 1
                    while f"{slug}-{suffix}" in seen_slugs:
                        suffix += 1
                    slug = f"{slug}-{suffix}"
                seen_slugs.add(slug)
                docs.append(HtmlDoc(
                    slug=slug,
                    title=title,
                    text=_normalize_whitespace(text),
                    category=category,
                    source_file=path.name,
                    icon_url=None,
                    position=len(docs),
                    extra={}
                ))
        # if we found sections, return them
        if docs:
            return docs

    # Process unique candidates
    processed_containers: set[int] = set()

    for pos, el in enumerate(unique_candidates):
        # try to find title
        title = _find_title_in_container(el)
        # try to find icon url
        icon_url = None
        style = el.get("style", "")
        icon_url = _extract_icon_url_from_style(style)
        if not icon_url:
            img = el.find("img")
            if img:
                icon_url = img.get("src")

        # extract full text from a reasonable parent container to include description blocks
        container = el
        for _ in range(2):
            parent = container.parent
            if parent and isinstance(parent, Tag):
                # collect direct children
                siblings = [c for c in parent.find_all(recursive=False) if isinstance(c, Tag)]
                # count how many children look like separate cards (have icon/img or look like item container)
                candidate_children = 0
                for c in siblings:
                    style = c.get("style", "") or ""
                    if "/items/" in style or "/units/" in style:
                        candidate_children += 1
                        continue
                    if c.find("img", src=lambda s: s and ("/items/" in s or "/units/" in s)):
                        candidate_children += 1
                        continue
                    if _looks_like_item_container(c):
                        candidate_children += 1

                # If parent contains multiple candidate children, it's likely a list of cards -> don't climb
                if candidate_children > 1:
                    break

                # otherwise climb (parent likely contains description pieces for a single card)
                container = parent
            else:
                break

        # dedupe by container to avoid processing the same parent multiple times
        container_id = id(container)
        if container_id in processed_containers:
            continue
        processed_containers.add(container_id)

        full_text = _get_text_from_container(container)

        # If title not found, try to infer from icon_url filename
        if not title and icon_url:
            # take last path component
            name = Path(icon_url).stem
            # replace underscores with spaces
            title = name.replace("_", " ").replace("-", " ").strip()

        if not title:
            # fallback: take first line of text as title if short
            first_line = full_text.splitlines()[0] if full_text else ""
            if first_line and len(first_line.split()) <= 8:
                title = first_line

        if not title:
            # skip if no title and no meaningful text
            if not full_text or len(full_text) < 20:
                continue
            title = full_text[:40].split("\n")[0]

        # normalize text
        full_text = _normalize_whitespace(full_text)

        # generate slug and avoid collisions
        base_slug = _slugify(title)
        slug = base_slug or f"doc-{pos}"
        if slug in seen_slugs:
            suffix = 1
            while f"{slug}-{suffix}" in seen_slugs:
                suffix += 1
            slug = f"{slug}-{suffix}"
        seen_slugs.add(slug)

        # minimal length guard
        if len(full_text) < 10:
            continue

        docs.append(HtmlDoc(
            slug=slug,
            title=title,
            text=full_text,
            category=category,
            source_file=path.name,
            icon_url=icon_url,
            position=pos,
            extra={}
        ))

    print(
        f"  debug: produced_docs={len(docs)} (unique containers processed={len(processed_containers) if 'processed_containers' in locals() else 0})")

    return docs
