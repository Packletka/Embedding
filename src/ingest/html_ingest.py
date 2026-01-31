from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import re
import unicodedata

from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse


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
    t = unicodedata.normalize("NFC", title)
    # Remove soft hyphen / zero-width chars that break slugs
    t = t.replace("\u00ad", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
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


# --- Специфичные парсеры (general / heroes) ---------------------------------

_STOPWORDS_FACET = {
    "аспекты", "способности", "таланты", "новый", "сильно изменён", "сильно изменен",
    "слабо изменён", "слабо изменен", "изменён", "изменен", "удалён", "удален",
    "базовые атрибуты", "базовые атрибуты не изменились",
}


def _get_img_src(img: Tag) -> str:
    if not isinstance(img, Tag) or img.name != "img":
        return ""
    # src может быть в src / data-src / srcset (берём первый)
    src = (img.get("src") or img.get("data-src") or "").strip()
    if not src:
        srcset = (img.get("srcset") or "").strip()
        if srcset:
            src = srcset.split(",")[0].split()[0].strip()
    return src


def _is_hero_icon_img(img: Tag) -> bool:
    # В реальном HTML src может содержать query-string, может не оканчиваться на .png,
    # поэтому проверяем "содержит путь к heroes".
    src = _get_img_src(img)
    return bool(src) and "/dota_react/heroes/" in src


def _extract_hero_slug_from_img(img: Tag) -> Optional[str]:
    src = _get_img_src(img)
    if not src or "/dota_react/heroes/" not in src:
        return None
    try:
        # Берём только path (без ?query)
        p = urlparse(src).path
        stem = Path(p).stem
        return stem or None
    except Exception:
        return None


def _extract_hero_slug_from_href(href: str) -> Optional[str]:
    if not href:
        return None
    m = re.search(r"/hero/([^/?#]+)", href)
    return m.group(1) if m else None


def _find_hero_name(hero_block: Tag, hero_slug: str) -> str:
    # Prefer a text anchor that points to /hero/<slug> and has non-empty text.
    for a in hero_block.find_all("a", href=lambda h: h and f"/hero/{hero_slug}" in h):
        txt = (a.get_text(strip=True) or "").strip()
        if txt:
            return txt
    # Fallback: title-case slug
    return hero_slug.replace("-", " ").title()


def _find_hero_icon_url_in_block(hero_block: Tag, hero_slug: str) -> Optional[str]:
    # 1) img src
    img = hero_block.find("img", src=lambda s: s and f"/dota_react/heroes/{hero_slug}" in s)
    if img is not None:
        src = _get_img_src(img)
        return src or None

    # 2) background-image in style
    for el in hero_block.find_all(style=True):
        style = el.get("style", "") or ""
        if f"/dota_react/heroes/{hero_slug}" in style:
            m = re.search(r"url\((['\"]?)([^'\")]+)\1\)", style)
            if m:
                return m.group(2)
    return None


def _find_hero_block_from_seed(seed: Tag, hero_slug: str) -> Optional[Tag]:
    """
    Находит контейнер одного героя, начиная от seed (иконка героя или ссылка /hero/<slug>).

    Логика мягкая (чтобы не терять случаи типа:
    - hero имеет только таланты (Jakiro),
    - hero имеет только базовые изменения (Crystal Maiden),
    - hero имеет только аспекты (Queen of Pain)).

    Требования к candidate:
    - не должен быть "общим списком" (много hero icons/hero links),
    - должен содержать хотя бы один признак "внутри есть изменения" для конкретного героя:
        * заголовки "Способности"/"Аспекты"/"Таланты" (или английские аналоги),
        * ability icon (/dota_react/abilities/),
        * facet icon (/facets/),
        * bullet lines (div.eSxy...).
    """
    cur = seed.find_parent("div") if isinstance(seed, Tag) else None
    candidate = None
    depth = 0
    while cur is not None and isinstance(cur, Tag) and depth < 16:
        hero_icons_inside = cur.find_all("img", src=lambda s: s and "/dota_react/heroes/" in s)
        hero_links_inside = cur.find_all("a", href=lambda h: h and "/hero/" in h)

        # если явно "список героев" — прекращаем подниматься
        if len(hero_icons_inside) > 1 or len(hero_links_inside) > 6:
            break

        has_abilities_header = cur.find(
            string=lambda t: isinstance(t, str) and t.strip() in ("Способности", "Abilities")) is not None
        has_aspects_header = cur.find(
            string=lambda t: isinstance(t, str) and t.strip() in ("Аспекты", "Facets")) is not None
        has_talents_header = cur.find(
            string=lambda t: isinstance(t, str) and t.strip() in ("Таланты", "Talents")) is not None

        has_ability_icon = cur.find("img", src=lambda s: s and "/dota_react/abilities/" in s) is not None
        has_facet_icon = (cur.find(style=lambda s: s and "/facets/" in s) is not None) or (
                cur.find("img", src=lambda s: s and "/facets/" in s) is not None
        )
        has_bullets = cur.find(class_="eSxyZNZqYCF1Y3wTL5PaK") is not None

        if has_ability_icon or has_abilities_header or has_aspects_header or has_facet_icon or has_talents_header or has_bullets:
            candidate = cur
            # если внутри есть аспекты + способности (или ability icon) — это почти точно нужный блок
            if has_aspects_header and (has_abilities_header or has_ability_icon):
                return cur

        cur = cur.parent if isinstance(cur.parent, Tag) else None
        depth += 1

    return candidate


def _extract_general_section_docs(soup: BeautifulSoup, path: Path) -> List[HtmlDoc]:
    """
    В general_changes.html секции выглядят как повторяющиеся блоки с заголовком div._3WahKl...
    Возвращаем 1 HtmlDoc на секцию (не одну простыню).

    Важно: на странице есть и "корневой" заголовок (title страницы), и подкатегория с таким же названием
    ("Общие изменения"). Поэтому пропускаем именно заголовок страницы (по структуре), а не по тексту.
    """
    heading_divs = soup.find_all("div", class_=lambda c: c and "_3WahKl" in c)
    if not heading_divs:
        return []

    # Page/root title (заголовок страницы). На dota2.com он часто равен "Общие изменения",
    # но первый heading может быть "Captains Mode", поэтому не берём слепо heading_divs[0].
    preferred_root_titles = ["Общие изменения", "General Changes"]
    root_title = ""
    for cand in preferred_root_titles:
        if any(((hd.get_text(strip=True) or "").strip() == cand) for hd in heading_divs):
            root_title = cand
            break
    if not root_title:
        root_title = (heading_divs[0].get_text(strip=True) or "").strip() or "Общие изменения"

    # Пытаемся найти именно заголовок страницы: заголовок с root_title, чей родитель НЕ содержит пунктов eSxy...
    root_title_el: Tag | None = None
    for h in heading_divs:
        t = (h.get_text(strip=True) or "").strip()
        if t != root_title:
            continue
        parent = h.parent if isinstance(h.parent, Tag) else None
        if parent is None:
            continue
        has_lines = bool(parent.find(class_="eSxyZNZqYCF1Y3wTL5PaK"))
        if not has_lines:
            root_title_el = h
            break

    # Fallback: если не нашли по структуре, считаем корневым первый заголовок.
    if root_title_el is None and heading_divs and (heading_divs[0].get_text(strip=True) or "").strip() == root_title:
        root_title_el = heading_divs[0]

    docs: List[HtmlDoc] = []
    seen_slugs: set[str] = set()

    for h in heading_divs:
        # Skip only the page title element (not all sections with the same text)
        if root_title_el is not None and h is root_title_el:
            continue

        title = (h.get_text(strip=True) or "").strip()
        if not title:
            continue
        if len(title) > 80:
            continue

        section = h.parent if isinstance(h.parent, Tag) else None
        if section is None:
            continue

        # Пункты секции (обычно это div.eSxy...)
        lines: list[str] = []
        for ln in section.find_all(class_="eSxyZNZqYCF1Y3wTL5PaK"):
            t = (ln.get_text(" ", strip=True) or "").strip()
            t = re.sub(r"\s+", " ", t)
            if t:
                lines.append(t)

        body = "\n".join(lines) if lines else _get_text_from_container(section)
        text = _normalize_whitespace(f"{root_title}\n{title}\n{body}")

        slug = f"general__{_slugify(title)}" or f"general__sec-{len(docs)}"
        if slug in seen_slugs:
            k = 2
            while f"{slug}-{k}" in seen_slugs:
                k += 1
            slug = f"{slug}-{k}"
        seen_slugs.add(slug)

        docs.append(HtmlDoc(
            slug=slug,
            title=title,
            text=text,
            category="general",
            source_file=path.name,
            icon_url=None,
            position=len(docs),
            extra={"entity_type": "general_section", "context_path": f"General > {title}"}
        ))

    return docs


def _pick_facet_name_from_lines(lines: List[str]) -> Optional[str]:
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.lower() in _STOPWORDS_FACET:
            continue
        if re.search(r"[A-Za-z]", s) and len(s) <= 60:
            return s
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.lower() in _STOPWORDS_FACET:
            continue
        if 2 <= len(s) <= 80:
            return s
    return None


def _extract_hero_docs_hierarchical(soup: BeautifulSoup, path: Path) -> List[HtmlDoc]:
    """
    Парсер heroes_changes.html (иерархический):

    Возвращает документы 4 типов:
    - hero-aggregated doc: <hero_slug> (включает ВСЕ изменения героя: base + facets + abilities + talents)
    - ability docs: <hero>__ability__<ability>
    - facet docs: <hero>__facet__<facet>
    - talents doc: <hero>__talents

    Проблемы, которые закрываем:
    - герой с ONLY Talents (Jakiro) раньше не попадал в pointers.jsonl вообще
    - герой с ONLY Facets (Queen of Pain) попадал, но без hero-aggregated doc (искать по имени героя было хуже)
    """
    docs: List[HtmlDoc] = []
    seen_doc_slugs: set[str] = set()
    processed_blocks: set[int] = set()
    processed_hero_slugs: set[str] = set()

    # --- собрать seeds в порядке появления в HTML ---
    seeds: List[tuple[str, Tag, Optional[str]]] = []  # (hero_slug, seed_tag, icon_url_if_known)

    for img in soup.find_all("img"):
        if _is_hero_icon_img(img):
            hero_slug = _extract_hero_slug_from_img(img)
            if hero_slug:
                icon_url = _get_img_src(img) or None
                seeds.append((hero_slug, img, icon_url))

    for a in soup.find_all("a", href=True):
        hero_slug = _extract_hero_slug_from_href(a.get("href", ""))
        if hero_slug:
            seeds.append((hero_slug, a, None))

    def _dedup_lines(lines: List[str]) -> List[str]:
        seen = set()
        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if ln in seen:
                continue
            seen.add(ln)
            out.append(ln)
        return out

    def _descendant_of(tag: Tag, root: Optional[Tag]) -> bool:
        if root is None:
            return False
        return any(p is root for p in tag.parents)

    def _extract_section_root(hero_block: Tag, header_texts: tuple[str, ...]) -> Optional[Tag]:
        """Возвращает контейнер секции, который идёт сразу после заголовка (если распознаётся)."""
        hdr = hero_block.find(string=lambda t: isinstance(t, str) and t.strip() in header_texts)
        if not hdr:
            return None
        hdr_div = hdr.find_parent("div")
        if not hdr_div or not isinstance(hdr_div, Tag):
            return None
        sib = hdr_div.find_next_sibling("div")
        if sib and isinstance(sib, Tag):
            return sib
        return None

    for hero_slug, seed_tag, seed_icon_url in seeds:
        if not hero_slug or hero_slug in processed_hero_slugs:
            continue
        processed_hero_slugs.add(hero_slug)

        hero_block = _find_hero_block_from_seed(seed_tag, hero_slug)
        if hero_block is None:
            continue
        if id(hero_block) in processed_blocks:
            continue
        processed_blocks.add(id(hero_block))

        hero_name = _find_hero_name(hero_block, hero_slug)
        hero_icon_url = seed_icon_url or _find_hero_icon_url_in_block(hero_block, hero_slug)

        # --- определяем корни секций (для исключения из base) ---
        abilities_root = _extract_section_root(hero_block, ("Способности", "Abilities"))
        talents_root = _extract_section_root(hero_block, ("Таланты", "Talents"))
        # facets/aspects root иногда "сложнее", поэтому используем старую эвристику ниже

        # --- Abilities: собираем как отдельные docs + сохраним для aggregated ---
        ability_entries: List[tuple[str, str]] = []  # (ability_name, body)
        ability_docs: List[HtmlDoc] = []

        ability_imgs = hero_block.find_all("img", src=lambda s: s and "/dota_react/abilities/" in s)
        ability_seen: set[int] = set()
        for aimg in ability_imgs:
            cont = aimg.find_parent("div")
            chosen = None
            depth = 0
            while cont is not None and isinstance(cont, Tag) and depth < 14:
                if cont is hero_block:
                    break
                if cont.find(class_="eSxyZNZqYCF1Y3wTL5PaK") is not None:
                    txt = _get_text_from_container(cont)
                    lines = [ln for ln in txt.splitlines() if ln]
                    if len(lines) >= 2:
                        head = lines[0].strip()
                        if head and head.lower() not in _STOPWORDS_FACET:
                            if any(len(ln.strip()) > 8 for ln in lines[1:]):
                                chosen = cont
                                break
                cont = cont.parent if isinstance(cont.parent, Tag) else None
                depth += 1

            if chosen is None:
                continue
            if id(chosen) in ability_seen:
                continue
            ability_seen.add(id(chosen))

            ability_txt = _get_text_from_container(chosen)
            lines = [ln for ln in ability_txt.splitlines() if ln]
            if not lines:
                continue
            ability_name = lines[0].strip()
            body = "\n".join(lines[1:]).strip()

            if ability_name.lower() in _STOPWORDS_FACET:
                continue
            if not body:
                continue

            ability_entries.append((ability_name, body))

            ability_slug = f"{hero_slug}__ability__{_slugify(ability_name)}"
            if ability_slug in seen_doc_slugs:
                k = 2
                while f"{ability_slug}-{k}" in seen_doc_slugs:
                    k += 1
                ability_slug = f"{ability_slug}-{k}"
            seen_doc_slugs.add(ability_slug)

            final_text = _normalize_whitespace(f"Hero: {hero_name}\nAbility: {ability_name}\n{body}")

            ability_docs.append(HtmlDoc(
                slug=ability_slug,
                title=f"{hero_name} — {ability_name}",
                text=final_text,
                category="heroes",
                source_file=path.name,
                icon_url=_get_img_src(aimg) or None,
                position=-1,
                extra={"entity_type": "hero_ability", "entity_name": ability_name, "parent_slug": hero_slug,
                       "parent_name": hero_name}
            ))

        # --- Facets / Aspects: отдельные docs + aggregated ---
        facet_entries: List[tuple[str, List[str]]] = []  # (facet_name, change_lines)
        facet_docs: List[HtmlDoc] = []
        aspects_section = None

        aspects_header = hero_block.find(string=lambda t: isinstance(t, str) and t.strip() in ("Аспекты", "Facets"))
        if aspects_header:
            cur = aspects_header.find_parent("div")
            depth = 0
            while cur is not None and isinstance(cur, Tag) and depth < 12:
                if cur.find(string=lambda t: isinstance(t, str) and t.strip() in ("Способности", "Abilities")):
                    break
                if cur.find(style=lambda s: s and "/facets/" in s) or cur.find("img",
                                                                               src=lambda s: s and "/facets/" in s):
                    aspects_section = cur
                cur = cur.parent if isinstance(cur.parent, Tag) else None
                depth += 1

        if aspects_section:
            facet_icons: List[Tag] = []
            for el in aspects_section.find_all(True):
                style = el.get("style", "") or ""
                if "/facets/" in style:
                    facet_icons.append(el)
            facet_icons += aspects_section.find_all("img", src=lambda s: s and "/facets/" in s)

            facet_entry_seen: set[int] = set()
            for icon_el in facet_icons:
                cont = icon_el.find_parent("div") if isinstance(icon_el, Tag) else None
                chosen = None
                depth = 0
                while cont is not None and isinstance(cont, Tag) and depth < 12:
                    if cont.find(string=lambda t: isinstance(t, str) and t.strip() in ("Способности", "Abilities")):
                        break
                    txt = _get_text_from_container(cont)
                    lines = [ln for ln in txt.splitlines() if ln]
                    has_facet_name = _pick_facet_name_from_lines(lines) is not None
                    has_rus_line = any(re.search(r"[А-Яа-я]", ln) and len(ln) > 10 for ln in lines)
                    if has_facet_name and has_rus_line:
                        chosen = cont
                        break
                    cont = cont.parent if isinstance(cont.parent, Tag) else None
                    depth += 1

                if chosen is None:
                    continue
                if id(chosen) in facet_entry_seen:
                    continue
                facet_entry_seen.add(id(chosen))

                txt = _get_text_from_container(chosen)
                lines = [ln for ln in txt.splitlines() if ln]
                facet_name = _pick_facet_name_from_lines(lines)
                if not facet_name:
                    continue

                # вытащим "линии изменений" (без мусора)
                change_lines = []
                for ln in lines:
                    lns = ln.strip()
                    if not lns:
                        continue
                    low = lns.lower()
                    if low in _STOPWORDS_FACET:
                        continue
                    if lns == facet_name:
                        continue
                    if lns == hero_name:
                        continue
                    # пропускаем явные статусы
                    if low in (
                    "новый", "сильно изменён", "сильно изменен", "слабо изменён", "слабо изменен", "изменён", "изменен",
                    "удалён", "удален"):
                        continue
                    # оставляем "содержательные" строки
                    if re.search(r"[А-Яа-я0-9]", lns) and len(lns) >= 10:
                        change_lines.append(lns)
                change_lines = _dedup_lines(change_lines)

                facet_entries.append((facet_name, change_lines))

                facet_slug = f"{hero_slug}__facet__{_slugify(facet_name)}"
                if facet_slug in seen_doc_slugs:
                    k = 2
                    while f"{facet_slug}-{k}" in seen_doc_slugs:
                        k += 1
                    facet_slug = f"{facet_slug}-{k}"
                seen_doc_slugs.add(facet_slug)

                # В тексте facet-doc оставляем исходный txt (он полезен), но добавляем префиксы Hero/Facet
                facet_text = _normalize_whitespace(f"Hero: {hero_name}\nFacet: {facet_name}\n{txt}")

                facet_docs.append(HtmlDoc(
                    slug=facet_slug,
                    title=f"{hero_name} — Facet: {facet_name}",
                    text=facet_text,
                    category="heroes",
                    source_file=path.name,
                    icon_url=None,
                    position=-1,
                    extra={"entity_type": "hero_facet", "entity_name": facet_name, "parent_slug": hero_slug,
                           "parent_name": hero_name}
                ))

        # --- Talents ---
        talents_lines: List[str] = []
        if talents_root:
            for bd in talents_root.find_all(class_="eSxyZNZqYCF1Y3wTL5PaK"):
                line = bd.get_text(" ", strip=True)
                if line:
                    talents_lines.append(line)
        talents_lines = _dedup_lines(talents_lines)

        talents_doc: Optional[HtmlDoc] = None
        if talents_lines:
            talents_slug = f"{hero_slug}__talents"
            if talents_slug in seen_doc_slugs:
                k = 2
                while f"{talents_slug}-{k}" in seen_doc_slugs:
                    k += 1
                talents_slug = f"{talents_slug}-{k}"
            seen_doc_slugs.add(talents_slug)

            t_body = "\n".join(talents_lines).strip()
            t_text = _normalize_whitespace(f"Hero: {hero_name}\nTalents\n{t_body}")
            talents_doc = HtmlDoc(
                slug=talents_slug,
                title=f"{hero_name} — Talents",
                text=t_text,
                category="heroes",
                source_file=path.name,
                icon_url=None,
                position=-1,
                extra={"entity_type": "hero_talents", "entity_name": "talents", "parent_slug": hero_slug,
                       "parent_name": hero_name}
            )

        # --- Base lines (исключаем bullets из abilities/facets/talents) ---
        excluded_roots = [abilities_root, aspects_section, talents_root]
        bullet_divs = hero_block.find_all(class_="eSxyZNZqYCF1Y3wTL5PaK")
        base_lines: List[str] = []
        for bd in bullet_divs:
            if any(_descendant_of(bd, r) for r in excluded_roots if r is not None):
                continue
            line = bd.get_text(" ", strip=True)
            if line:
                base_lines.append(line)
        base_lines = _dedup_lines(base_lines)

        # --- Решение: создавать hero-aggregated doc если есть хоть какие-то изменения ---
        has_meaningful = bool(base_lines or ability_entries or facet_entries or talents_lines)
        if not has_meaningful:
            # пропускаем "пустых" героев, чтобы не засорять индекс
            continue

        # --- Hero aggregated doc ---
        agg_lines: List[str] = [f"Hero: {hero_name}"]

        if base_lines:
            agg_lines.append("Base:")
            agg_lines += [f"- {ln}" for ln in base_lines]

        if facet_entries:
            agg_lines.append("Facets:")
            for fname, flines in facet_entries:
                if flines:
                    agg_lines.append(f"- {fname}: " + "; ".join(flines))
                else:
                    agg_lines.append(f"- {fname}")

        if ability_entries:
            agg_lines.append("Abilities:")
            for aname, abody in ability_entries:
                # abody уже в \n, сделаем компактный one-liner для aggregate, чтобы не раздувать
                compact = " ".join([x.strip() for x in abody.splitlines() if x.strip()])
                agg_lines.append(f"- {aname}: {compact}")

        if talents_lines:
            agg_lines.append("Talents:")
            agg_lines += [f"- {ln}" for ln in talents_lines]

        hero_doc_text = _normalize_whitespace("\n".join(agg_lines).strip())

        base_slug = hero_slug
        if base_slug in seen_doc_slugs:
            k = 2
            while f"{base_slug}-{k}" in seen_doc_slugs:
                k += 1
            base_slug = f"{base_slug}-{k}"
        seen_doc_slugs.add(base_slug)

        # add hero doc first, then children docs (so pointer order is stable)
        docs.append(HtmlDoc(
            slug=base_slug,
            title=hero_name,
            text=hero_doc_text,
            category="heroes",
            source_file=path.name,
            icon_url=hero_icon_url,
            position=len(docs),
            extra={"entity_type": "hero", "entity_name": hero_name, "parent_slug": None, "parent_name": None}
        ))

        # --- append children in stable order: facets, abilities, talents ---
        for fd in facet_docs:
            docs.append(fd.__class__(**{**fd.__dict__, "position": len(docs)}))  # type: ignore

        for ad in ability_docs:
            docs.append(ad.__class__(**{**ad.__dict__, "position": len(docs)}))  # type: ignore

        if talents_doc is not None:
            # talents doc after abilities; can be before if you want
            docs.append(talents_doc.__class__(**{**talents_doc.__dict__, "position": len(docs)}))  # type: ignore

    return docs


def extract_docs_from_html(path: str | Path, category: str) -> List[HtmlDoc]:
    """
    Универсальный HTML-ингест для страниц патчноутов.
    Возвращает список HtmlDoc для заданной категории.
    category: 'heroes', 'items', 'neutral_items', 'neutral_creeps', 'general'
    """
    path = Path(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Fast paths for known layouts
    if category == "general":
        sec_docs = _extract_general_section_docs(soup, path)
        if sec_docs:
            print(f"  debug: produced_docs={len(sec_docs)} (general sections)")
            return sec_docs

    if category == "heroes":
        hero_docs = _extract_hero_docs_hierarchical(soup, path)
        if hero_docs:
            print(f"  debug: produced_docs={len(hero_docs)} (heroes hierarchical docs)")
            return hero_docs

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
