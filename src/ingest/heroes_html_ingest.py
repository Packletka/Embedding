from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup


_ATTR_MAP = {
    "hero_strength.png": "strength",
    "hero_agility.png": "agility",
    "hero_intelligence.png": "intelligence",
    "hero_universal.png": "universal",
}


@dataclass(frozen=True)
class HeroDoc:
    hero_slug: str  # "puck"
    hero_name: str  # "Puck"
    attribute: str  # "intelligence"
    text: str  # полный текст карточки героя (без мусора)


def _norm_ws(s: str) -> str:
    return " ".join(s.split()).strip()


def extract_hero_docs_from_html(html_path: str | Path) -> List[HeroDoc]:
    """
    Парсим heroes_changes.html и вытаскиваем docs по героям.
    Опираемся не на обфусцированные классы, а на устойчивые признаки:
    - ссылка вида /hero/<slug>
    - наличие иконки атрибута (hero_strength/agility/intelligence/universal)
    """
    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    docs: List[HeroDoc] = []

    # Внутри карточки героя есть ссылки /hero/<slug>.
    # Берём те <a>, где текст = имя героя (не пустой) и href начинается с /hero/
    hero_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        name = _norm_ws(a.get_text(" ", strip=True))  # Используем strip=False
        if href.startswith("/hero/") and name:
            # имя героя обычно чистое: "Puck", "Shadow Fiend"
            # (отсекаем слишком длинные)
            if 2 <= len(name) <= 40:
                hero_links.append(a)

    seen: set[str] = set()

    for a in hero_links:
        href = a["href"]  # /hero/puck
        hero_slug = href.split("/hero/")[-1].strip("/")

        # некоторые ссылки могут повторяться внутри карточки
        if not hero_slug or hero_slug in seen:
            continue

        # пытаемся подняться до контейнера карточки, где есть иконка атрибута и портрет героя
        container = a
        card = None
        for _ in range(10):
            container = container.parent
            if container is None or getattr(container, "name", None) is None:
                break
            if container.name != "div":
                continue

            imgs = container.find_all("img")
            srcs = [im.get("src", "") for im in imgs]

            has_attr_icon = any(any(key in s for key in _ATTR_MAP.keys()) for s in srcs)
            has_hero_portrait = any("/apps/dota2/images/dota_react/heroes/" in s for s in srcs)

            if has_attr_icon and has_hero_portrait:
                card = container
                break

        if card is None:
            continue

        # имя героя
        hero_name = _norm_ws(a.get_text(" ", strip=True))  # Используем strip=False

        # атрибут
        attribute = "unknown"
        for im in card.find_all("img"):
            src = im.get("src", "")
            for key, val in _ATTR_MAP.items():
                if key in src:
                    attribute = val
                    break
            if attribute != "unknown":
                break

        # текст карточки: берём все видимые строки внутри card
        # (они уже хорошо очищаются stripped_strings)
        parts = [p.strip() for p in card.stripped_strings if p.strip()]
        # иногда первое слово = имя героя уже есть, но это нормально
        text = _norm_ws("\n".join(parts))

        docs.append(HeroDoc(
            hero_slug=hero_slug,
            hero_name=hero_name,
            attribute=attribute,
            text=text,
        ))
        seen.add(hero_slug)

    return docs
