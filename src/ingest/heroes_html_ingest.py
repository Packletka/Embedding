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
    hero_slug: str  # "pudge"
    hero_name: str  # "Pudge"
    attribute: str  # "strength"
    text: str  # полный текст карточки героя


def extract_hero_docs_from_html(html_path: str | Path) -> List[HeroDoc]:
    """
    Парсим heroes_changes.html и вытаскиваем docs по героям.
    """
    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    docs: List[HeroDoc] = []

    # ПРОБЛЕМА: Мы ищем по портретам, но возможно структура другая
    # Давайте сначала найдем ВСЕ элементы, которые содержат иконки атрибутов
    attr_elements = []

    # Ищем все изображения с иконками атрибутов
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if any(key in src for key in _ATTR_MAP.keys()):
            attr_elements.append(img)

    # Теперь для каждой иконки атрибута находим контейнер героя
    processed_slugs = set()

    for i, attr_icon in enumerate(attr_elements):
        # Поднимаемся до контейнера, который содержит всю информацию о герое
        container = attr_icon.find_parent("div", recursive=True)

        # Поднимаемся еще на несколько уровней, чтобы захватить больше контекста
        for _ in range(3):
            if container and container.parent and container.parent.name == "div":
                container = container.parent

        if not container:
            continue

        # Ищем ссылку на героя внутри этого контейнера
        hero_link = None
        for a in container.find_all("a"):
            href = a.get("href", "")
            if "/hero/" in href:
                hero_link = a
                break

        if not hero_link:
            # Если не нашли, попробуем найти в ближайших соседях
            parent = container.parent
            if parent:
                for a in parent.find_all("a"):
                    href = a.get("href", "")
                    if "/hero/" in href:
                        hero_link = a
                        break

        if not hero_link:
            continue

        hero_slug = hero_link["href"].split("/hero/")[-1].strip("/")
        hero_name = hero_link.get_text(strip=True)

        # Проверяем, не обрабатывали ли мы уже этого героя
        if hero_slug in processed_slugs:
            continue

        # Определяем атрибут
        attribute = "unknown"
        src = attr_icon.get("src", "")
        for key, val in _ATTR_MAP.items():
            if key in src:
                attribute = val
                break

        # Извлекаем ВЕСЬ текст из контейнера
        # Сначала удаляем все скрипты и стили
        for script in container.find_all(["script", "style"]):
            script.decompose()

        # Получаем весь текст
        text = container.get_text(separator="\n", strip=True)

        # Очищаем текст от лишних пробелов и пустых строк
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        full_text = "\n".join(lines)

        # Для отладки выведем первые 100 символов текста
        preview = full_text[:100] + "..." if len(full_text) > 100 else full_text

        # Если текст слишком короткий, возможно, мы не в том контейнере
        if len(full_text) < 20:
            # Пропускаем этого героя, возможно это дубликат или неправильный контейнер
            continue

        # Добавляем героя в список
        docs.append(HeroDoc(
            hero_slug=hero_slug,
            hero_name=hero_name,
            attribute=attribute,
            text=full_text,
        ))
        processed_slugs.add(hero_slug)

    print(f"  Total unique heroes found: {len(docs)}")

    # Если все еще не нашли героев, давайте попробуем другой подход
    if len(docs) == 0:

        # Альтернативный подход: ищем все div с несколькими классами
        all_divs = soup.find_all("div")

        # Ищем div, которые содержат и иконку атрибута, и ссылку на героя
        for div in all_divs:
            # Проверяем, есть ли внутри иконка атрибута
            has_attr = False
            for img in div.find_all("img"):
                src = img.get("src", "")
                if any(key in src for key in _ATTR_MAP.keys()):
                    has_attr = True
                    break

            if not has_attr:
                continue

            # Проверяем, есть ли ссылка на героя
            hero_link = div.find("a", href=lambda h: h and "/hero/" in h)
            if not hero_link:
                continue

            hero_slug = hero_link["href"].split("/hero/")[-1].strip("/")
            hero_name = hero_link.get_text(strip=True)

            # Извлекаем текст
            text = div.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            full_text = "\n".join(lines)

            if len(full_text) > 50:  # Более либеральный порог
                # Определяем атрибут
                attribute = "unknown"
                for img in div.find_all("img"):
                    src = img.get("src", "")
                    for key, val in _ATTR_MAP.items():
                        if key in src:
                            attribute = val
                            break
                    if attribute != "unknown":
                        break

                if hero_slug not in processed_slugs:
                    docs.append(HeroDoc(
                        hero_slug=hero_slug,
                        hero_name=hero_name,
                        attribute=attribute,
                        text=full_text,
                    ))
                    processed_slugs.add(hero_slug)

    print(f"  Total heroes found (including alternative): {len(docs)}")

    return docs
