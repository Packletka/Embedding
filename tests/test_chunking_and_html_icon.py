# tests/test_chunking_and_html_icon.py
from __future__ import annotations

from pathlib import Path

from src.chunking.chunker import split_long_text_by_chars, normalize_for_chunking
from src.ingest.html_ingest import extract_docs_from_html

DATA_DIR = Path("data/raw")


def _reconstruct_from_parts(parts: list[str], overlap: int) -> str:
    """
    Попытка корректно склеить части, учитывая overlap_chars.
    Алгоритм: берём первую часть, затем для каждой следующей ищем максимальное перекрытие
    (до overlap) между концом текущей и началом следующей и добавляем неперекрывающуюся часть.
    """
    if not parts:
        return ""
    cur = parts[0]
    for nxt in parts[1:]:
        # ищем максимальную длину перекрытия (от overlap до 1)
        max_ov = min(len(cur), len(nxt), overlap)
        found = 0
        for k in range(max_ov, 0, -1):
            if cur.endswith(nxt[:k]):
                found = k
                break
        cur = cur + nxt[found:]
    # нормализуем результат так же, как исходный текст нормализуется перед разбиением
    return normalize_for_chunking(cur)


def test_split_long_text_by_chars_reconstructs():
    # сформируем длинный текст с повторяющимися словами
    words = ["слово"] * 200 + ["раз"] * 150 + ["деталь"] * 120
    text = " ".join(words)
    # нормализуем исходный текст так же, как делает chunker
    norm = normalize_for_chunking(text)

    max_chars = 200
    overlap = 50

    parts = split_long_text_by_chars(norm, max_chars=max_chars, overlap_chars=overlap)

    # базовые проверки
    assert isinstance(parts, list)
    assert len(parts) >= 2

    # Попытка реконструкции
    recon = _reconstruct_from_parts(parts, overlap)
    # Нормализуем реконструированный текст и сравниваем с исходной нормализацией
    assert normalize_for_chunking(recon) == norm


def test_extract_icon_url_from_html_items():
    # Проверяем на items_changes.html и neutral_items_changes.html (если они есть)
    for fname in ("items_changes.html", "neutral_items_changes.html", "neutral_creeps_changes.html"):
        path = DATA_DIR / fname
        if not path.exists():
            # если файл отсутствует — пропускаем тест для него
            continue

        docs = extract_docs_from_html(path, category="items" if "items" in fname and "neutral" not in fname else
        ("neutral_items" if "neutral_items" in fname else "neutral_creeps"))
        assert isinstance(docs, list)
        assert len(docs) > 0

        # хотя бы у одного документа должен быть icon_url с ожидаемым фрагментом
        found_icon = False
        for d in docs:
            # icon_url может быть None, но ожидаем, что хотя бы у одного документа он присутствует
            if getattr(d, "icon_url", None):
                url = d.icon_url
                # ожидаем, что URL содержит /items/ или /units/
                if "/items/" in url or "/units/" in url:
                    found_icon = True
                    break

        assert found_icon, f"No icon_url with /items/ or /units/ found in {fname}"
