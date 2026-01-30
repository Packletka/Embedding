from __future__ import annotations

from src.chunking.chunker import ChunkerConfig, chunk_html_doc, split_long_text_by_chars


def test_chunk_html_doc_short():
    title = "Healing Salve"
    text = "Начальный запас в лавке увеличен с 4 до 5 штук\nСтоимость увеличена с 50 до 60 золота"
    cfg = ChunkerConfig(max_chars=1200, overlap_chars=200, min_chars=10)

    chunks = chunk_html_doc(title, text, cfg)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert title in chunks[0]
    assert "Начальный запас" in chunks[0]


def test_chunk_html_doc_long_splits_and_prepends_title():
    title = "Very Long Item"
    # сформируем длинный текст > 500 символов
    paragraph = "Описание. " + ("Деталь " * 100)  # примерно > 500 символов
    text = "\n\n".join([paragraph for _ in range(3)])  # ещё длиннее

    cfg = ChunkerConfig(max_chars=200, overlap_chars=50, min_chars=20)
    chunks = chunk_html_doc(title, text, cfg)

    # Ожидаем, что текст разбился на несколько чанков
    assert isinstance(chunks, list)
    assert len(chunks) > 1

    # Каждый чанк должен начинаться с заголовка и содержать часть текста
    for ch in chunks:
        assert ch.startswith(title)
        # после заголовка должен быть перенос и текст
        assert "\n" in ch
        # проверим, что в чанке есть слово "Деталь"
        assert "Деталь" in ch


def test_split_long_text_by_chars_basic():
    text = " ".join(["word"] * 1000)
    parts = split_long_text_by_chars(text, max_chars=200, overlap_chars=50)
    assert isinstance(parts, list)
    assert len(parts) >= 4
    # все части не пустые
    assert all(p.strip() for p in parts)
