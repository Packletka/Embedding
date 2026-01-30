from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_HERO_HEADER_STOP = {
    "изменения",
    "источник",
    "способности",
    "таланты",
    "аспекты",
    "фасеты",
    "предметы",
    "нейтральные предметы",
    "нейтральные крипы",
}


def _is_hero_header_line(line: str) -> bool:
    """
    Эвристика: строка выглядит как заголовок героя (например: 'Puck', 'Primal Beast', 'Shadow Fiend').
    Без списка героев — только по форме строки.
    """
    s = line.strip()
    if not s:
        return False

    low = s.lower()

    # отсекаем служебные заголовки
    if low in _HERO_HEADER_STOP:
        return False

    # слишком длинно/коротко — маловероятно имя героя
    if len(s) < 2 or len(s) > 30:
        return False

    # заголовки умений часто с двоеточием: "Mana Burn:"
    if ":" in s:
        return False

    # если есть цифры — это не имя героя
    if any(ch.isdigit() for ch in s):
        return False

    # допускаем буквы, пробелы, апостроф, дефис
    # (учтём и латиницу, и кириллицу, т.к. имена героев в патче обычно латиницей)
    if not re.fullmatch(r"[A-Za-zА-Яа-я' -]+", s):
        return False

    # если это явно предложение (много пробелов) — скорее не заголовок
    # (у героев обычно 1–2 слова)
    if s.count(" ") >= 4:
        return False

    return True


def split_long_text_by_chars(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Разбивает text на фрагменты длиной <= max_chars с перекрытием overlap_chars.
    Сохраняет слова целиком (резать по пробелу, если возможно).
    """
    if not text:
        return []

    t = " ".join(text.split())
    n = len(t)
    if n <= max_chars:
        return [t]

    parts = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        # попытка сдвинуть конец к последнему пробел внутри окна
        if end < n:
            k = t.rfind(" ", start, end)
            if k > start:
                end = k
        part = t[start:end].strip()
        if part:
            parts.append(part)
        if end == n:
            break
        # следующий старт с учётом overlap
        start = max(0, end - overlap_chars)
    return parts


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """
    Режем длинный блок на подпорции по max_chars, стараясь резать по границе пробела.
    """
    t = " ".join(text.split())
    if not t:
        return []

    if len(t) <= max_chars:
        return [t]

    out = []
    i = 0
    while i < len(t):
        j = min(i + max_chars, len(t))
        if j < len(t):
            # попытка резать по пробелу
            k = t.rfind(" ", i, j)
            if k != -1 and k > i + max_chars * 0.6:
                j = k
        out.append(t[i:j].strip())
        i = j
    return [x for x in out if x]


def chunk_hero_card(text: str) -> List[str]:
    """
    Чанкинг для карточек героев из HTML.
    Каждого героя оставляем как один цельный чанк.
    """
    if not text:
        return []

    # Убираем лишние пробелы, но сохраняем структуру
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return []

    # Объединяем все строки в один чанк
    full_text = "\n".join(lines)
    return [full_text]


def chunk_heroes_by_headers(page_text: str, max_chars: int = 1200) -> List[str]:
    """
    Специальный чанкинг для heroes: сначала режем по заголовкам героев,
    затем каждый блок (если слишком длинный) режем по max_chars.
    """
    # Нормализуем переносы: pdfplumber может давать странные разрывы
    lines = [ln.strip() for ln in page_text.splitlines()]
    lines = [ln for ln in lines if ln]  # выкидываем пустые

    blocks: List[str] = []
    cur: List[str] = []

    for ln in lines:
        if _is_hero_header_line(ln):
            # встретили нового героя -> закрываем предыдущий блок
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
        cur.append(ln)

    if cur:
        blocks.append("\n".join(cur).strip())

    # теперь режем блоки по длине
    chunks: List[str] = []
    for b in blocks:
        chunks.extend(_split_long_text(b, max_chars=max_chars))

    return chunks


# --- новая функция chunk_html_doc ---
def chunk_html_doc(title: str, text: str, cfg: ChunkerConfig) -> List[str]:
    """
    Чанкинг для HTML-документа (одна карточка/секция).
    - Если текст короткий (<= cfg.max_chars) — возвращаем один чанк (title + текст или только текст).
    - Если текст длинный — разбиваем на фрагменты и в начало каждого фрагмента добавляем заголовок для контекста.
    """
    if not text:
        return []

    # Нормализуем текст (используем уже существующую функцию)
    full_text = normalize_for_chunking(text)
    if not full_text:
        return []

    # Если текст короткий — вернуть как есть (с заголовком в начале для контекста)
    if len(full_text) <= cfg.max_chars:
        # Если заголовок уже присутствует в тексте — не дублируем
        if title and title.strip() and title.strip() not in full_text.splitlines()[0]:
            return [f"{title}\n\n{full_text}"]
        return [full_text]

    # Длинный текст — разбиваем
    # Используем публичную split_long_text_by_chars (или внутреннюю _split_long_text)
    fragments = split_long_text_by_chars(full_text, cfg.max_chars, cfg.overlap_chars)

    out: List[str] = []
    for frag in fragments:
        frag = frag.strip()
        if not frag or len(frag) < cfg.min_chars:
            continue
        # Добавляем заголовок в начало каждого фрагмента для контекста
        if title and title.strip():
            chunk = f"{title}\n\n{frag}"
        else:
            chunk = frag
        out.append(chunk)

    return out


@dataclass
class ChunkerConfig:
    max_chars: int = 1200
    overlap_chars: int = 200
    min_chars: int = 30


def normalize_for_chunking(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    text = normalize_for_chunking(text)
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap_chars)

    return chunks


def chunk_text(text: str, cfg: ChunkerConfig) -> List[str]:
    paras = split_paragraphs(text)
    out: List[str] = []
    for p in paras:
        for c in split_long_text(p, cfg.max_chars, cfg.overlap_chars):
            if len(c) >= cfg.min_chars:
                out.append(c)
    return out
