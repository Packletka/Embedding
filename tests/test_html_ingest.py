from __future__ import annotations

from pathlib import Path
import pytest

from src.ingest.html_ingest import extract_docs_from_html, HtmlDoc

DATA_DIR = Path("data/raw")


@pytest.mark.parametrize("fname,category,expected_title_fragment", [
    ("items_changes.html", "items", "Healing Salve"),
    ("neutral_items_changes.html", "neutral_items", "Ripper"),
    ("neutral_creeps_changes.html", "neutral_creeps", "Сатир"),
    ("general_changes.html", "general", "Общие изменения"),
    ("heroes_changes.html", "heroes", "Pudge"),  # optional if heroes html exists
])
def test_extract_docs_basic(fname, category, expected_title_fragment):
    path = DATA_DIR / fname
    if not path.exists():
        pytest.skip(f"Missing test file {path}")
    docs = extract_docs_from_html(path, category)
    assert isinstance(docs, list)
    assert len(docs) > 0, f"No docs extracted for {fname}"
    # at least one doc should contain expected fragment in title or text
    found = False
    for d in docs:
        assert isinstance(d, HtmlDoc)
        if expected_title_fragment.lower() in d.title.lower() or expected_title_fragment.lower() in d.text.lower():
            found = True
            break
    assert found, f"Expected fragment '{expected_title_fragment}' not found in any doc for {fname}"


def test_slug_uniqueness():
    path = DATA_DIR / "items_changes.html"
    if not path.exists():
        pytest.skip("Missing items_changes.html")
    docs = extract_docs_from_html(path, "items")
    slugs = [d.slug for d in docs]
    assert len(slugs) == len(set(slugs)), "Slugs are not unique"
