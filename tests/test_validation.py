"""English news-diary validation replaces the obsolete JLPT learning tests."""
import json
from pathlib import Path

import pytest

from scripts.generate_sentence import parse_output, read_context, render_prompt


def _context():
    terms = ["satellite", "cabbage", "tourists"]
    return {"keywords": terms, "selected": [
        {"text": keyword, "publisher": "Example", "title": f"News about {keyword}", "url": f"https://example.test/{keyword}"}
        for keyword in terms
    ]}


def _entry():
    return ("I woke up to find a satellite parked next to my cabbage patch. "
            "A group of tourists gathered around the odd machine and asked me for directions. "
            "I offered them tea while we waited for the spacecraft to take off. "
            "Nobody believed what had happened, but my garden looked better than ever.")


def test_parse_diary_accepts_english_with_three_terms():
    assert parse_output(json.dumps({"text": _entry()}), _context()["keywords"]) == _entry()


def test_parse_diary_rejects_missing_keyword():
    with pytest.raises(ValueError, match="keywords"):
        parse_output(json.dumps({"text": _entry().replace("cabbage", "lettuce")}), _context()["keywords"])


def test_parse_diary_rejects_extra_fields():
    with pytest.raises(ValueError, match="only"):
        parse_output(json.dumps({"text": _entry(), "translation_en": "No"}), _context()["keywords"])


def test_read_context_rejects_unattributed_selection(tmp_path: Path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"keywords": ["satellite", "cabbage", "tourists"], "selected": []}))
    with pytest.raises(ValueError, match="three selected"):
        read_context(path)


def test_prompt_contains_grounded_keywords_and_sources():
    prompt = render_prompt(_context(), "{{NEWS_CONTEXT}} / {{PROMPT_SEED}}", "123")
    assert "cabbage" in prompt and '"publisher": "Example"' in prompt and "123" in prompt
