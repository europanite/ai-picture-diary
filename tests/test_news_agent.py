"""Offline news pipeline tests with a miniature POS/NER double."""
from datetime import datetime, timezone

import pytest

from news_agent.collector import Article, parse_feed
from news_agent.keywords import extract_keywords
from news_agent.selector import select_keywords

NOW = datetime(2026, 10, 16, 4, tzinfo=timezone.utc)
RSS = b'''<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Satellite crosses coastal sky</title><link>https://example.org/a</link><pubDate>Fri, 16 Oct 2026 02:00:00 GMT</pubDate></item>
<item><title>Old headline ignored</title><link>https://example.org/b</link><pubDate>Mon, 05 Oct 2026 02:00:00 GMT</pubDate></item>
</channel></rss>'''
ATOM = b'''<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Harvesters gather near cabbage farms</title><link href="https://example.org/c"/><updated>2026-10-16T00:00:00Z</updated></entry>
</feed>'''


class FakeToken:
    def __init__(self, text, i):
        self.text, self.i = text, i
        low = text.lower()
        if low in {"should", "fled", "investigates", "crosses", "moves", "surprise", "surprises", "explore", "gather", "visit", "visits", "carries", "grows", "strikes", "continue", "launches", "warns"}:
            self.pos_ = "VERB" if low != "should" else "AUX"
        elif low in {"above", "near", "in", "from", "as", "on", "at"}:
            self.pos_ = "ADP"
        elif low in {"a", "the"}:
            self.pos_ = "DET"
        elif low in {"tiny", "coastal", "ancient", "distant", "mysterious", "massive"}:
            self.pos_ = "ADJ"
        elif low in {"fbi", "texas", "london", "npr", "starship", "nvidia", "antarctic"}:
            self.pos_ = "PROPN"
        else:
            self.pos_ = "NOUN"
        self.is_stop = low in {"a", "the", "as", "in", "from", "on", "at", "should"}
        self.is_punct = self.is_space = False


class FakeSpan(list):
    def __init__(self, tokens, label=""):
        super().__init__(tokens)
        self.label_ = label


class FakeDoc:
    def __init__(self, title):
        import re
        self.tokens = [FakeToken(word, i) for i, word in enumerate(re.findall(r"[A-Za-z]+", title))]
        self.ents = []
        self.noun_chunks = []
        chunk = []
        for token in self.tokens + [FakeToken("visits", len(self.tokens))]:
            if token.pos_ in {"ADJ", "NOUN", "PROPN"}:
                chunk.append(token)
            elif chunk:
                self.noun_chunks.append(FakeSpan(chunk))
                chunk = []
        for token in self.tokens:
            if token.pos_ == "PROPN":
                self.ents.append(FakeSpan([token], "ORG"))

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, idx):
        value = self.tokens[idx]
        return FakeSpan(value) if isinstance(idx, slice) else value


class FakeNLP:
    def pipe(self, titles, batch_size=32):
        return (FakeDoc(title) for title in titles)


class DeterministicEncoder:
    def encode(self, texts, *, normalize_embeddings):
        assert normalize_embeddings is True
        vectors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        return [vectors[i % 3] for i, _ in enumerate(texts)]


def extract(articles, **kwargs):
    return extract_keywords(articles, nlp=FakeNLP(), now=NOW, **kwargs)


def test_rss_recency_and_atom_parsing():
    a = parse_feed(RSS, "Example", now=NOW)
    b = parse_feed(ATOM, "Atom", now=NOW)
    assert len(a) == 1 and a[0].title == "Satellite crosses coastal sky"
    assert len(b) == 1 and b[0].publisher == "Atom"


def test_specificity_rejects_verbs_and_generic_phrases():
    articles = [
        Article("1", "Source A", "Damage should continue as FBI investigates", "https://example.org/1", NOW.isoformat()),
        Article("2", "Source B", "Starship rocket launches from Texas", "https://example.org/2", NOW.isoformat()),
        Article("3", "Source C", "Mysterious cabbage virus strikes farms", "https://example.org/3", NOW.isoformat()),
    ]
    candidates = extract(articles, per_article=6)
    terms = {c.text.lower() for c in candidates}
    assert "damage should" not in terms
    assert "fbi investigates" not in terms
    assert "fled" not in terms
    assert "fbi" in terms and "texas" in terms
    assert any(term.endswith("cabbage virus") for term in terms)
    assert all(c.kind in {"named_entity", "noun_phrase", "proper_noun"} for c in candidates)


def test_fresh_headline_beats_old_headline_other_things_equal():
    articles = [
        Article("old", "A", "Starship rocket launches", "https://example.org/a", "2026-10-14T04:00:00Z"),
        Article("new", "B", "Texas rocket launches", "https://example.org/b", "2026-10-16T03:00:00Z"),
    ]
    candidates = extract(articles, per_article=5)
    starship = next(c for c in candidates if c.text.lower() == "starship")
    texas = next(c for c in candidates if c.text.lower() == "texas")
    assert texas.freshness > starship.freshness
    assert texas.relevance > starship.relevance


def test_keyword_extraction_has_provenance():
    articles = [
        Article("1", "Source A", "Satellite moves above riverside farms", "https://example.org/1", NOW.isoformat()),
        Article("2", "Source B", "Cabbage prices surprise farmers", "https://example.org/2", NOW.isoformat()),
        Article("3", "Source C", "Tourists explore Antarctic observatory", "https://example.org/3", NOW.isoformat()),
    ]
    words = extract(articles, per_article=3)
    assert len(words) >= 3
    assert all(word.url.startswith("https://") for word in words)
    assert {x.publisher for x in words} == {"Source A", "Source B", "Source C"}


def test_selection_deterministic_cross_article_and_specific():
    articles = [
        Article("1", "A", "Satellite crosses coastal sky", "https://example.org/1", NOW.isoformat()),
        Article("2", "B", "Cabbage harvest surprises farmers", "https://example.org/2", NOW.isoformat()),
        Article("3", "C", "Tourists visit ancient lighthouse", "https://example.org/3", NOW.isoformat()),
    ]
    words = extract(articles, per_article=3)
    one = select_keywords(words, DeterministicEncoder(), seed=55)
    two = select_keywords(words, DeterministicEncoder(), seed=55)
    assert one == two
    assert len(one["keywords"]) == 3
    assert len({x["article_id"] for x in one["selected"]}) == 3
    assert len({x["publisher"] for x in one["selected"]}) >= 2
    assert all(x["specificity"] >= 2.0 for x in one["selected"])
    assert 0 <= one["selection"]["mean_distance"] <= 2


def test_insufficient_articles_fail():
    a = [Article("1", "A", "Satellite coastal sky", "https://example.org/1", NOW.isoformat())]
    with pytest.raises(ValueError, match="three"):
        select_keywords(extract(a), DeterministicEncoder(), seed=55)


def test_cli_offline_replay(tmp_path, monkeypatch, capsys):
    import json
    import sys
    import types
    from news_agent.cli import main
    articles = [
        Article("a", "Source A", "Satellite carries a tiny telescope", "https://example.org/a", NOW.isoformat()),
        Article("b", "Source B", "Cabbage harvest grows in valley", "https://example.org/b", NOW.isoformat()),
        Article("c", "Source C", "Tourists visit distant island", "https://example.org/c", NOW.isoformat()),
    ]
    fixture = tmp_path / "articles.json"
    fixture.write_text(json.dumps([a.to_dict() for a in articles]))
    output = tmp_path / "selection.json"
    mock_module = types.ModuleType("sentence_transformers")
    mock_module.SentenceTransformer = lambda name, device=None: DeterministicEncoder()
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_module)
    monkeypatch.setattr("news_agent.cli.load_nlp", lambda name: FakeNLP())
    assert main(["--articles-json", str(fixture), "--output", str(output), "--seed", "4"]) == 0
    selection = json.loads(output.read_text())
    assert len(selection["keywords"]) == 3
    assert selection["article_count"] == 3
    assert selection["nlp_model"] == "en_core_web_sm"
    assert "keywords" in capsys.readouterr().out


def test_quality_wins_over_high_embedding_distance():
    candidates = [
        # Fake candidate score relevance includes specificity + freshness.
        __import__("news_agent.keywords", fromlist=["Keyword"]).Keyword(
            f"NamedPlace{i}", str(i), f"Publisher {i}", "headline", f"https://example.org/{i}",
            NOW.isoformat(), 5.0 if i < 3 else 2.0, "named_entity", 3.5, 1.0,
        ) for i in range(6)
    ]
    selected = select_keywords(candidates, DeterministicEncoder(), seed=2, mode="balanced")
    assert all(item["relevance"] >= 5.0 for item in selected["selected"])

# These regressions match the problematic *real* Oct 2026 headline fragments.
# Fake tokens include character offsets and spaCy-style punctuation splitting.
class OffsetToken:
    def __init__(self, text, idx, i, pos="PROPN"):
        self.text, self.idx, self.i, self.pos_ = text, idx, i, pos
        self.is_punct = pos == "PUNCT"
        self.is_space = False
        self.is_stop = False


class OffsetSpan(list):
    def __init__(self, full_text, tokens, label="PERSON"):
        super().__init__(tokens)
        self.text = full_text
        self.label_ = label


def test_hyphenated_person_name_preserves_original_punctuation():
    from news_agent.keywords import _clean

    name = OffsetSpan("Jordon Steele-John", [
        OffsetToken("Jordon", 0, 0),
        OffsetToken("Steele", 7, 1),
        OffsetToken("-", 13, 2, "PUNCT"),
        OffsetToken("John", 14, 3),
    ])
    assert _clean(name, kind="named_entity") == "Jordon Steele-John"
    assert _clean(name, kind="proper_noun") == "Jordon Steele-John"


def test_headline_new_is_removed_without_breaking_geographic_names():
    from news_agent.keywords import _clean

    boilerplate = OffsetSpan("New White House", [
        OffsetToken("New", 0, 0, "ADJ"),
        OffsetToken("White", 4, 1),
        OffsetToken("House", 10, 2),
    ], "ORG")
    place = OffsetSpan("New York", [
        OffsetToken("New", 0, 0), OffsetToken("York", 4, 1),
    ], "GPE")
    publisher = OffsetSpan("New York Times", [
        OffsetToken("New", 0, 0), OffsetToken("York", 4, 1),
        OffsetToken("Times", 9, 2),
    ], "ORG")
    assert _clean(boilerplate, kind="named_entity") == "White House"
    assert _clean(place, kind="named_entity") == "New York"
    assert _clean(publisher, kind="named_entity") == "New York Times"


def test_generic_medical_phrase_is_rejected_but_concrete_noun_survives():
    from news_agent.keywords import _clean

    generic = OffsetSpan("surgery and cancer diagnosis", [
        OffsetToken("surgery", 0, 0, "NOUN"),
        OffsetToken("and", 8, 1, "CCONJ"),
        OffsetToken("cancer", 12, 2, "NOUN"),
        OffsetToken("diagnosis", 19, 3, "NOUN"),
    ])
    generic2 = OffsetSpan("cancer diagnosis", [
        OffsetToken("cancer", 0, 0, "NOUN"),
        OffsetToken("diagnosis", 7, 1, "NOUN"),
    ])
    concrete = OffsetSpan("cabbage virus", [
        OffsetToken("cabbage", 0, 0, "NOUN"),
        OffsetToken("virus", 8, 1, "NOUN"),
    ])
    assert _clean(generic) == ""
    assert _clean(generic2) == ""
    assert _clean(concrete) == "cabbage virus"


def test_trained_ner_label_and_headline_centrality_affect_ranking():
    from news_agent.keywords import extract_keywords

    class StubDoc:
        def __init__(self, headline):
            self.tokens = [OffsetToken("Pope", 0, 0), OffsetToken("pictured", 5, 1, "VERB"),
                           OffsetToken("Vatican", 14, 2)]
            self.ents = [OffsetSpan("Pope", [self.tokens[0]], "PERSON"),
                         OffsetSpan("Vatican", [self.tokens[2]], "GPE")]
            self.noun_chunks = []

        def __iter__(self):
            return iter(self.tokens)

        def __getitem__(self, index):
            return OffsetSpan(" ".join(t.text for t in self.tokens[index]), self.tokens[index]) \
                if isinstance(index, slice) else self.tokens[index]

    class StubNLP:
        def pipe(self, texts, batch_size=32):
            return [StubDoc(text) for text in texts]

    results = extract_keywords([
        Article("pope", "Source", "Pope pictured near Vatican", "https://example.org/p", NOW.isoformat()),
    ], nlp=StubNLP(), now=NOW, per_article=8)
    pope = next(x for x in results if x.text == "Pope")
    vatican = next(x for x in results if x.text == "Vatican")
    assert pope.entity_label == "PERSON"
    assert pope.title_position < vatican.title_position
    assert pope.relevance > vatican.relevance
