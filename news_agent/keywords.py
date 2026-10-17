"""Extract timely, specific English terms using a pretrained English NLP model.

Only named entities, proper nouns and noun phrases are eligible. Preserve the
exact identity of real-world names, remove headline boilerplate, and penalize
abstract news phrases. This is heuristic ranking, not a factuality classifier.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from news_agent.collector import Article

# Exclude broad topics, headline boilerplate, and nouns with little visual identity.
GENERIC = frozenset("""
    after amid ahead analysis announcement announces breaking business change changes
    claim claims comment comments concerns controversy crisis damage day days decision
    development developments event events fact facts fears future government governments
    group groups history impact impacts incident information issue issues latest leader
    leaders life local major matter meeting meetings member members month months morning
    nation national news number numbers official officials people person plan plans
    police problem problems progress question questions reason reasons report reports
    result results rise role roles situation source sources story stories support
    thing things time times today tomorrow update updates way ways week weeks year years
    world response says said say should could will would new first last next more most
    good bad big top main key own other several many another country countries city
    cities state states case cases action actions deal deals push effort efforts move
    moves warning warns investigation investigates talks talk hit hits amid live video
    watch latest confirms confirms reveal reveals reveals told know shows show
""".split())
ALLOWED_ENTITY_LABELS = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART",
    "LAW", "NORP", "LANGUAGE",
})
VALID_WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'&.-]*$")
HEADLINE_MODIFIERS = frozenset({
    "new", "latest", "former", "recent", "another", "more", "fresh",
    "major", "top", "first", "second", "third", "several",
})
# These are usually descriptions of events, not specific entities or objects.
# They remain legal inside an actual named entity.
ABSTRACT_HEADS = frozenset({
    "diagnosis", "diagnoses", "surgery", "surgeries", "scrutiny",
    "allegation", "allegations", "investigation", "investigations",
    "controversy", "concerns", "concern", "pressure", "misconduct",
    "crisis", "crises", "claims", "claim", "reaction", "reactions",
    "questions", "question", "debate", "debates", "announcement",
    "announcements", "response", "responses", "discussions",
})
ENTITY_BONUSES = {
    "EVENT": 0.4, "PRODUCT": 0.4, "WORK_OF_ART": 0.35,
    "PERSON": 0.3, "ORG": 0.3, "FAC": 0.25,
    "GPE": 0.15, "LOC": 0.15,
}


@dataclass(frozen=True)
class Keyword:
    text: str
    article_id: str
    publisher: str
    title: str
    url: str
    published_at: str | None
    relevance: float
    kind: str = "noun_phrase"
    specificity: float = 0.0
    freshness: float = 0.0
    entity_label: str | None = None
    title_position: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def load_nlp(model: str | None = None):
    """Require a real trained tagger/NER model; no misleading regex fallback."""
    model_name = model or os.getenv("NEWS_NLP_MODEL", "en_core_web_sm")
    try:
        import spacy
        nlp = spacy.load(model_name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"English NLP model '{model_name}' is unavailable. "
            "Build the news-agent Docker image or install it with "
            "'python -m spacy download en_core_web_sm'."
        ) from exc
    if "ner" not in nlp.pipe_names or "parser" not in nlp.pipe_names:
        raise RuntimeError(f"{model_name} needs pretrained named-entity recognition and parsing")
    return nlp


def _clean(span, *, kind: str = "noun_phrase") -> str:
    """Normalize names without losing hyphens, apostrophes, or true place names.

    spaCy may split a hyphenated name across tokens. Preserve the original
    token offsets instead of reconstructing it from lexical tokens alone.
    """
    tokens = [t for t in span if not t.is_space]
    while tokens and tokens[0].pos_ in {"DET", "ADP", "PRON"}:
        tokens.pop(0)
    while len(tokens) > 1 and tokens[0].text.casefold() in HEADLINE_MODIFIERS:
        first = tokens[0].text.casefold()
        if first == "new" and kind in {"named_entity", "proper_noun"} and tokens[0].pos_ == "PROPN":
            break
        tokens.pop(0)
    while tokens and tokens[-1].is_punct:
        tokens.pop()
    if not tokens:
        return ""
    lexical = [t for t in tokens if not t.is_punct]
    if not lexical or len(lexical) > 5:
        return ""
    if any(t.pos_ in {"VERB", "AUX", "ADV", "PRON"} for t in lexical):
        return ""
    if any(t.pos_ in {"CCONJ", "SCONJ"} for t in lexical):
        return ""
    if any(t.pos_ == "ADP" and not (kind == "named_entity" and t.text.casefold() == "of")
           for t in lexical):
        return ""
    if not any(t.pos_ in {"NOUN", "PROPN"} for t in lexical):
        return ""
    if lexical[-1].pos_ not in {"NOUN", "PROPN"}:
        return ""

    words = [t.text for t in lexical]
    if not all(VALID_WORD.fullmatch(t) for t in words):
        return ""
    if len(words) == 1 and len(words[0]) < 3:
        return ""
    if all(word.casefold() in GENERIC for word in words):
        return ""
    if kind != "named_entity":
        if words[-1].casefold() in GENERIC or words[-1].casefold() in ABSTRACT_HEADS:
            return ""
        if any(t.text.casefold() in ABSTRACT_HEADS for t in lexical) and not any(
            t.pos_ == "PROPN" for t in lexical
        ):
            return ""

    # Retain the original substring between the first and last accepted token.
    # This preserves genuine entity punctuation.
    original = getattr(span, "text", None)
    if original is not None and hasattr(tokens[0], "idx") and hasattr(tokens[-1], "idx"):
        start = tokens[0].idx - span[0].idx
        stop = tokens[-1].idx + len(tokens[-1].text) - span[0].idx
        phrase = original[start:stop]
    elif original is not None and tokens[0] is span[0] and tokens[-1] is span[-1]:
        phrase = original
    else:
        # Only used by small offline NLP test doubles without source offsets.
        phrase = " ".join(t.text for t in tokens)
    phrase = re.sub(r"(?<=\w)\s*[-‐‑–]\s*(?=\w)", "-", phrase)
    phrase = re.sub(r"(?<=\w)\s*['’]\s*(?=\w)", "'", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9'&. -]*[A-Za-z0-9]", phrase):
        return ""
    return phrase


def _freshness(published_at: str | None, now: datetime) -> float:
    if not published_at:
        return 0.0
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (now - published).total_seconds() / 3600)
    except ValueError:
        return 0.0
    # Smooth preference, not a hard cutoff. Date filtering occurs in collector.
    return math.exp(-age_hours / 24.0)


def _spans(doc) -> list[tuple[str, str, str | None, float]]:
    """Collect normalized candidates with NER labels and title prominence."""
    found: dict[str, tuple[str, str, str | None, float]] = {}
    priority = {"named_entity": 3, "proper_noun": 2, "noun_phrase": 1}
    title_tokens = max(1, len(list(doc)))

    def add(span, kind: str, label: str | None = None) -> None:
        phrase = _clean(span, kind=kind)
        if not phrase:
            return
        key = phrase.casefold()
        start_token = next((t.i for t in span if not t.is_punct and not t.is_space), 0)
        position = max(0.0, min(1.0, start_token / title_tokens))
        candidate = (phrase, kind, label, position)
        if key not in found or priority[kind] > priority[found[key][1]]:
            found[key] = candidate

    for entity in doc.ents:
        if entity.label_ in ALLOWED_ENTITY_LABELS:
            add(entity, "named_entity", entity.label_)
    for phrase in doc.noun_chunks:
        # Keep concrete multi-token noun phrases, never arbitrary bigrams.
        if len([t for t in phrase if t.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"}]) >= 2:
            add(phrase, "noun_phrase")
    # NER can miss proper names. Recover consecutive proper names (including
    # hyphenated names) rather than treating adjacent name tokens separately.
    sequence = list(doc)
    pos = 0
    while pos < len(sequence):
        token = sequence[pos]
        if token.pos_ != "PROPN" or token.is_stop:
            pos += 1
            continue
        end = pos + 1
        while end < len(sequence):
            next_token = sequence[end]
            if next_token.pos_ == "PROPN":
                end += 1
            elif (next_token.is_punct and next_token.text in {"-", "‐", "‑"}
                  and end + 1 < len(sequence) and sequence[end + 1].pos_ == "PROPN"):
                end += 2
            else:
                break
        if end - pos > 1:
            add(doc[sequence[pos].i:sequence[end - 1].i + 1], "proper_noun")
        for part in sequence[pos:end]:
            if part.pos_ == "PROPN" and not part.is_stop:
                add(doc[part.i:part.i + 1], "proper_noun")
        pos = end
    return list(found.values())


def extract_keywords(
    articles: list[Article], *, per_article: int = 5,
    max_candidates: int = 90, now: datetime | None = None, nlp=None,
) -> list[Keyword]:
    """Rank candidates by named-entity specificity, rarity, and article recency.

    High-scoring concepts are NOT a claim that the story is historically unique;
    specificity is a transparent heuristic, not a popularity or truth score.
    """
    if not articles:
        return []
    if nlp is None:
        nlp = load_nlp()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    docs = list(nlp.pipe([article.title for article in articles], batch_size=32))
    if len(docs) != len(articles):
        raise ValueError("NLP document count does not match articles")
    observed = [_spans(doc) for doc in docs]
    df = Counter(key for spans in observed for key in {text.casefold() for text, _, _, _ in spans})
    n = len(articles)
    all_options: list[Keyword] = []
    for article, spans in zip(articles, observed):
        fresh = _freshness(article.published_at, now)
        options: list[Keyword] = []
        for phrase, kind, label, title_position in spans:
            count = len(phrase.split())
            base = {"named_entity": 3.0 if count > 1 else 2.45,
                    "proper_noun": 2.35, "noun_phrase": 2.6}[kind]
            # More specific concepts earn an additional specificity score.
            specificity = base + min(0.6, max(0, count - 1) * 0.3)
            specificity += ENTITY_BONUSES.get(label or "", 0.0)
            # Title-leading subjects tend to be more central than background
            # descriptions buried at the end of a headline.
            centrality = 0.25 * (1.0 - title_position)
            rarity = math.log((n + 1) / (df[phrase.casefold()] + 0.5)) / math.log(n + 1)
            quality = specificity + 0.8 * max(0.0, rarity) + 0.9 * fresh + centrality
            options.append(Keyword(
                phrase, article.id, article.publisher, article.title, article.url,
                article.published_at, round(quality, 6), kind,
                round(specificity, 6), round(fresh, 6),
                label, round(title_position, 6),
            ))
        options.sort(key=lambda x: (-x.relevance, x.text.casefold(), x.article_id))
        all_options.extend(options[:per_article])

    # Deduplicate across publishers and rank GLOBALLY: old article order must
    # not consume the candidate budget before newer, more specific headlines.
    result: list[Keyword] = []
    seen: set[str] = set()
    for item in sorted(all_options, key=lambda x: (-x.relevance, x.text.casefold(), x.article_id)):
        if item.text.casefold() not in seen:
            result.append(item)
            seen.add(item.text.casefold())
        if len(result) >= max_candidates:
            break
    return result
