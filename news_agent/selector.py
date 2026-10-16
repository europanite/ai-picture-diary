"""Choose high-quality, timely, distinct news concepts with semantic diversity."""
from __future__ import annotations

import itertools
import math
import random
from typing import Protocol

from news_agent.keywords import Keyword


class Encoder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> object: ...


def _distance(a: object, b: object) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return max(0.0, min(2.0, 1.0 - dot))


def select_keywords(candidates: list[Keyword], model: Encoder, *, seed: int, mode: str = "balanced", sample_limit: int = 30000) -> dict:
    """Rank specific, newsworthy terms FIRST; semantic diversity is secondary.

    - balanced: maximize 80% quality + 20% medium/high semantic distance.
    - max: maximize semantic distance among the eligible quality-filtered terms.
    - random: a reproducible random selection from the same eligible set.
    """
    if mode not in ("balanced", "max", "random"):
        raise ValueError(f"unsupported mode: {mode}")
    # Generic unigrams must never re-enter simply because they are far apart.
    qualified = [x for x in candidates if x.kind in {"named_entity", "proper_noun", "noun_phrase"} and x.specificity >= 2.0]
    qualified.sort(key=lambda x: (-x.relevance, x.text.casefold(), x.article_id))
    qualified = qualified[:90]
    if len({x.article_id for x in qualified}) < 3:
        raise ValueError("Fewer than three articles contain specific named entities or noun phrases")

    rng = random.Random(seed)
    encoded = model.encode([x.text for x in qualified], normalize_embeddings=True)
    vectors = [list(map(float, vec)) for vec in encoded]
    if len(vectors) != len(qualified) or not vectors or not all(math.isfinite(x) for row in vectors for x in row):
        raise ValueError("invalid embedding vectors")

    must_cross_sources = len({x.publisher for x in qualified}) >= 2
    eligible = []
    for triple in itertools.combinations(range(len(qualified)), 3):
        picks = [qualified[i] for i in triple]
        if len({x.article_id for x in picks}) != 3:
            continue
        if must_cross_sources and len({x.publisher for x in picks}) < 2:
            continue
        tokens = [set(x.text.casefold().split()) for x in picks]
        if any(tokens[i] & tokens[j] for i, j in ((0, 1), (1, 2), (0, 2))):
            continue
        eligible.append(triple)
    if not eligible:
        raise ValueError("No valid cross-article combinations with specific concepts")
    if len(eligible) > sample_limit:
        eligible = rng.sample(eligible, sample_limit)

    scores = []
    for triple in eligible:
        a, b, c = triple
        distances = (_distance(vectors[a], vectors[b]), _distance(vectors[b], vectors[c]), _distance(vectors[c], vectors[a]))
        mean_distance = sum(distances) / 3
        mean_relevance = sum(qualified[i].relevance for i in triple) / 3
        # Prioritize time-sensitive specificity. Normalize quality only to make
        # the score interpretable; clip it to one for extraordinary entity names.
        quality = min(1.0, mean_relevance / 5.2)
        semantic = max(0.0, 1.0 - abs(mean_distance - 0.95) / 0.95)
        ranking = 0.8 * quality + 0.2 * semantic
        scores.append((ranking, mean_distance, mean_relevance, distances, triple))

    if mode == "random":
        choice = rng.choice(scores)
    elif mode == "max":
        choice = max(scores, key=lambda x: (x[1], x[2], x[4]))
    else:
        # First require near-best topical specificity/freshness. Semantic distance
        # can only break ties among similarly newsworthy triples.
        best_quality = max(x[2] for x in scores)
        topical = [x for x in scores if x[2] >= best_quality - 0.25]
        # Seed controls which near-optimal combination appears in the diary.
        high = sorted(topical, key=lambda x: (x[0], x[2], x[4]), reverse=True)
        cutoff = high[0][0] - 0.005
        finalists = [x for x in high if x[0] >= cutoff][:20]
        choice = rng.choice(finalists)

    rank, mean_distance, mean_relevance, distances, indices = choice
    selected = [qualified[i].to_dict() for i in indices]
    return {
        "keywords": [x["text"] for x in selected],
        "selected": selected,
        "selection": {
            "method": mode,
            "metric": "weighted_specificity_recency_and_cosine_distance",
            "rank_score": round(rank, 6),
            "mean_relevance": round(mean_relevance, 6),
            "mean_specificity": round(sum(x["specificity"] for x in selected) / 3, 6),
            "mean_freshness": round(sum(x["freshness"] for x in selected) / 3, 6),
            "mean_distance": round(mean_distance, 6),
            "pairwise_distances": [round(x, 6) for x in distances],
            "candidates": len(qualified),
            "eligible_triples_sampled": len(scores),
            "seed": seed,
        },
    }
