"""CLI: RSS -> headline candidates -> MiniLM embeddings -> selected keywords."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from news_agent.collector import Article, fetch_articles
from news_agent.keywords import extract_keywords, load_nlp
from news_agent.selector import select_keywords


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "news_agent" / "sources.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect English news keywords for AI picture diaries")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=Path("data/news/selection.json"))
    p.add_argument("--articles-json", type=Path, help="Read a saved list of articles instead of fetching RSS (offline replay)")
    p.add_argument("--mode", choices=("balanced", "max", "random"), default="balanced")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-age-hours", type=int, default=48)
    p.add_argument("--nlp-model", default="en_core_web_sm", help="Trained English NER/POS model")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args(argv)
    seed = args.seed if args.seed is not None else int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    if args.articles_json:
        articles = [Article(**x) for x in json.loads(args.articles_json.read_text(encoding="utf-8"))]
        errors: list[str] = []
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        articles, errors = fetch_articles(config["sources"], now=datetime.now(timezone.utc), max_age_hours=args.max_age_hours)
    if len(articles) < 3:
        raise SystemExit(f"Not enough fresh articles ({len(articles)}): {'; '.join(errors)}")
    candidates = extract_keywords(articles, nlp=load_nlp(args.nlp_model))
    if len(candidates) < 3:
        raise SystemExit("Not enough English keyword candidates")
    # Explicit failure is preferable to silently substituting lexical distance.
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.model, device="cpu")
    payload = select_keywords(candidates, encoder, seed=seed, mode=args.mode)
    payload.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": args.model,
        "nlp_model": args.nlp_model,
        "max_age_hours": args.max_age_hours,
        "article_count": len(articles),
        "source_count": len({a.publisher for a in articles}),
        "source_errors": errors,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"keywords": payload["keywords"], "mean_distance": payload["selection"]["mean_distance"], "mean_specificity": payload["selection"]["mean_specificity"], "mean_freshness": payload["selection"]["mean_freshness"], "output": str(args.output)}))
    for error in errors:
        print(f"RSS warning: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
