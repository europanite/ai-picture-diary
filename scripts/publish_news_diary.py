"""Validate and publish an English news-inspired diary to the existing feed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.generate_sentence import build_entry_payload, parse_output, read_context, story_keywords


def publish(*, draft_path: Path, selection_path: Path, public_dir: Path,
            tz_name: str = "Asia/Tokyo", now: datetime | None = None) -> dict:
    context = read_context(selection_path)
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if not isinstance(draft, dict) or not isinstance(draft.get("text"), str):
        raise ValueError("Draft must contain a text string")
    text = parse_output(json.dumps({"text": draft["text"]}), story_keywords(context))
    expected = build_entry_payload(context, text)
    if draft != expected:
        raise ValueError("Draft does not match the current news selection or its provenance")

    utc_now = now or datetime.now(timezone.utc)
    if utc_now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    utc_now = utc_now.astimezone(timezone.utc)
    local_now = utc_now.astimezone(ZoneInfo(tz_name))
    iso = utc_now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # Stable per-day identity: re-running the same draft will not flood the feed.
    identity = json.dumps({"text": text, "keywords": context["keywords"],
                           "links": expected["links"]}, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    feed_stem = f"feed_{local_now:%Y%m%d}_{digest}"
    entry = {
        "id": feed_stem,
        "kind": "news_diary",
        "date": local_now.date().isoformat(),
        "created_at": iso,
        "published_at": iso,
        "text": text,
        "tweet": text,
        "place": "",
        "avatar_image": "image/avatar/normal.png",
        "permalink": f"./posts/{feed_stem}/index.html",
        "fictional": True,
        "links": expected["links"],
        "keywords": expected["keywords"],
        "story_keywords": expected["story_keywords"],
        "source_articles": expected["source_articles"],
        "selection": expected["selection"],
    }
    feed_path = public_dir / "feed" / f"{feed_stem}.json"
    feed_path.parent.mkdir(parents=True, exist_ok=True)

    if feed_path.exists():
        old = json.loads(feed_path.read_text(encoding="utf-8"))
        old_items = old.get("items", []) if isinstance(old, dict) else old
        if not isinstance(old_items, list) or len(old_items) != 1 or not isinstance(old_items[0], dict):
            raise ValueError(f"Existing snapshot has unexpected format: {feed_path}")
        if (old_items[0].get("text") != text
                or old_items[0].get("keywords") != expected["keywords"]
                or old_items[0].get("source_articles") != expected["source_articles"]):
            raise ValueError(f"Snapshot ID collision or provenance mismatch: {feed_path}")
        created = False
    else:
        snapshot = {"updated_at": iso, "items": [entry]}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=feed_path.parent,
                                         prefix=".feed-", suffix=".json", delete=False) as fp:
            temp_path = Path(fp.name)
            json.dump(snapshot, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        os.replace(temp_path, feed_path)
        created = True

    # The indexer publishes latest.json as a page pointer and creates the
    # matching post HTML. Existing illustrated snapshots remain intact.
    subprocess.run([sys.executable, str(Path(__file__).with_name("build_feed_pages.py")),
                    "--public-dir", str(public_dir)], check=True)
    return {"id": feed_stem, "created": created, "feed": str(feed_path),
            "keywords": context["keywords"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Publish a validated fictional English news diary")
    p.add_argument("--draft", type=Path, default=Path("data/news/draft.json"))
    p.add_argument("--selection", type=Path, default=Path("data/news/selection.json"))
    p.add_argument("--public-dir", type=Path, default=Path("frontend/app/public"))
    p.add_argument("--timezone", default="Asia/Tokyo")
    args = p.parse_args(argv)
    try:
        result = publish(draft_path=args.draft, selection_path=args.selection,
                         public_dir=args.public_dir, tz_name=args.timezone)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"Diary publish failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
