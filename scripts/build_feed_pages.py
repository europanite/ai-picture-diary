#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_items(path: Path) -> list[dict[str, Any]]:
    try:
        obj = load_json(path)
    except Exception:
        return []

    if isinstance(obj, dict):
        items = obj.get("items")
        if isinstance(items, list):
            return [it for it in items if isinstance(it, dict)]
        return [obj]

    if isinstance(obj, list):
        return [it for it in obj if isinstance(it, dict)]

    return []


def sort_key(it: dict[str, Any]) -> tuple[str, str]:
    primary = str(
        it.get("created_at")
        or it.get("generated_at")
        or it.get("published_at")
        or it.get("date")
        or ""
    )
    return (primary, str(it.get("id") or ""))


def to_public_image_url(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("./")
    return f"./{rel}"


def maybe_inject_image(item: dict[str, Any], feed_file: Path, public_dir: Path) -> dict[str, Any]:
    """
    - For single-post snapshot files named feed_*.json, the filename stem is canonical.
    - If item.image / item.image_url is missing, but public/image/<stem>.png exists,
      inject that image path.
    """
    out = dict(item)

    stem = feed_file.stem  # e.g. feed_20260422_123456
    if stem.startswith("feed_"):
        out["id"] = stem

    image = str(out.get("image") or "").strip()
    image_url = str(out.get("image_url") or "").strip()

    if not image and not image_url:
        for ext in ("png", "jpg", "jpeg", "webp"):
            candidate_rel = f"image/{stem}.{ext}"
            candidate_abs = public_dir / candidate_rel
            if candidate_abs.exists():
                injected = to_public_image_url(candidate_rel)
                out["image"] = injected
                out["image_url"] = injected
                break

    elif image and not image_url:
        out["image_url"] = image
    elif image_url and not image:
        out["image"] = image_url

    return out


def build_index(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(items),
        "items": items,
    }


def build_post_html(item: dict[str, Any]) -> str:
    text = html.escape(str(item.get("text") or item.get("tweet") or ""))
    keywords = [str(k) for k in item.get("keywords", []) if isinstance(k, str)]
    sources = item.get("source_articles", [])
    keyword_html = ", ".join(html.escape(k) for k in keywords)
    source_html = "".join(
        '<li><a href="' + html.escape(s["url"], quote=True) + '" rel="noopener noreferrer">'
        + html.escape(s.get("publisher", "Source") + ": " + s.get("title", "")) + '</a></li>'
        for s in sources
        if isinstance(s, dict) and str(s.get("url", "")).startswith(("http://", "https://"))
    )
    image = str(item.get("image") or item.get("image_url") or "").strip()

    image_html = f'<p><img src="{html.escape(image)}" alt="" style="max-width:100%;height:auto;" /></p>' if image else ""

    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        f"  <title>{text or 'Post'}</title>",
        "</head>",
        "<body>",
        "  <main>",
        f"    <p>{text}</p>" if text else "",
        image_html,
        f"    <p><strong>Keywords:</strong> {keyword_html}</p>" if keyword_html else "",
        f"    <section><h2>News inspiration</h2><ul>{source_html}</ul></section>" if source_html else "",
        "    <p><em>This diary is fictional. News headlines are creative prompts, not a factual narrative.</em></p>" if source_html else "",
        "  </main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(p for p in parts if p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", default="frontend/app/public")
    parser.add_argument("--base-url", default=".")
    parser.add_argument("--page-size", type=int, default=20)
    args = parser.parse_args()

    public_dir = Path(args.public_dir)
    feed_dir = public_dir / "feed"
    posts_dir = public_dir / "posts"
    latest_path = public_dir / "latest.json"

    feed_dir.mkdir(parents=True, exist_ok=True)
    posts_dir.mkdir(parents=True, exist_ok=True)

    feed_files = sorted(feed_dir.glob("feed_*.json"))
    all_items: list[dict[str, Any]] = []

    for feed_file in feed_files:
        raw_items = extract_items(feed_file)
        for item in raw_items:
            normalized = maybe_inject_image(item, feed_file, public_dir)

            post_id = str(normalized.get("id") or feed_file.stem)
            normalized["id"] = post_id
            normalized["permalink"] = f"./posts/{post_id}/index.html"

            all_items.append(normalized)

            post_dir = posts_dir / post_id
            post_dir.mkdir(parents=True, exist_ok=True)
            (post_dir / "index.html").write_text(
                build_post_html(normalized),
                encoding="utf-8",
            )

    all_items.sort(key=sort_key, reverse=True)

    page_size = max(1, int(args.page_size))
    total_pages = (len(all_items) + page_size - 1) // page_size if all_items else 1

    page_paths: list[str] = []
    for page_no in range(total_pages):
        start = page_no * page_size
        end = start + page_size
        page_items = all_items[start:end]

        page_name = f"page_{page_no:03d}.json"
        page_path = feed_dir / page_name
        page_paths.append(f"./feed/{page_name}")

        payload: dict[str, Any] = {
            "items": page_items,
            "count": len(page_items),
            "page": page_no,
            "page_size": page_size,
            "total_items": len(all_items),
            "total_pages": total_pages,
            "next_feed_url": f"./feed/page_{page_no + 1:03d}.json" if page_no + 1 < total_pages else "",
        }
        write_json(page_path, payload)

    index_payload = {
        "items": all_items,
        "count": len(all_items),
        "page_size": page_size,
        "total_pages": total_pages,
        "first_feed_url": "./feed/page_000.json" if page_paths else "",
        "pages": page_paths,
    }
    write_json(feed_dir / "index.json", index_payload)

    latest_payload = {
        "feed_url": "./feed/page_000.json" if page_paths else "./feed/index.json",
        "updated_at": all_items[0].get("created_at")
        or all_items[0].get("generated_at")
        or all_items[0].get("published_at")
        or all_items[0].get("date")
        if all_items
        else "",
    }
    write_json(latest_path, latest_payload)

    print(f"Wrote {len(all_items)} items from {len(feed_files)} feed snapshots")
    print(f"Wrote feed index: {feed_dir / 'index.json'}")
    print(f"Wrote latest pointer: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())