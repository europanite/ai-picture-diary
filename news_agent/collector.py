"""Fetch recent RSS/Atom headlines without downloading full articles."""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class Article:
    id: str
    publisher: str
    title: str
    url: str
    published_at: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_markup(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.sub(r"\s+", " ", value).strip()


def _publication_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        date = parsedate_to_datetime(value)
    except (ValueError, TypeError, IndexError):
        try:
            date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.astimezone(timezone.utc)


def _text(element: ET.Element, names: list[str]) -> str:
    for name in names:
        result = element.find(name)
        if result is not None and result.text:
            return result.text.strip()
    return ""


def parse_feed(content: bytes, publisher: str, *, now: datetime, max_age_hours: int = 72) -> list[Article]:
    """Parse RSS 2.0 and Atom; skip items without dated, recent headlines."""
    root = ET.fromstring(content)
    atom = "{http://www.w3.org/2005/Atom}"
    items = root.findall(".//item")
    is_atom = False
    if not items:
        items = root.findall(f"{atom}entry")
        is_atom = True
    limit = now.astimezone(timezone.utc) - timedelta(hours=max_age_hours)
    output: list[Article] = []
    for node in items:
        if is_atom:
            title = _strip_markup(_text(node, [f"{atom}title"]))
            date_raw = _text(node, [f"{atom}published", f"{atom}updated"])
            link = ""
            for elm in node.findall(f"{atom}link"):
                if elm.get("rel", "alternate") == "alternate" and elm.get("href"):
                    link = str(elm.get("href"))
                    break
        else:
            title = _strip_markup(_text(node, ["title"]))
            date_raw = _text(node, ["pubDate", "{http://purl.org/dc/elements/1.1/}date"])
            link = _text(node, ["link"])
        date = _publication_date(date_raw)
        if not title or not link.startswith(("https://", "http://")) or date is None:
            continue
        if not (limit <= date <= now.astimezone(timezone.utc) + timedelta(minutes=15)):
            continue
        id_ = hashlib.sha256(link.encode("utf-8")).hexdigest()[:16]
        output.append(Article(id_, publisher, title, link, date.isoformat()))
    return output


def fetch_articles(sources: list[dict], *, now: datetime, max_age_hours: int = 72, timeout: int = 12) -> tuple[list[Article], list[str]]:
    """Continue across failing sources; require viable articles downstream."""
    output: dict[str, Article] = {}
    errors: list[str] = []
    for source in sources:
        name, url = str(source["name"]), str(source["url"])
        try:
            if not url.startswith("https://"):
                raise ValueError("RSS source must use HTTPS")
            req = Request(url, headers={"User-Agent": "AI-Picture-Diary-NewsAgent/1.0 (RSS headlines only)", "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"})
            with urlopen(req, timeout=timeout) as response:
                payload = response.read(1_500_001)
            if len(payload) > 1_500_000:
                raise ValueError("RSS response too large")
            articles = parse_feed(payload, name, now=now, max_age_hours=max_age_hours)
            for article in articles:
                output.setdefault(article.id, article)
        except (OSError, ValueError, ET.ParseError, KeyError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return sorted(output.values(), key=lambda a: (a.published_at or "", a.id), reverse=True), errors
