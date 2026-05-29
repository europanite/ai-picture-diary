#!/usr/bin/env python3
"""
Generate an illustration image for the latest post and patch feed JSON(s) to reference it.

- Reads:  LATEST_PATH (default: frontend/app/public/latest.json)
- Writes: frontend/app/public/image/<feed_stem>.png
- Patches:
  - feed snapshot file (feed/feed_<...>.json) in BOTH shapes:
      * {"items":[...]} (legacy)
      * {date,text,...} (current single-object)

Supports BOTH latest.json shapes:
- entry object shape and pointer shape {"feed_url":"./feed/page_000.json","updated_at":"..."}
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import AutoPipelineForText2Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VLM_PROMPT_PATH = PROJECT_ROOT / "scripts" / "prompts" / "vlm" / "positive.txt"
DEFAULT_VLM_NEGATIVE_PROMPT_PATH = PROJECT_ROOT / "scripts" / "prompts" / "vlm" / "negative.txt"


def _resolve_repo_path(value: str | None) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_text_file(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _load_prompt_template(
    *,
    inline_env: str,
    path_envs: tuple[str, ...],
    default_path: Path,
) -> str:
    inline = os.environ.get(inline_env)
    if inline is not None and inline.strip():
        return inline.strip()

    for env_name in path_envs:
        path = _resolve_repo_path(os.environ.get(env_name))
        if path is not None:
            return _read_text_file(path)

    return _read_text_file(default_path)


MODEL_ID = os.environ.get("SDXL_MODEL_ID", "stabilityai/sdxl-turbo").strip()
LORA_PATH = os.environ.get("LORA_PATH", "").strip()
LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.8"))
PROMPT_TMPL = _load_prompt_template(
    inline_env="PROMPT",
    path_envs=("VLM_PROMPT_PATH",),
    default_path=DEFAULT_VLM_PROMPT_PATH,
)
NEGATIVE_TMPL = _load_prompt_template(
    inline_env="NEGATIVE",
    path_envs=("VLM_NEGATIVE_PROMPT_PATH", "NEGATIVE_PROMPT_PATH"),
    default_path=DEFAULT_VLM_NEGATIVE_PROMPT_PATH,
)
STEPS = int(os.environ.get("STEPS", "6"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "2.0"))
SEED_OVERRIDE = (os.environ.get("SEED"))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
DEVICE = "cpu"  # GitHub Actions runner is typically CPU for this job



def artifact_public_dir(*, latest_path: Path | None = None) -> Path:
    lp = Path(latest_path) if latest_path is not None else Path(
        os.environ.get("LATEST_PATH", "frontend/app/public/latest.json")
    )
    return lp.parent


def artifact_latest_path(public_dir: Path | None = None) -> Path:
    override = (os.environ.get("LATEST_PATH") or "").strip()
    if override:
        return Path(override)
    if public_dir is not None:
        return Path(public_dir) / "latest.json"
    return Path("frontend/app/public/latest.json")


def artifact_feed_dir(public_dir: Path | None = None) -> Path:
    if public_dir is not None:
        return Path(public_dir) / "feed"
    return artifact_public_dir() / "feed"

def artifact_image_dir(*, latest_path: Path | None = None) -> Path:
    override = (os.environ.get("OUT_DIR") or "").strip()
    if override:
        return Path(override)
    return artifact_public_dir(latest_path=latest_path) / "image"

def resolve_public_image_url(path: Path, *, latest_path: Path | None = None) -> str:
    public_dir = artifact_public_dir(latest_path=latest_path)
    try:
        return path.relative_to(public_dir).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_public_avatar_url(value: str, *, latest_path: Path | None = None) -> str:
    return (value or "").strip()

def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_str(x: Any) -> str:
    return str(x) if x is not None else ""

def _render(t: str, *, place: str, core: str) -> str:
    s = (t or "").strip()
    if not s:
        return ""
    return (
        s
        .replace("{core}", core)
        .replace("{place}", place)
    )

def _first_nonempty_line(text: str) -> str:
    for line in str(text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def extract_scene_text(text: str) -> str:
    """
    Convert the stored feed text into the single sentence that should be drawn.

    The current pipeline stores a multi-line block such as:
      うんこがきれいに見える。
      学習ポイント: ...
      英訳: ...

    Only the first sentence should drive the illustration.
    """
    scene = _first_nonempty_line(text)
    scene = re.sub(r"^(例文|本文|文)\s*[:：]\s*", "", scene)
    scene = re.sub(r"\s+", " ", scene).strip()
    return scene

def newest_feed_snapshot(feed_dir: Path) -> Optional[Path]:
    candidates = sorted(
        feed_dir.glob("feed_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_latest_entry(latest_path: Path) -> tuple[dict[str, Any], Path]:
    """
    Resolve the current target entry for illustration.

    Returns:
      (entry_dict, snapshot_feed_path)
    """
    public_dir = artifact_public_dir(latest_path=latest_path)
    feed_dir = artifact_feed_dir(public_dir)

    latest_obj = load_json(latest_path)
    if not isinstance(latest_obj, dict):
        raise SystemExit("ERROR: latest.json is not a JSON object")

    if safe_str(latest_obj.get("date")).strip() and safe_str(latest_obj.get("text")).strip():
        feed_stem = safe_str(latest_obj.get("id")).strip()
        if feed_stem:
            snap = feed_dir / f"{feed_stem}.json"
            if snap.exists():
                return latest_obj, snap
            snap = feed_dir / f"feed_{feed_stem}.json"
            if snap.exists():
                return latest_obj, snap

        snap = newest_feed_snapshot(feed_dir)
        if snap is None:
            raise SystemExit("ERROR: could not determine latest feed snapshot")
        return latest_obj, snap

    feed_url = safe_str(latest_obj.get("feed_url")).strip()
    if not feed_url:
        raise SystemExit("ERROR: latest.json missing date/text and feed_url")

    rel = feed_url.replace("\\", "/").lstrip("./")
    page_path = public_dir / rel
    if not page_path.exists():
        raise SystemExit(f"ERROR: feed page referenced by latest.json was not found: {page_path}")

    page_obj = load_json(page_path)
    items = page_obj.get("items") if isinstance(page_obj, dict) else None
    if not isinstance(items, list) or not items:
        raise SystemExit(f"ERROR: feed page has no items: {page_path}")

    entry = items[0]
    if not isinstance(entry, dict):
        raise SystemExit(f"ERROR: first feed page item is not an object: {page_path}")

    feed_stem = safe_str(entry.get("id")).strip()
    if not feed_stem:
        raise SystemExit("ERROR: first feed page item has no id")

    snap = feed_dir / f"{feed_stem}.json"
    if not snap.exists():
        snap = newest_feed_snapshot(feed_dir)
        if snap is None:
            raise SystemExit(f"ERROR: snapshot not found for id={feed_stem}")

    return entry, snap

def build_prompt(text: str, place: str) -> tuple[str, str]:
    core = " ".join((text or "").split()).strip()[:240]
    p = (place or "").strip()

    no_text_rule = (
        "plain illustration only, no readable text, no letters, no captions, "
        "no speech bubbles, no signs, no posters, no labels, no handwriting"
    )

    if PROMPT_TMPL:
        prompt = _render(PROMPT_TMPL, place=p, core=core)
        prompt = f"{prompt}, {no_text_rule}"
    else:
        base = f"cinematic illustration, {p}, based on this short story: {core}" if p else f"cinematic illustration, based on this short story: {core}"
        prompt = f"{base}, {no_text_rule}"

    negative = _render(NEGATIVE_TMPL, place=p, core=core) if NEGATIVE_TMPL else ""
    negative = ", ".join(
        x for x in [
            negative,
            "text, letters, words, captions, speech bubbles, dialogue balloons, signs, posters, labels, handwriting, watermark, logo",
        ]
        if x
    )
    return prompt, negative

def _match_item(item: dict, *, date: str, text: str, generated_at: str) -> bool:
    if not isinstance(item, dict):
        return False
    same_dt = safe_str(item.get("date")).strip() == date and safe_str(item.get("text")).strip() == text
    same_ga = bool(generated_at) and safe_str(item.get("generated_at")).strip() == generated_at
    return same_dt or same_ga


def patch_feed_file(
    feed_path: Path,
    *,
    date: str,
    text: str,
    generated_at: str,
    feed_stem: str,
    rel_image_url: str,
    image_prompt: str,
    image_negative: str = "",
    image_model: str = MODEL_ID,
    image_generated_at: str,
    image_lora: str = "",
    image_lora_scale: float = 0.0,
) -> bool:
    """
    Patch a feed JSON file that may be:
      - {"items":[...]} (legacy)
      - [{...}, {...}]  (rare)
      - {...}           (current snapshot single-object)
    """
    if not feed_path.exists():
        return False

    obj = load_json(feed_path)
    changed = False

    def apply_patch(it: dict) -> None:
        nonlocal changed
        if not _match_item(it, date=date, text=text, generated_at=generated_at):
            return
        # Canonicalize to feed-stem ID so web can apply stem-match rule.
        old_id = safe_str(it.get("id")).strip()
        if old_id and old_id != feed_stem and "legacy_id" not in it:
            it["legacy_id"] = old_id
        it["id"] = feed_stem
        it["permalink"] = f"./?post={feed_stem}"
        it["image"] = rel_image_url
        it["image_url"] = rel_image_url
        avatar_image = safe_str(it.get("avatar_image")).strip()
        if avatar_image:
            it["avatar_url"] = resolve_public_avatar_url(avatar_image)
        it["image_prompt"] = image_prompt
        it["image_negative"] = image_negative
        it["image_model"] = image_model
        if image_lora:
            it["image_lora"] = image_lora
            it["image_lora_scale"] = float(image_lora_scale)
        it["image_generated_at"] = image_generated_at
        changed = True

    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        for it in obj["items"]:
            if isinstance(it, dict):
                apply_patch(it)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                apply_patch(it)
    elif isinstance(obj, dict):
        # Current snapshot format: the object itself is the item
        apply_patch(obj)

    if changed:
        dump_json(feed_path, obj)
    return changed


def patch_latest_entry_object_only(latest_path: Path, rel_image_url: str, now_iso: str) -> bool:
    """
    Patch latest.json only when it is still an entry-object shape.
    Leave pointer latest.json untouched.
    """
    obj = load_json(latest_path)
    if not isinstance(obj, dict):
        return False

    if not (safe_str(obj.get("date")).strip() and safe_str(obj.get("text")).strip()):
        return False

    obj["image"] = rel_image_url
    obj["image_url"] = rel_image_url
    obj["image_generated_at"] = now_iso
    avatar_image = safe_str(obj.get("avatar_image")).strip()
    if avatar_image:
        obj["avatar_url"] = resolve_public_avatar_url(avatar_image)
    dump_json(latest_path, obj)
    return True

def resolve_fixed_image_path(value: str, public_dir: Path) -> Path:
    """Resolve a fixed image path from latest.json (value may be absolute, repo-relative, or FEED_PATH-relative)."""
    v = (value or "").strip()
    p0 = Path(v)
    candidates = [
        p0,
        public_dir / v,
        public_dir / "image" / v,
        public_dir / "image" / "fixed" / v,
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    msg = " | ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"fixed image not found. tried: {msg}")

def main() -> int:
    public_dir = artifact_public_dir()
    latest_path = artifact_latest_path(public_dir)

    if not latest_path.exists():
        print(f"ERROR: latest.json not found: {latest_path}")
        return 2

    entry, feed_path = resolve_latest_entry(latest_path)

    date = safe_str(entry.get("date")).strip()
    text = safe_str(entry.get("text") or entry.get("tweet")).strip()
    place = safe_str(entry.get("place")).strip()
    generated_at = safe_str(
        entry.get("created_at") or entry.get("generated_at") or entry.get("published_at")
    ).strip()

    if not date or not text:
        print("ERROR: resolved entry missing date/text")
        return 2

    feed_dir = artifact_feed_dir(public_dir)
    feeds = sorted(feed_dir.glob("feed_*.json"), reverse=True)
    feed_stem = feed_path.stem

    out_dir = artifact_image_dir(latest_path=latest_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = safe_str(entry.get("image_fixed") or entry.get("fixed_image")).strip()
    lora_tag = ""
    image_model = MODEL_ID

    if fixed:
        try:
            src = resolve_fixed_image_path(fixed, public_dir)
        except Exception as e:
            print(f"ERROR: {e}")
            return 2
        out_path = out_dir / f"{feed_stem}{(src.suffix or '.png')}"
        print(f"MODEL_ID={MODEL_ID}")
        print("mode=fixed_image")
        print(f"fixed_image={src}")
        print(f"feed_stem={feed_stem}")
        print(f"out_path={out_path}")
        shutil.copyfile(src, out_path)
        prompt = f"[fixed_image] {src.as_posix()}"
        negative = ""
        image_model = "fixed"
    else:
        out_path = out_dir / f"{feed_stem}.png"

        seed = int(SEED_OVERRIDE) if SEED_OVERRIDE.isdigit() else random.randint(0, 2**31 - 1)
        seed += SEED_OFFSET
        prompt, negative = build_prompt(text, place)

        print(f"MODEL_ID={MODEL_ID}")
        print("mode=text2img")
        print(f"seed={seed}")
        print(f"feed_stem={feed_stem}")
        print(f"out_path={out_path}")
        print(f"prompt={prompt}")
        print(f"negative={negative}")

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        pipe = pipe.to(DEVICE)

        image_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative,
            "num_inference_steps": int(STEPS),
            "guidance_scale": float(GUIDANCE_SCALE),
            "generator": generator,
        }

        if LORA_PATH:
            p = Path(LORA_PATH)
            if not p.exists():
                print(f"ERROR: LORA_PATH not found: {p}")
                return 2
            try:
                pipe.load_lora_weights(str(p))
                lora_tag = p.name
                image_kwargs["cross_attention_kwargs"] = {"scale": float(LORA_SCALE)}
            except Exception as e:
                print(f"ERROR: failed to load LoRA: {p} ({e})")
                return 2

        image = pipe(**image_kwargs).images[0]
        image.save(out_path)


    rel_image_url = resolve_public_image_url(out_path)
    now_iso = now_iso_utc()

    patched_latest = patch_latest_entry_object_only(
        latest_path=latest_path,
        rel_image_url=rel_image_url,
        now_iso=now_iso,
    )
    print(f"patched_latest_entry_object={patched_latest}")

    # Patch the resolved snapshot feed file first.
    patched = patch_feed_file(
        feed_path,
        date=date,
        text=text,
        generated_at=generated_at,
        feed_stem=feed_stem,
        rel_image_url=rel_image_url,
        image_prompt=prompt,
        image_negative=negative,
        image_model=image_model,
        image_generated_at=now_iso,
        image_lora=lora_tag,
        image_lora_scale=float(LORA_SCALE),
    )
    print(f"patched_snapshot={patched} path={feed_path}")

    for extra_feed in feeds:
        if extra_feed == feed_path:
            continue
        patch_feed_file(
            extra_feed,
            date=date,
            text=text,
            generated_at=generated_at,
            feed_stem=feed_stem,
            rel_image_url=rel_image_url,
            image_prompt=prompt,
            image_negative=negative,
            image_model=image_model,
            image_generated_at=now_iso,
            image_lora=lora_tag,
            image_lora_scale=float(LORA_SCALE),
        )

    # (Optional) Patch legacy aggregations if they exist
    for legacy in [
        public_dir / "feed.json",
        public_dir / "output.json",
        feed_dir / "feed.json",
        feed_dir / "output.json",
        feed_dir / "feed_latest.json",
    ]:
        if legacy.exists():
            ok = patch_feed_file(
                    legacy,
                    date=date,
                    text=text,
                    generated_at=generated_at,
                    feed_stem=feed_stem,
                    rel_image_url=rel_image_url,
                    image_prompt=prompt,
                    image_negative=negative,
                    image_model=image_model,
                    image_generated_at=now_iso,
                    image_lora=lora_tag,
                    image_lora_scale=LORA_SCALE,
            )
            print(f"patched_legacy={ok} path={legacy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
