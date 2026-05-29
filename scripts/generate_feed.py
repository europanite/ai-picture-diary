import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PUBLIC_DIR = Path("frontend/app/public")
LATEST_PATH = PUBLIC_DIR / "latest.json"
FEED_DIR = PUBLIC_DIR / "feed"
SNAPSHOT_DIR = PUBLIC_DIR / "snapshot"
IMAGE_DIR = PUBLIC_DIR / "image" / "generated"
DEFAULT_IMAGE = "image/avatar/normal.png"


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_sentence() -> str:
    try:
        result = subprocess.run(
            ["python", "scripts/generate_sentence.py"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = result.stdout.strip()
        if text:
            return text
    except Exception:
        pass
    return ""


def generate_image(sentence: str) -> tuple[str, str | None]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"diary_{local_stamp()}.png"
    output_path = IMAGE_DIR / output_name

    result = subprocess.run(
        ["python", "scripts/generate_illustration.py", sentence, str(output_path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0 or not output_path.exists():
        return DEFAULT_IMAGE, None

    prompt = result.stdout.strip() or None
    return f"image/generated/{output_name}", prompt


def build_entry(sentence: str, image_path: str, image_prompt: str | None) -> dict:
    now = utc_now_iso_z()
    entry = {
        "kind": "diary",
        "text": sentence,
        "tweet": sentence,
        "place": "ai-picture-diary",
        "published_at": now,
        "created_at": now,
        "updated_at": now,
        "avatar_image": DEFAULT_IMAGE,
        "image": image_path,
        "fixed_image": "",
        "links": [],
        "weather": None,
    }
    if image_prompt:
        entry["image_prompt"] = image_prompt
    return entry


def main() -> int:
    sentence = generate_sentence()
    image_path, image_prompt = generate_image(sentence)
    entry = build_entry(sentence, image_path, image_prompt)

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    FEED_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    LATEST_PATH.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (FEED_DIR / f"feed_{local_stamp()}.json").write_text(
        json.dumps([entry], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (SNAPSHOT_DIR / "latest_weather.json").write_text(
        json.dumps(
            {
                "ok": True,
                "source": "generate_feed.py",
                "generated_at": utc_now_iso_z(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(entry, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
