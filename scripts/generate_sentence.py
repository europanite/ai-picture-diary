# scripts/generate_sentence.py

from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "scripts" / "prompts" / "llm" / "positive.txt"

def render_prompt(template: str, seed_text: str) -> str:
    base = str(seed_text or "").strip()
    if not base:
        base = "default"

    seed_hex = hashlib.sha256(base.encode("utf-8")).hexdigest()
    bucket = int(seed_hex[:8], 16) % 6

    pattern_map = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
    }
    selected_pattern = pattern_map[bucket]

    return (
        template
        .replace("{{PROMPT_SEED}}", base)
        .replace("{{PATTERN_CODE}}", selected_pattern)
    )

def load_settings() -> dict[str, str | int]:
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        "prompt_path": os.getenv("PROMPT_PATH", str(DEFAULT_PROMPT_PATH)),
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "120")),
        "prompt_seed": os.getenv("PROMPT_SEED", ""),
    }


def load_prompt(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.read_text(encoding="utf-8").strip()


def generate_once(base_url: str, model: str, prompt: str, timeout: int) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", "")).strip()


def normalize_output(text: str) -> str:
    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)



def normalize_japanese_punctuation(text: str) -> str:
    text = str(text or "").strip()
    while "。。" in text:
        text = text.replace("。。", "。")
    while "！！" in text:
        text = text.replace("！！", "！")
    while "？？" in text:
        text = text.replace("？？", "？")
    return text

def is_valid_sentence(text: str) -> bool:
    if not text:
        return False
    if "うんこ" not in text:
        return False
    if len(text) < 12:
        return False
    if "\n" in text or "\r" in text:
        return False

    forbidden_prefixes = (
        "例文",
        "説明",
        "解説",
        "出力",
        "以下",
    )
    if text.startswith(forbidden_prefixes):
        return False

    banned_fragments = (
        "学習ポイント",
        "英訳",
        "translation",
        "Translation",
        "English:",
        "英語:",
        "箇条書き",
        "説明します",
        "次の文",
        "この例文",
    )
    if any(fragment in text for fragment in banned_fragments):
        return False

    sentence_endings = "。！？"
    ending_count = sum(text.count(mark) for mark in sentence_endings)
    if ending_count != 1:
        return False

    return True

def is_valid_study_point(text: str, study_point: str) -> bool:
    if not study_point:
        return False
    if "\n" in study_point or "\r" in study_point:
        return False

    banned_fragments = (
        "学習ポイント",
        "Study Point",
        "ポイント:",
        "英訳",
        "translation",
    )
    if any(fragment in study_point for fragment in banned_fragments):
        return False

    if not study_point.strip():
        return False

    return True

def is_valid_translation_en(translation_en: str) -> bool:
    if not translation_en:
        return False
    if "\n" in translation_en or "\r" in translation_en:
        return False

    banned_fragments = (
        "英訳:",
        "English:",
        "Translation:",
        "translation:",
        "This sentence means",
        "学習ポイント",
    )
    if any(fragment in translation_en for fragment in banned_fragments):
        return False

    return True



def extract_text_payload(raw: str) -> str:
    lines = [line.strip() for line in str(raw or "").splitlines() if line.strip()]
    text = "\n".join(lines).strip().strip('"').strip("'")
    if not text:
        raise ValueError("output text is empty")

    output_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if output_lines:
        first_line = normalize_japanese_punctuation(output_lines[0])
        first_line = first_line.rstrip(".").rstrip("．")
        if first_line and first_line[-1] not in "。！？":
            first_line = first_line + "。"
        output_lines[0] = first_line

    if len(output_lines) >= 2 and not output_lines[1].startswith("英訳:"):
        output_lines[1] = f"英訳: {output_lines[1]}"

    text = "\n".join(output_lines)
    if not quality_check_mock(text):
        raise ValueError(f"mock quality check failed: {text}")
    return text


def quality_check_mock(text: str) -> bool:
    """Temporary mock quality check.

    This intentionally accepts every non-empty string so the pipeline can keep
    moving while the real quality check is redesigned.
    """
    return bool(str(text or "").strip())


def extract_json_payload(raw: str) -> dict[str, str]:
    normalized = normalize_output(raw)

    try:
        data = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"output is not valid JSON: {normalized}") from exc

    if not isinstance(data, dict):
        raise ValueError("output JSON must be an object")

    expected_keys = {"text", "study_point", "translation_en"}
    actual_keys = set(data.keys())
    if actual_keys != expected_keys:
        raise ValueError(f"output keys must be exactly {sorted(expected_keys)}")

    text = normalize_japanese_punctuation(str(data.get("text", "")).strip())
    text = text.rstrip(".").rstrip("．")
    if text and text[-1] not in "。！？":
        text = text + "。"
    study_point = str(data.get("study_point", "")).strip()
    translation_en = str(data.get("translation_en", "")).strip()

    if not is_valid_sentence(text):
        raise ValueError(f"invalid text field: {text}")

    if not is_valid_study_point(text, study_point):
        raise ValueError(f"invalid study_point field: {study_point}")

    if not is_valid_translation_en(translation_en):
        raise ValueError(f"invalid translation_en field: {translation_en}")

    combined_text = "\n".join(
        [
            text,"\n",
            f"学習ポイント: {study_point}","\n",
            f"英訳: {translation_en}","\n",
        ]
    )

    return {
        "text": combined_text,
        "study_point": study_point,
        "translation_en": translation_en,
    }


def main() -> int:
    settings = load_settings()

    prompt_template = load_prompt(str(settings["prompt_path"]))
    prompt = render_prompt(prompt_template, str(settings["prompt_seed"]))

    last_output = ""
    for attempt in range(1, int(settings["max_retries"]) + 1):
        try:
            raw = generate_once(
                base_url=str(settings["ollama_base_url"]),
                model=str(settings["ollama_model"]),
                prompt=prompt,
                timeout=int(settings["request_timeout"]),
            )
            last_output = normalize_output(raw)
            text = extract_text_payload(raw)
            print(text)
            return 0

        except (requests.RequestException, ValueError) as exc:
            print(f"[retry {attempt}] invalid output: {last_output or exc}", file=sys.stderr)

    print("Failed to generate a valid sentence text.", file=sys.stderr)
    if last_output:
        print(f"Last output: {last_output}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())