import os
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException
from schemas import DiaryGenerateRequest, DiaryGenerateResponse

router = APIRouter(prefix="/diary", tags=["diary"])


class GenerationError(RuntimeError):
    pass


def _load_prompt(topic: str) -> str:
    prompt_path = Path(os.getenv("PROMPT_PATH_POSITIVE", "/prompts/llm/positive.txt"))
    try:
        template = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise GenerationError(f"failed to read prompt template: {prompt_path}") from exc
    return template.replace("{{topic}}", topic)


def _normalize_output(text: str) -> str:
    s = (text or "").strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip(" -*\t") for line in s.split("\n") if line.strip()]
    if not lines:
        return ""
    return " ".join(lines)


def _validate_sentence(sentence: str) -> tuple[bool, str]:
    if not sentence:
        return False, "empty"
    if sentence.count("。") != 1:
        return False, "must be exactly one Japanese sentence"
    if len(sentence) < 12:
        return False, "too short"
    return True, "ok"


def _call_ollama(*, prompt: str, model: str, base_url: str, timeout_s: int, temperature: float) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise GenerationError(f"ollama request failed: {exc}") from exc
    except ValueError as exc:
        raise GenerationError("ollama returned non-json response") from exc

    text = data.get("response")
    if not isinstance(text, str) or not text.strip():
        raise GenerationError("ollama returned empty response")
    return text


@router.get("/health")
def diary_health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/generate", response_model=DiaryGenerateResponse)
def generate_sentence(payload: DiaryGenerateRequest) -> DiaryGenerateResponse:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("RAG_MODEL", "qwen3:8b")
    timeout_s = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))
    prompt = _load_prompt(payload.topic)

    last_error = "generation failed"
    for attempt in range(1, payload.max_retries + 1):
        try:
            raw = _call_ollama(
                prompt=prompt,
                model=model,
                base_url=base_url,
                timeout_s=timeout_s,
                temperature=payload.temperature,
            )
            sentence = _normalize_output(raw)
            ok, reason = _validate_sentence(sentence)
            if ok:
                return DiaryGenerateResponse(sentence=sentence, model=model, retries_used=attempt)
            last_error = f"validation failed: {reason}"
        except GenerationError as exc:
            last_error = str(exc)

    raise HTTPException(status_code=502, detail=last_error)
