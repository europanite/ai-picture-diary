"""No network, Hugging Face, or Ollama required for diary -> feed regression tests."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.generate_sentence import build_entry_payload, main as generate_main, parse_output, save_draft, story_keywords, render_prompt
from scripts.publish_news_diary import publish


@pytest.fixture(autouse=True)
def mock_distinctive_word_dependencies(monkeypatch):
    """Exercise deterministic POS and rarity ranking offline in regression tests."""
    from scripts import generate_sentence

    class Token:
        def __init__(self, text, start):
            self.text, self.idx = text, start
            self.is_punct = False
            self.is_stop = text.lower() in {"and", "the", "of", "in"}
            self.pos_ = ("VERB" if text.lower() in {"attack", "investigate", "faces"}
                         else "ADJ" if text.lower() in {"israeli", "olive", "quantum", "brazilian"}
                         else "PROPN" if text.lower() == "amazon"
                         else "NOUN")

    class Tagger:
        def __call__(self, phrase):
            import re
            return [Token(m.group(), m.start()) for m in re.finditer(r"[A-Za-z'-]+", phrase)]

    freqs = {"israeli": 4.4, "settlers": 3.5, "olive": 4.2,
             "farmers": 4.3, "attack": 5.2, "quantum": 4.2,
             "prototypes": 3.7, "robotics": 3.9, "company": 5.6,
             "brazilian": 4.3, "amazon": 4.5, "deforestation": 3.3}
    monkeypatch.setattr(generate_sentence, "_news_nlp", lambda: Tagger())
    monkeypatch.setattr(generate_sentence, "_zipf_frequency", lambda w: freqs.get(w.lower(), 4.8))


TEXT = (
    "I woke to discover Eva Marie Saint beside a little stage in New Delhi. "
    "An old newspaper announced that Israeli settlers attack olive farmers, but the "
    "actors immediately reminded everyone it was only a prop for our fictional play. "
    "I spent the afternoon painting paper olive trees and watching the crowded street "
    "turn into an imaginary theater. No one in my dream knew what was real."
)


def selection():
    keywords = ["Eva Marie Saint", "New Delhi", "Israeli settlers attack olive farmers"]
    return {
        "keywords": keywords,
        "selected": [
            {"text": k, "publisher": f"Example News {i}", "title": f"Headline about {k}",
             "url": f"https://example.com/story/{i}", "published_at": "2026-10-16T17:00:00Z"}
            for i, k in enumerate(keywords)
        ],
        "selection": {"mean_distance": 0.94, "method": "balanced"},
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def test_payload_provenance_is_from_selection_only():
    payload = build_entry_payload(selection(), TEXT)
    assert payload["kind"] == "news_diary"
    assert payload["keywords"] == selection()["keywords"]
    assert payload["selection"]["embedding_model"] == selection()["embedding_model"]
    assert all(url.startswith("https://") for url in payload["links"])
    assert all(s["published_at"].endswith("Z") for s in payload["source_articles"])


def test_save_draft_writes_valid_utf8_json(tmp_path):
    path = tmp_path / "nested" / "draft.json"
    save_draft(path, build_entry_payload(selection(), TEXT))
    assert json.loads(path.read_text())["text"] == TEXT


def test_generator_mocked_ollama_writes_draft(tmp_path, monkeypatch, capsys):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection()), encoding="utf-8")
    draft_path = tmp_path / "draft.json"

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"text": TEXT})}

    def fake_post(url, json, timeout):
        assert url == "http://localhost:11434/api/generate"
        assert json["format"]["required"] == ["text"]
        assert "Eva Marie Saint" in json["prompt"]
        return FakeResponse()

    monkeypatch.setattr("scripts.generate_sentence.requests.post", fake_post)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    monkeypatch.setattr("sys.argv", ["generate_sentence.py", "--context", str(selection_path),
                                    "--output-json", str(draft_path)])
    assert generate_main() == 0
    assert json.loads(draft_path.read_text())["text"] == TEXT
    assert TEXT in capsys.readouterr().out


def test_stale_legacy_prompt_path_is_ignored(tmp_path, monkeypatch):
    """Old .env PROMPT_PATH must not make the English diary writer fail."""
    context_path = tmp_path / "selection.json"
    context_path.write_text(json.dumps(selection()), encoding="utf-8")
    draft_path = tmp_path / "draft.json"

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"text": TEXT})}

    monkeypatch.setenv("PROMPT_PATH", "scripts/prompts/llm/sentence.txt")
    monkeypatch.delenv("NEWS_DIARY_PROMPT_PATH", raising=False)
    monkeypatch.setattr("scripts.generate_sentence.requests.post", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(sys, "argv", ["generate_sentence.py", "--context", str(context_path),
                                      "--output-json", str(draft_path)])
    assert generate_main() == 0
    assert json.loads(draft_path.read_text(encoding="utf-8"))["text"] == TEXT


def test_news_diary_custom_prompt_path(tmp_path, monkeypatch):
    """The new explicit prompt override is honored independently of PROMPT_PATH."""
    context_path = tmp_path / "selection.json"
    context_path.write_text(json.dumps(selection()), encoding="utf-8")
    prompt_path = tmp_path / "custom-prompt.txt"
    prompt_path.write_text("Custom English prompt: {{STORY_KEYWORDS}}", encoding="utf-8")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"text": TEXT})}

    def fake_post(url, json, timeout):
        assert "Custom English prompt:" in json["prompt"]
        assert "Eva Marie Saint" in json["prompt"]
        return FakeResponse()

    monkeypatch.setenv("PROMPT_PATH", "scripts/prompts/llm/sentence.txt")
    monkeypatch.setenv("NEWS_DIARY_PROMPT_PATH", str(prompt_path))
    monkeypatch.setattr("scripts.generate_sentence.requests.post", fake_post)
    monkeypatch.setattr(sys, "argv", ["generate_sentence.py", "--context", str(context_path)])
    assert generate_main() == 0


def test_generator_does_not_publish_invalid_response(tmp_path, monkeypatch):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection()))
    draft_path = tmp_path / "draft.json"

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"text": "I saw a satellite."})}

    monkeypatch.setattr("scripts.generate_sentence.requests.post", lambda *a, **k: FakeResponse())
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setattr("sys.argv", ["generate_sentence.py", "--context", str(selection_path),
                                    "--output-json", str(draft_path)])
    assert generate_main() == 1
    assert not draft_path.exists()


def publish_fixture(tmp_path):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection()))
    draft_path = tmp_path / "draft.json"
    save_draft(draft_path, build_entry_payload(selection(), TEXT))
    return draft_path, selection_path


def test_publish_existing_feed_shape_and_html(tmp_path):
    draft_path, selection_path = publish_fixture(tmp_path)
    public = tmp_path / "public"
    result = publish(draft_path=draft_path, selection_path=selection_path,
                     public_dir=public, now=datetime(2026, 10, 17, 1, 25, tzinfo=timezone.utc))
    assert result["created"] is True
    entry_file = public / "feed" / (result["id"] + ".json")
    entry = json.loads(entry_file.read_text())["items"][0]
    assert entry["id"] == result["id"]
    assert entry["kind"] == "news_diary"
    assert entry["date"] == "2026-10-17"
    assert entry["fictional"] is True
    assert entry["source_articles"][0]["publisher"] == "Example News 0"
    index = json.loads((public / "feed" / "index.json").read_text())
    assert index["items"][0]["id"] == result["id"]
    assert json.loads((public / "latest.json").read_text())["feed_url"] == "./feed/page_000.json"
    html = (public / "posts" / result["id"] / "index.html").read_text()
    assert "News inspiration" in html and "fictional" in html
    assert "example.com/story/2" in html


def test_repeated_publication_is_idempotent(tmp_path):
    draft_path, selection_path = publish_fixture(tmp_path)
    public = tmp_path / "public"
    now = datetime(2026, 10, 17, 1, 25, tzinfo=timezone.utc)
    first = publish(draft_path=draft_path, selection_path=selection_path, public_dir=public, now=now)
    second = publish(draft_path=draft_path, selection_path=selection_path, public_dir=public, now=now)
    assert first["id"] == second["id"] and second["created"] is False
    assert len(list((public / "feed").glob("feed_*.json"))) == 1


def test_modified_attribution_rejected_before_writing(tmp_path):
    draft_path, selection_path = publish_fixture(tmp_path)
    draft = json.loads(draft_path.read_text())
    draft["source_articles"][0]["url"] = "https://example.invalid/modified"
    draft_path.write_text(json.dumps(draft))
    public = tmp_path / "public"
    with pytest.raises(ValueError, match="provenance"):
        publish(draft_path=draft_path, selection_path=selection_path, public_dir=public)
    assert not public.exists()


def test_changed_selection_rejected_before_writing(tmp_path):
    draft_path, selection_path = publish_fixture(tmp_path)
    context = selection()
    context["selected"][0]["url"] = "https://example.com/different"
    selection_path.write_text(json.dumps(context))
    public = tmp_path / "public"
    with pytest.raises(ValueError, match="provenance"):
        publish(draft_path=draft_path, selection_path=selection_path, public_dir=public)
    assert not public.exists()


def test_publication_preserves_existing_archived_entry(tmp_path):
    draft_path, selection_path = publish_fixture(tmp_path)
    public = tmp_path / "public"
    feed = public / "feed"
    feed.mkdir(parents=True)
    historical = feed / "feed_20261001_010101.json"
    previous = {"id": "feed_20261001_010101", "kind": "diary", "text": "An old diary.",
                "created_at": "2026-10-01T01:01:01Z", "image": "image/feed_20261001_010101.png"}
    content = {"items": [previous]}
    historical.write_text(json.dumps(content), encoding="utf-8")
    publish(draft_path=draft_path, selection_path=selection_path,
            public_dir=public, now=datetime(2026, 10, 17, tzinfo=timezone.utc))
    assert json.loads(historical.read_text()) == content
    index = json.loads((feed / "index.json").read_text())
    assert index["count"] == 2
    assert any(item.get("image") == previous["image"] for item in index["items"])


def test_retry_targets_missing_keyword_and_reuses_draft(tmp_path, monkeypatch, capsys):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection()), encoding="utf-8")
    draft_path = tmp_path / "draft.json"
    first_draft = (
        "I visited New Delhi in my dream and saw Eva Marie Saint standing by a cardboard "
        "theater with painted windows. Every paper prop blew across the stage as the wind "
        "grew stronger. I followed the props through an imaginary market and found an old "
        "stage curtain hidden behind a bright red shop sign. It felt like a strange movie."
    )
    answers = [first_draft, TEXT]
    prompts = []

    class FakeResponse:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"text": self.value})}

    def fake_post(url, json, timeout):
        prompts.append(json["prompt"])
        return FakeResponse(answers.pop(0))

    monkeypatch.setattr("scripts.generate_sentence.requests.post", fake_post)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setattr(sys, "argv", ["generate_sentence.py", "--context", str(selection_path),
                                      "--output-json", str(draft_path)])
    assert generate_main() == 0
    assert json.loads(draft_path.read_text())["text"] == TEXT
    assert len(prompts) == 2
    assert "REVISION REQUIRED" in prompts[1]
    assert 'Israeli settlers attack olive farmers' in prompts[1]
    assert first_draft in prompts[1]
    assert "Missing" in capsys.readouterr().err or "missing" in prompts[1].lower()


def test_missing_word_error_identifies_only_absent_terms():
    from scripts.generate_sentence import MissingKeywordsError, parse_output
    partial = (
        "I watched Eva Marie Saint wander around New Delhi in my dream, past a tiny "
        "bookshop filled with postcards and paper boats. The paper boats sailed across "
        "the shelves like colorful fish, and I wondered whether the street had turned "
        "into an enormous stage. An audience watched from behind the window in silence."
    )
    with pytest.raises(MissingKeywordsError) as error:
        parse_output(json.dumps({"text": partial}), selection()["keywords"])
    assert error.value.missing == ["Israeli settlers attack olive farmers"]


def test_missing_keyword_retry_never_writes_unvalidated_draft(tmp_path, monkeypatch, capsys):
    context_path = tmp_path / "selection.json"
    context_path.write_text(json.dumps(selection()))
    output = tmp_path / "draft.json"
    weak = (
        "I walked across an imaginary city and visited a paper theater in my dream. "
        "Every person there seemed to be carrying a long painted umbrella. The actors "
        "smiled and waved as the lights came on, and I found a little cardboard boat "
        "floating on the stage. It felt like an unusually quiet afternoon."
    )

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"text": weak})}

    monkeypatch.setattr("scripts.generate_sentence.requests.post", lambda *a, **kw: FakeResponse())
    monkeypatch.setenv("MAX_RETRIES", "2")
    monkeypatch.setattr(sys, "argv", ["generate_sentence.py", "--context", str(context_path),
                                      "--output-json", str(output)])
    assert generate_main() == 1
    assert not output.exists()
    assert "missing required keywords" in capsys.readouterr().err


def test_rare_word_not_the_last_two_words():
    from scripts.generate_sentence import distinctive_word
    assert distinctive_word("Israeli settlers attack olive farmers") == "settlers"


def test_generic_verbs_filtered_even_if_rare():
    from scripts.generate_sentence import distinctive_word
    assert distinctive_word("robotics company investigate quantum prototypes") == "prototypes"


def test_word_frequency_rather_than_last_position_controls_story_seed():
    context = selection()
    context["selected"][2]["kind"] = "noun_phrase"
    assert story_keywords(context)[2] == "settlers"
    assert context["keywords"][2] == "Israeli settlers attack olive farmers"


def test_long_event_phrase_becomes_short_story_seed_without_losing_source():
    context = selection()
    context["selected"][2]["kind"] = "noun_phrase"
    assert story_keywords(context) == ["Eva Marie Saint", "New Delhi", "settlers"]
    assert context["keywords"][2] == "Israeli settlers attack olive farmers"
    prompt = render_prompt(context, "STORY: {{STORY_KEYWORDS}} / SOURCE: {{NEWS_CONTEXT}}", "42")
    assert '"settlers"' in prompt
    assert "Israeli settlers attack olive farmers" in prompt


def test_named_entity_stays_verbatim_even_when_long():
    context = selection()
    context["selected"][2].update({
        "text": "International Monetary Fund", "kind": "named_entity", "entity_label": "ORG",
    })
    context["keywords"][2] = "International Monetary Fund"
    assert story_keywords(context)[2] == "International Monetary Fund"


def test_short_event_anchor_must_still_appear():
    context = selection()
    context["selected"][2]["kind"] = "noun_phrase"
    good = ("I met Eva Marie Saint in New Delhi, where settlers and olive farmers were selling painted "
            "paper fruit at a tiny theater. I wandered between strange stalls, imagining "
            "that the paper oranges were lanterns and that the streets were a stage "
            "made of colored glass. I kept the handmade ticket in my pocket as a "
            "souvenir of a very odd day.")
    assert parse_output(json.dumps({"text": good}), story_keywords(context)) == good
    from scripts.generate_sentence import MissingKeywordsError
    with pytest.raises(MissingKeywordsError, match="settlers"):
        parse_output(json.dumps({"text": good.replace("settlers", "vendors")}), story_keywords(context))


def test_writer_and_publisher_accept_short_event_anchor(tmp_path, monkeypatch):
    context = selection()
    context["selected"][2]["kind"] = "noun_phrase"
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(context), encoding="utf-8")
    draft_path = tmp_path / "draft.json"
    story = ("I visited New Delhi today and saw a poster honoring Eva Marie Saint "
             "above a row of market stalls. Near the stalls, settlers and olive farmers were "
             "painting blue flowers onto a wall. I helped them carry paper lanterns "
             "into a courtyard, then stayed to watch a shadow puppet show. Everything "
             "looked so vivid that I forgot this was only an imaginary town.")
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": json.dumps({"text": story})}
    prompts = []
    def fake_post(url, json, timeout):
        prompts.append(json["prompt"])
        return FakeResponse()
    monkeypatch.setattr("scripts.generate_sentence.requests.post", fake_post)
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setattr(sys, "argv", ["generate_sentence.py", "--context", str(selection_path),
                                     "--output-json", str(draft_path)])
    assert generate_main() == 0
    draft = json.loads(draft_path.read_text())
    assert draft["keywords"][2] == "Israeli settlers attack olive farmers"
    assert draft["story_keywords"][2] == "settlers"
    assert "Required STORY KEYWORDS" in prompts[0]
    output = publish(draft_path=draft_path, selection_path=selection_path,
                     public_dir=tmp_path / "public",
                     now=datetime(2026, 10, 17, 1, 25, tzinfo=timezone.utc))
    assert output["created"] is True
    feed = json.loads(Path(output["feed"]).read_text())["items"][0]
    assert feed["story_keywords"][2] == "settlers"


def test_publisher_rejects_tampered_story_keywords(tmp_path):
    context = selection()
    context["selected"][2]["kind"] = "noun_phrase"
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(context), encoding="utf-8")
    story = ("I visited New Delhi today and met Eva Marie Saint beside a quiet "
             "marketplace. Nearby, settlers and olive farmers were building a colorful paper "
             "carousel for a small theater. I spent the afternoon painting a tiny "
             "wooden door and watching people hang blue flags from the ceiling. "
             "By sunset, the market looked like a scene from a fantastic dream.")
    draft = build_entry_payload(context, story)
    draft["story_keywords"][2] = "different words"
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        publish(draft_path=draft_path, selection_path=selection_path,
                public_dir=tmp_path / "public")


def test_three_word_event_phrase_uses_distinctive_word():
    """Regresses the scheduled run failure on Brazilian Amazon deforestation."""
    context = selection()
    context["selected"][2].update({
        "text": "Brazilian Amazon deforestation", "kind": "noun_phrase",
        "entity_label": None,
    })
    context["keywords"][2] = "Brazilian Amazon deforestation"
    assert story_keywords(context) == ["Eva Marie Saint", "New Delhi", "deforestation"]
    assert context["keywords"][2] == "Brazilian Amazon deforestation"
    prompt = render_prompt(context, "{{STORY_KEYWORDS}} / {{NEWS_CONTEXT}}", "42")
    assert '"deforestation"' in prompt
    assert "Brazilian Amazon deforestation" in prompt


def test_misclassified_event_entity_is_reduced():
    context = selection()
    context["selected"][2].update({
        "text": "Brazilian Amazon deforestation", "kind": "named_entity", "entity_label": "LOC",
    })
    context["keywords"][2] = "Brazilian Amazon deforestation"
    assert story_keywords(context)[2] == "deforestation"


def test_short_entity_and_two_word_phrase_stay_intact():
    context = selection()
    context["selected"][2].update({"text": "olive farmers", "kind": "noun_phrase"})
    context["keywords"][2] = "olive farmers"
    assert story_keywords(context)[2] == "olive farmers"
    context["selected"][2].update({"text": "World Health Organization", "kind": "named_entity"})
    context["keywords"][2] = "World Health Organization"
    assert story_keywords(context)[2] == "World Health Organization"


def test_three_word_event_generated_and_published(tmp_path, monkeypatch):
    context = selection()
    context["selected"][2].update({"text": "Brazilian Amazon deforestation", "kind": "noun_phrase"})
    context["keywords"][2] = "Brazilian Amazon deforestation"
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(context), encoding="utf-8")
    draft_path = tmp_path / "draft.json"
    story = (
        "I met Eva Marie Saint in New Delhi, where a colorful theater displayed a "
        "painting about deforestation. I wandered into an imaginary garden full of "
        "paper trees, and I helped the artists hang blue lanterns from the ceiling. "
        "The exhibit was made of old newspapers and wooden boxes. After sunset, "
        "we sat together and watched shadows move across the painted walls."
    )

    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": json.dumps({"text": story})}

    monkeypatch.setattr("scripts.generate_sentence.requests.post", lambda *a, **k: FakeResponse())
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setattr(sys, "argv", ["generate_sentence.py", "--context", str(selection_path),
                                      "--output-json", str(draft_path)])
    assert generate_main() == 0
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    assert draft["keywords"][2] == "Brazilian Amazon deforestation"
    assert draft["story_keywords"][2] == "deforestation"
    output = publish(draft_path=draft_path, selection_path=selection_path,
                     public_dir=tmp_path / "public")
    assert output["created"] is True
