# tests/test_validation.py

from scripts.generate_sentence import (
    extract_json_payload,
    is_valid_sentence,
    normalize_output,
)


def test_normalize_output_merges_lines() -> None:
    raw = " 彼は困惑し、\n うんこを踏んだことに気づいて沈黙した。 \n"
    assert normalize_output(raw) == "彼は困惑し、 うんこを踏んだことに気づいて沈黙した。"


def test_is_valid_sentence_accepts_single_sentence() -> None:
    text = "彼は突然の腹痛に動揺し、うんこを我慢したまま面接に臨んだ。"
    assert is_valid_sentence(text) is True


def test_is_valid_sentence_rejects_missing_keyword() -> None:
    text = "彼は突然の腹痛に動揺し、そのまま面接に臨んだ。"
    assert is_valid_sentence(text) is False


def test_is_valid_sentence_rejects_multiple_sentences() -> None:
    text = "彼は困惑した。うんこを踏んだからだ。"
    assert is_valid_sentence(text) is False


def test_is_valid_sentence_rejects_explanatory_prefix() -> None:
    text = "例文：彼はうんこを見て困惑した。"
    assert is_valid_sentence(text) is False


def test_extract_json_payload_accepts_full_contract() -> None:
    raw = (
        '{"text":"うんこが落ちている以上、そのままにしておくわけにはいかない。",'
        '"study_point":"The grammar pattern \\"〜わけにはいかない\\" means that circumstances or social expectations prevent someone from doing something.",'
        '"translation_en":"As long as poop is lying there, we cannot just leave it as it is."}'
    )
    payload = extract_json_payload(raw)

    assert payload["text"] == (
        "うんこが落ちている以上、そのままにしておくわけにはいかない。\n"
        'Study point: The grammar pattern "〜わけにはいかない" means that circumstances or social expectations prevent someone from doing something.\n'
        "English translation: As long as poop is lying there, we cannot just leave it as it is."
    )
    assert '"〜わけにはいかない"' in payload["study_point"]
    assert payload["translation_en"] == "As long as poop is lying there, we cannot just leave it as it is."


def test_extract_json_payload_rejects_empty_study_point() -> None:
    raw = (
        '{"text":"うんこが落ちている以上、そのままにしておくわけにはいかない。",'
        '"study_point":"",'
        '"translation_en":"As long as poop is lying there, we cannot just leave it as it is."}'
    )

    try:
        extract_json_payload(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "invalid study_point field" in str(exc) or "study_point is empty" in str(exc)


def test_extract_json_payload_rejects_empty_translation_en() -> None:
    raw = (
        '{"text":"うんこが落ちている以上、そのままにしておくわけにはいかない。",'
        '"study_point":"The grammar pattern \\"〜わけにはいかない\\" means that circumstances prevent the action.",'
        '"translation_en":""}'
    )

    try:
        extract_json_payload(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "invalid translation_en field" in str(exc) or "translation_en is empty" in str(exc)


def test_extract_json_payload_rejects_invalid_text_with_label() -> None:
    raw = (
        '{"text":"例文: うんこが落ちている以上、そのままにしておくわけにはいかない。",'
        '"study_point":"The grammar pattern \\"〜わけにはいかない\\" means that circumstances prevent the action.",'
        '"translation_en":"As long as poop is lying there, we cannot just leave it as it is."}'
    )

    try:
        extract_json_payload(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "invalid text field" in str(exc)


def test_extract_json_payload_accepts_multiline_json_after_normalization() -> None:
    raw = """
    {
      "text": "うんこを放置しておくわけにはいかない。",
      "study_point": "The grammar pattern \\"〜わけにはいかない\\" means that circumstances prevent the action.",
      "translation_en": "We cannot simply leave the poop there."
    }
    """
    payload = extract_json_payload(raw)

    assert payload["text"] == "うんこを放置しておくわけにはいかない。"
    assert payload["translation_en"] == "We cannot simply leave the poop there."


def test_extract_json_payload_rejects_extra_keys() -> None:
    raw = (
        '{"text":"うんこが落ちている以上、そのままにしておくわけにはいかない。",'
        '"study_point":"The grammar pattern \\"〜わけにはいかない\\" means that circumstances prevent the action.",'
        '"translation_en":"As long as poop is lying there, we cannot just leave it as it is.",'
        '"extra":"ng"}'
    )

    try:
        extract_json_payload(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "output keys must be exactly" in str(exc)


def test_extract_json_payload_rejects_study_point_without_target_marker() -> None:
    raw = (
        '{"text":"うんこが落ちている以上、そのままにしておくわけにはいかない。",'
        '"study_point":"Circumstances prevent the action.",'
        '"translation_en":"As long as poop is lying there, we cannot just leave it as it is."}'
    )

    try:
        extract_json_payload(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "invalid study_point field" in str(exc)


def test_extract_json_payload_rejects_translation_with_label() -> None:
    raw = (
        '{"text":"うんこが落ちている以上、そのままにしておくわけにはいかない。",'
        '"study_point":"The grammar pattern \\"〜わけにはいかない\\" means that circumstances prevent the action.",'
        '"translation_en":"English: As long as poop is lying there, we cannot just leave it as it is."}'
    )

    try:
        extract_json_payload(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "invalid translation_en field" in str(exc)
