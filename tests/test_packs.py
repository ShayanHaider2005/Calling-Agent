"""Unit tests for src/packs.py — language pack loading."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.packs import load_pack, available_languages, LanguagePack, DEFAULT_VOICES


def test_load_english_pack():
    p = load_pack("english")
    assert p.language == "english"
    assert p.adapter is None
    assert p.tts_voice == DEFAULT_VOICES["english"]


def test_load_urdu_pack():
    p = load_pack("urdu")
    assert p.language == "urdu"
    assert p.tts_voice == DEFAULT_VOICES["urdu"]
    assert p.detection.get("script") is not None


def test_load_hindi_pack():
    p = load_pack("hindi")
    assert p.language == "hindi"
    assert p.tts_voice == DEFAULT_VOICES["hindi"]


def test_available_languages():
    langs = available_languages()
    assert "english" in langs
    assert "urdu" in langs
    assert "hindi" in langs


def test_language_pack_repr():
    p = LanguagePack("english", adapter="checkpoints/best")
    assert "english" in repr(p)
    assert "checkpoints/best" in repr(p)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
