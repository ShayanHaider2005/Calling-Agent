"""Unit tests for src/eval.py — normalization and WER computation."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval import normalize_text, compute_wer, evaluate


def test_english_normalization():
    assert normalize_text("Hello, World!", "english") == "hello world"
    assert normalize_text("  Multiple   spaces  ", "english") == "multiple spaces"


def test_urdu_normalization():
    # ta marbuta -> ha, alif madda -> alif, diacritics removed
    assert normalize_text("مَرْحَبًا", "urdu") == "مرحبا"
    assert normalize_text("آپ", "urdu") == "اپ"
    assert normalize_text("میرا نام احمد رضا ہے۔", "urdu") == "ميرا نام احمد رضا هي"
    # gol ha -> ha, bari yeh -> yeh
    assert normalize_text("ہے", "urdu") == "هي"


def test_hindi_normalization():
    assert normalize_text("नमस्ते।", "hindi") == "नमस्ते"
    assert normalize_text("नमस्ते, दुनिया!", "hindi") == "नमस्ते दुनिया"


def test_spanish_normalization():
    assert normalize_text("¡Hola, mundo!", "spanish") == "hola mundo"
    assert normalize_text("¿Cómo estás?", "spanish") == "cómo estás"


def test_compute_wer_perfect():
    wer, det = compute_wer(["hello world"], ["hello world"])
    assert wer == 0.0
    assert det["total_words"] == 2


def test_compute_wer_substitution():
    wer, det = compute_wer(["hello world"], ["hello there"])
    assert abs(wer - 0.5) < 1e-9


def test_compute_wer_deletion():
    wer, _ = compute_wer(["hello world"], ["hello"])
    assert abs(wer - 0.5) < 1e-9


def test_evaluate_groups(tmp_path=None):
    import csv
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        meta = os.path.join(td, "metadata.csv")
        with open(meta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "language", "text", "is_mixed"])
            w.writerow(["a.wav", "english", "hello world", "0"])
            w.writerow(["b.wav", "urdu", "میرا نام احمد ہے", "0"])
            w.writerow(["c.wav", "english", "hello world", "1"])
        hyps = {"a.wav": "hello world", "b.wav": "میرا نام احمد هے",
                "c.wav": "hello there"}
        res = evaluate(meta, hyps)
        # one substitution (world->there) out of 8 total words
        assert abs(res["overall"] - 0.125) < 1e-9
        # english rows: a.wav correct, c.wav has 1 error in 4 words
        assert abs(res["per_language"]["english"] - 0.25) < 1e-9
        assert res["per_language"]["urdu"] == 0.0
        assert res["mixed"] == 0.5
        assert res["pure"] == 0.0
        assert res["n"] == 3


def test_evaluate_mixed_flag():
    import csv
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        meta = os.path.join(td, "metadata.csv")
        with open(meta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "language", "text", "is_mixed"])
            w.writerow(["a.wav", "english", "hello world", "0"])
            w.writerow(["b.wav", "english", "hello world", "1"])
        hyps = {"a.wav": "hello world", "b.wav": "hello there"}
        res = evaluate(meta, hyps)
        assert res["pure"] == 0.0
        assert res["mixed"] == 0.5
