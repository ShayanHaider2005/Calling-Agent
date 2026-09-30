"""Unit tests for scripts/ingest_links.py — segmentation, filtering, license allowlist."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ingest_links import (
    check_license, segment_audio, captions_match_audio, two_outputs_agree,
    filter_segment, MATCH_WER_THRESHOLD,
)


def test_check_license():
    assert check_license("CC-BY-4.0")
    assert check_license("CC0")
    assert check_license("MIT")
    assert check_license("apache-2.0")  # case-insensitive
    assert not check_license("All Rights Reserved")
    assert not check_license("")
    assert not check_license(None)


def test_captions_match_audio():
    assert captions_match_audio("hello world", "hello world", "english")
    assert captions_match_audio("hello, world!", "hello world", "english")
    assert not captions_match_audio("hello world", "goodbye world", "english")
    assert not captions_match_audio("", "hello", "english")


def test_two_outputs_agree():
    assert two_outputs_agree("the cat sat", "the cat sat", "english")
    assert two_outputs_agree("the cat sat", "the cat sat down", "english")
    assert not two_outputs_agree("the cat sat", "a dog ran", "english")
    assert not two_outputs_agree("", "hello", "english")


def test_filter_segment():
    # captions match -> keep
    keep, reason = filter_segment("hello world", "hello world", "something else", "english")
    assert keep and reason == "captions_match_audio"
    # two outputs agree -> keep
    keep, reason = filter_segment("different", "hello world", "hello world", "english")
    assert keep and reason == "two_outputs_agree"
    # neither -> reject
    keep, reason = filter_segment("a", "b", "c", "english")
    assert not keep and reason == "rejected"


def test_segment_audio():
    import numpy as np
    sr = 16000
    sil = np.zeros(sr, dtype=np.float32)
    t = np.linspace(0, 1, sr, endpoint=False)
    speech = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # speech, silence, speech
    pcm = np.concatenate([speech, sil, speech])
    segs = segment_audio(pcm, sr)
    assert len(segs) == 2, f"expected 2 segments, got {len(segs)}"
    # each segment should be roughly 1s
    for s, e in segs:
        assert 0.5 * sr < (e - s) < 1.5 * sr


def test_segment_audio_silence():
    import numpy as np
    pcm = np.zeros(16000, dtype=np.float32)
    segs = segment_audio(pcm, 16000)
    # no speech -> no segments
    assert len(segs) == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
