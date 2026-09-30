"""Unit tests for src/augment.py — noise and phone-quality augmentation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.augment import add_noise, phone_quality, augment


def test_add_noise_shape_and_dtype():
    pcm = np.ones(16000, dtype=np.float32)
    out = add_noise(pcm, snr_db=10.0, seed=42)
    assert out.shape == pcm.shape
    assert out.dtype == np.float32


def test_add_noise_changes_signal():
    pcm = np.ones(16000, dtype=np.float32)
    out = add_noise(pcm, snr_db=10.0, seed=42)
    assert not np.allclose(out, pcm)


def test_add_noise_zero_signal():
    pcm = np.zeros(16000, dtype=np.float32)
    out = add_noise(pcm, snr_db=10.0)
    assert np.allclose(out, 0)


def test_phone_quality_downsamples():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    pcm = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = phone_quality(pcm, orig_sr=sr, target_sr=8000)
    assert len(out) == 8000
    assert out.dtype == np.float32


def test_phone_quality_normalizes():
    sr = 16000
    pcm = (0.5 * np.ones(sr, dtype=np.float32))
    out = phone_quality(pcm, orig_sr=sr, target_sr=8000)
    assert np.max(np.abs(out)) <= 1.0


def test_augment_phone():
    sr = 16000
    pcm = np.ones(sr, dtype=np.float32)
    out = augment(pcm, sr=sr, phone=True, seed=1)
    assert len(out) == 8000


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
