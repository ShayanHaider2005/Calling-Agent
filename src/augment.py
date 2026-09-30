"""Audio augmentation for training data: background noise + phone-quality 8kHz.

Used to make the STT robust to real phone-call conditions. All functions are
CPU/numpy only.
"""
import numpy as np


def add_noise(pcm, snr_db=15.0, seed=None):
    """Add white Gaussian noise at a given SNR (dB). Returns float32 array."""
    if seed is not None:
        np.random.seed(seed)
    rms = float(np.sqrt(np.mean(np.square(pcm)))) if len(pcm) else 0.0
    if rms == 0:
        return pcm.copy()
    noise_rms = rms / (10 ** (snr_db / 20.0))
    noise = np.random.normal(0, noise_rms, len(pcm)).astype(np.float32)
    return (pcm + noise).astype(np.float32)


def phone_quality(pcm, orig_sr=16000, target_sr=8000):
    """Simulate phone audio: downsample to 8 kHz, add mild noise, normalize."""
    if orig_sr <= target_sr:
        return pcm.copy()
    # simple downsampling by linear interpolation
    n_out = int(len(pcm) * target_sr / orig_sr)
    if n_out < 1:
        return pcm.copy()
    x_old = np.linspace(0, 1, len(pcm))
    x_new = np.linspace(0, 1, n_out)
    down = np.interp(x_new, x_old, pcm).astype(np.float32)
    # mild noise + normalize to [-1, 1]
    down = add_noise(down, snr_db=20.0)
    peak = float(np.max(np.abs(down))) if len(down) else 1.0
    if peak > 0:
        down = down / peak * 0.9
    return down.astype(np.float32)


def augment(pcm, sr=16000, phone=False, noise_snr=15.0, seed=None):
    """Apply augmentation. If phone=True, also downsample to 8 kHz."""
    out = add_noise(pcm, snr_db=noise_snr, seed=seed)
    if phone:
        out = phone_quality(out, orig_sr=sr, target_sr=8000)
    return out
