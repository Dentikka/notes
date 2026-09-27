"""Loudness measurement (ITU-R BS.1770-4) and an offline peak limiter for the master bus."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter

from notes.engine.dsp import db_to_gain, gain_to_db

__all__ = ["limit", "lufs", "peak_db"]


def _k_weighting(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two K-weighting stages (high shelf, then high-pass), derived for any sample rate."""
    f0, gain, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10.0 ** (gain / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf_b = np.array([(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0])
    shelf_a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    hp_b = np.array([1.0, -2.0, 1.0])
    hp_a = np.array([1.0, 2.0 * (k * k - 1.0) / (1.0 + k / q + k * k), (1.0 - k / q + k * k) / (1.0 + k / q + k * k)])
    return [(shelf_b, shelf_a), (hp_b, hp_a)]


def lufs(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness of a ``(channels, samples)`` signal in LUFS (-inf if silent)."""
    y = np.atleast_2d(np.asarray(x, dtype=float))
    for b, a in _k_weighting(sr):
        y = lfilter(b, a, y, axis=-1)
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if y.shape[1] < block:
        power = float(np.mean(y**2, axis=1).sum())
        return -0.691 + 10.0 * np.log10(power) if power > 0 else float("-inf")
    cum = np.concatenate((np.zeros((y.shape[0], 1)), np.cumsum(y**2, axis=1)), axis=1)
    starts = np.arange(0, y.shape[1] - block + 1, hop)
    power = ((cum[:, starts + block] - cum[:, starts]) / block).sum(axis=0)
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10.0 * np.log10(power)
    keep = loud > -70.0
    if not keep.any():
        return float("-inf")
    relative = -0.691 + 10.0 * np.log10(power[keep].mean()) - 10.0
    keep &= loud > relative
    return float(-0.691 + 10.0 * np.log10(power[keep].mean()))


def peak_db(x: np.ndarray) -> float:
    return gain_to_db(float(np.max(np.abs(x))) if np.size(x) else 0.0)


def limit(x: np.ndarray, sr: int, ceiling_db: float = -1.0, window: float = 0.005) -> tuple[np.ndarray, float]:
    """Offline look-around peak limiter; returns (output, largest gain reduction in dB).

    The required gain is min-filtered and then box-smoothed over the same window, which
    guarantees the smoothed gain never exceeds what any sample needs: every value averaged
    at sample n is a minimum over a window that contains n.
    """
    ceiling = db_to_gain(ceiling_db)
    peaks = np.max(np.abs(np.atleast_2d(x)), axis=0)
    need = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
    if need.min() >= 1.0:
        return x, 0.0
    size = 2 * max(1, int(window * sr)) + 1
    gain = uniform_filter1d(minimum_filter1d(need, size=size, mode="nearest"), size=size, mode="nearest")
    gain = np.minimum(gain, need)
    return x * gain, -gain_to_db(float(gain.min()))
