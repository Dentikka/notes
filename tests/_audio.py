"""Shared signal measurements for tests."""

import numpy as np


def estimate_f0(x: np.ndarray, sr: int) -> float:
    """Autocorrelation peak with parabolic interpolation."""
    x = x - x.mean()
    spec = np.fft.rfft(x, 2 * len(x))
    ac = np.fft.irfft(spec * np.conj(spec))[: len(x)]
    lo = int(sr / 2000)
    k = lo + int(np.argmax(ac[lo : len(x) // 2]))
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    return sr / (k + 0.5 * (a - c) / (a - 2 * b + c))


def inharmonic_ratio(y: np.ndarray, f0: float, sr: int, width: float = 30.0) -> float:
    """Energy away from the harmonics of `f0` (aliasing, noise) relative to energy on them."""
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    harmonic = np.zeros(freqs.shape, dtype=bool)
    for k in range(1, int(sr / 2 / f0) + 1):
        harmonic |= np.abs(freqs - k * f0) < width
    band = freqs > 50.0
    return float(spec[band & ~harmonic].sum() / spec[band & harmonic].sum())
