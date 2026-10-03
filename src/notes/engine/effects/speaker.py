"""Impulse response of a miked 4x12 guitar cabinet, designed rather than measured.

A real cabinet's response is not a smooth curve: the paper cone breaks up into modes above
~1 kHz, so the response is jagged with peaks and notches a sixth of an octave apart, and
it falls off a cliff above 5–6 kHz. That irregularity is much of what separates a miked
cabinet from a filter, and is why amp simulators load measured impulse responses.

This one follows the typical shape of a closed 4x12 with Greenback-style speakers and a
dynamic mic near the cap edge: a low resonance near 100 Hz, a shallow low-mid dip,
the presence hump around 2–4 kHz, the cliff above it, plus a fixed pseudo-random ripple
that grows with frequency (the cone modes). The magnitude is turned into a minimum-phase
response, and a floor reflection and a few room reflections follow a millisecond or so
later.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["speaker_ir"]

_N_FFT = 8192
#: (Hz, dB) anchors of the cabinet + mic response.
_SHAPE = (
    (20.0, -36.0), (50.0, -18.0), (70.0, -8.0), (100.0, 0.0), (130.0, -0.5), (200.0, -2.0),
    (400.0, -4.0), (700.0, -3.5), (1000.0, -2.0), (1500.0, -1.0), (2000.0, 1.5), (2600.0, 4.0),
    (3200.0, 2.0), (3900.0, 3.5), (4600.0, 0.0), (5300.0, -8.0), (6200.0, -18.0), (7500.0, -28.0),
    (9000.0, -38.0), (12000.0, -50.0), (24000.0, -70.0),
)
_RIPPLE_SEED = 1960  # one fixed cabinet
#: Reflections after the direct sound: (delay ms, gain).
_ROOM = ((1.3, -0.35), (4.7, 0.22), (7.9, -0.15), (11.2, 0.12), (16.5, 0.08), (23.0, 0.05))


def _magnitude_db(freqs: np.ndarray, tone: float) -> np.ndarray:
    logf = np.log2(np.maximum(freqs, 10.0))
    anchors = np.log2([f for f, _ in _SHAPE])
    db = np.interp(logf, anchors, [d for _, d in _SHAPE])
    # Cone modes: ripple a sixth of an octave apart, growing from ~800 Hz up.
    rng = np.random.default_rng(_RIPPLE_SEED)
    knots = np.arange(np.log2(300.0), np.log2(16000.0), 1.0 / 6.0)
    ripple = CubicSpline(knots, rng.normal(0.0, 1.0, knots.size))(np.clip(logf, knots[0], knots[-1]))
    depth = 3.5 * np.clip((logf - np.log2(800.0)) / 2.5, 0.0, 1.0)
    db += depth * ripple
    # `tone` moves the presence and the cliff: +-3 dB around 3 kHz, the cliff +-15 % in frequency.
    db += (tone - 0.5) * 6.0 * np.exp(-0.5 * ((logf - np.log2(3000.0)) / 0.6) ** 2)
    shift = np.log2(1.0 + 0.3 * (tone - 0.5))
    cliff = np.interp(logf - shift, anchors, [d for _, d in _SHAPE]) - np.interp(logf, anchors, [d for _, d in _SHAPE])
    return db + np.where(freqs > 4600.0, cliff, 0.0)


def _minimum_phase(mag: np.ndarray) -> np.ndarray:
    """Minimum-phase impulse response with the given one-sided magnitude (homomorphic method)."""
    full = np.concatenate([mag, mag[-2:0:-1]])
    cep = np.fft.ifft(np.log(np.maximum(full, 1e-8))).real
    n = full.size
    fold = np.zeros(n)
    fold[0], fold[n // 2] = cep[0], cep[n // 2]
    fold[1 : n // 2] = 2.0 * cep[1 : n // 2]
    return np.fft.ifft(np.exp(np.fft.fft(fold))).real


@lru_cache(maxsize=16)
def speaker_ir(sr: int, tone: float = 0.5, room: float = 0.2) -> np.ndarray:
    """Cabinet impulse response at `sr`; unity gain at 1 kHz. `room` scales the reflections."""
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)
    mag = 10.0 ** (_magnitude_db(freqs, tone) / 20.0)
    ir = _minimum_phase(mag)[: _N_FFT // 2]
    ir *= np.hanning(2 * ir.size)[ir.size :] ** 0.5  # settle the tail
    out = ir.copy()
    for ms, gain in _ROOM:
        k = int(round(ms * 1e-3 * sr))
        if room and k < out.size:
            out[k:] += room * gain * ir[: out.size - k]
    at_1k = np.abs(np.fft.rfft(out, _N_FFT))[int(round(1000.0 * _N_FFT / sr))]
    return out / at_1k
