"""Procedural pluck response with passive string/body energy exchange.

The body is a designed sum of positive-mass damped modes, not a measured IR.
The bridge mobility enters both radiation and the string's boundary condition.
String losses and radiation remain approximations, not a calibrated guitar model.
See https://euphonics.org/5-1-1-coupling-a-string-to-the-instrument-body/ and
https://euphonics.org/5-3-synthesising-plucked-strings/ for the coupling formulation.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import next_fast_len

__all__ = ["body_mobility", "coupled_string"]

# Hz, Q, peak mobility (m/s/N): mathematical design values, not measurements.
_MODES = ((98, 14, .05), (198, 22, .04), (232, 28, .018), (296, 30, .013),
          (384, 34, .012), (466, 38, .010), (556, 40, .009), (640, 44, .008),
          (788, 48, .006), (962, 50, .005))


@lru_cache(maxsize=8)
def body_mobility(nfft: int, sr: int, sigma: float, size: float = 1., diffuse: float = .8) -> np.ndarray:
    """Positive-real bridge mobility evaluated at sigma + i*omega; read-only result.

    Positive modal residues make the boundary passive. ``diffuse`` controls a
    deterministic dense upper modal bank, rather than a recorded impulse or noise.
    """
    if nfft < 2 or sr < 4000 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("invalid mobility grid")
    if not np.isfinite(size) or size <= 0 or not np.isfinite(diffuse) or diffuse < 0:
        raise ValueError("body size must be positive and diffuse nonnegative")
    s = sigma + 2j * np.pi * np.fft.rfftfreq(nfft, 1 / sr)
    y = _mobility(s, sr, size, diffuse)
    y.flags.writeable = False
    return y


def _mobility(s: np.ndarray, sr: int, size: float, diffuse: float) -> np.ndarray:
    y = np.zeros_like(s, dtype=complex)
    modes = list(_MODES)
    for index, hz in enumerate(np.geomspace(1100, min(9000, .44 * sr), 48)):
        hz *= 1 + .018 * np.sin(index * 2.399963)
        modes.append((hz, 45 + 15 * np.sin(index * .7) ** 2, .003 * diffuse * (1100 / hz) ** .3))
    for hz, q, peak in modes:
        omega = 2 * np.pi * hz / size
        damping = omega / q
        y += peak * damping * s / (s*s + damping*s + omega*omega)
    return y


def _tuned_frequency(freq: float, sr: int, size: float, diffuse: float, coupling: float,
                     tension: float, length: float, stiffness: float) -> float:
    """Compensate reactive bridge loading so ``freq`` names the sounding fundamental."""
    mobility = complex(_mobility(np.asarray(2j * np.pi * freq), sr, size, diffuse)) * coupling
    bare = freq
    for _ in range(6):
        impedance = tension / (2 * length * bare)
        mode = 1 + np.arctan(1j * impedance * mobility).real / np.pi
        bare = freq / (mode * np.sqrt((1 + stiffness * mode*mode) / (1 + stiffness)))
    return float(bare)


def coupled_string(
    freq: float, n: int, sr: int, *, pick: float = .15, hardness: float = .9,
    sustain: float = 6., treble_decay: float = 3., stiffness: float = 4e-5,
    body_size: float = 1.1, diffuse: float = .4, coupling: float = .4,
    radiation: float = .25, tension: float = 70., length: float = .65,
) -> np.ndarray:
    """Release a point-plucked string attached to a compliant, energy-absorbing body.

    ``pick`` is relative to the vibrating length; ``length`` is in metres and
    tension in newtons. ``sustain`` is internal fundamental T60 at A3, before body
    losses. ``coupling`` scales the mobility; radiation tilts bridge velocity
    toward acceleration (0..1). No velocity normalization or sample playback.
    """
    positive = (freq, sustain, treble_decay, body_size, coupling, tension, length)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("frequency, decays, body size, coupling, tension and length must be finite and positive")
    if sr < 4000 or freq >= sr / 4:
        raise ValueError("coupled string requires sr >= 4000 and freq < sr/4")
    if not np.isfinite(pick) or not 0 < pick < 1:
        raise ValueError("pick must be between 0 and 1")
    if not np.isfinite(hardness) or not 0 <= hardness <= 1:
        raise ValueError("hardness must be between 0 and 1")
    if any(not np.isfinite(value) or value < 0 for value in (stiffness, diffuse)):
        raise ValueError("stiffness and diffuse must be finite and nonnegative")
    if not np.isfinite(radiation) or not 0 <= radiation <= 1:
        raise ValueError("radiation must be between 0 and 1")
    if n <= 0:
        return np.zeros(0)
    # A guitar is tuned after the string is attached to its bridge, not before.
    target = freq
    freq = _tuned_frequency(freq, sr, body_size, diffuse, coupling, tension, length, stiffness)
    nfft = next_fast_len(max(n * 2, int(sr * 12)))
    # Evaluate on a right-shifted Laplace contour to suppress circular wrap-around,
    # then undo the exponential window only over the requested output span.
    sigma = np.log(1e5) / (nfft / sr)
    hz = np.fft.rfftfreq(nfft, 1 / sr)
    s = sigma + 2j * np.pi * hz
    mobility = coupling * body_mobility(nfft, sr, sigma, body_size, diffuse)
    ratio = -1j * s / (2 * np.pi * freq)
    if stiffness:
        ratio *= np.sqrt(2 * (1 + stiffness) /
                         (1 + np.sqrt(1 + 4 * stiffness * ratio * ratio * (1 + stiffness))))
    t60 = sustain * (220 / target) ** .3
    loss = (1 / t60 + max(0, 1 / treble_decay - 1 / t60)
            * np.maximum(0, hz**2 - freq**2) / max(1, 4000**2 - freq**2))
    theta = np.pi * ratio - 1j * np.log(1000) * loss / (2 * freq)
    impedance = tension / (2 * length * freq)
    # Combine bridge-force transfer and string input impedance before division,
    # avoiding separate poles of cot(theta) at every rigid-string mode.
    denominator = np.sin(theta) - 1j * impedance * mobility * np.cos(theta)
    response = -(mobility / s) * np.sin(theta * (1 - pick)) / denominator
    cutoff = min(1500 * 2 ** (3 * hardness), .4 * sr)
    response /= (1 + s / (2 * np.pi * cutoff)) ** 2
    response *= (s / (2 * np.pi * 220)) ** radiation
    response *= s / (s + 2 * np.pi * 30)
    if nfft % 2 == 0:
        response[-1] = response[-1].real
    return np.fft.irfft(response, nfft)[:n] * sr * np.exp(sigma * np.arange(n) / sr)
