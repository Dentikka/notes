"""Designed magnetic pickup aperture for a nearly harmonic string waveguide.

The spatial average of a sinusoidal mode contributes a sinc factor; two equal
in-phase coils contribute a cosine factor. These are mathematical FIRs, not
measured responses. The field is approximated as uniform within each aperture.
See https://till.com/articles/PickupResponse/ and
https://till.com/articles/PickupMixing/ for the spatial response derivations.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import oaconvolve

__all__ = ["pickup_kernel", "spatial_pickup"]

_RADIUS = 16


@lru_cache(maxsize=128)
def pickup_kernel(freq: float, sr: int, position: float, width: float = 0.,
                  spacing: float = 0.) -> np.ndarray:
    """Read-only spatial FIR; lengths are fractions of the vibrating string.

    Position is measured from the bridge. Width is each coil's uniform sensing
    aperture; spacing separates two equal in-phase coils (zero selects one).
    Fundamental gain is normalized to unity, independently of output level.
    The returned FIR has 16 samples of numerical lookahead to remove on readout.
    """
    values = (freq, sr, position, width, spacing)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("pickup parameters must be finite")
    if sr < 1000 or int(sr) != sr or not 0 < freq < .45*sr:
        raise ValueError("pickup needs an integer sample rate and frequency below 0.45*sr")
    if not 0 < position < 1:
        raise ValueError("pickup position must lie inside the vibrating string")
    if width < 0 or spacing < 0:
        raise ValueError("invalid pickup geometry")
    extent = (width+spacing)/2
    if position-extent < 0 or position+extent > 1:
        raise ValueError("pickup aperture must lie inside the vibrating string")
    gain = 2*np.sin(np.pi*position)*np.cos(np.pi*spacing/2)*np.sinc(width/2)
    if abs(gain) < 1e-8:
        raise ValueError("pickup has negligible fundamental sensitivity")
    nodes, weights = np.polynomial.legendre.leggauss(16) if width else (np.array([0.]), np.array([2.]))
    centers = (position-spacing/2, position+spacing/2) if spacing else (position,)
    reference = position+extent
    delays, coefficients = [], []
    for center in centers:
        for node, weight in zip(nodes, weights, strict=True):
            at = center+width*node/2
            amount = weight/2/len(centers)
            delays.extend(((reference-at)*sr/(2*freq), (reference+at)*sr/(2*freq)))
            coefficients.extend((amount, -amount))
    count = int(np.ceil(max(delays)))+2*_RADIUS+1
    if count > 65536:
        raise ValueError("pickup delay exceeds the supported filter length")
    distance = np.arange(count)[None, :]-(np.array(delays)[:, None]+_RADIUS)
    window = (.42+.5*np.cos(np.pi*distance/_RADIUS)+.08*np.cos(2*np.pi*distance/_RADIUS))
    window *= abs(distance) < _RADIUS
    taps = np.sinc(distance)*window
    taps /= taps.sum(axis=1, keepdims=True)
    response = np.array(coefficients)@taps/gain
    response.flags.writeable = False
    return response


def spatial_pickup(signal: np.ndarray, freq: float, sr: int, position: float, width: float = 0.,
                   spacing: float = 0.) -> np.ndarray:
    """Sense a mono source without integer-rounded geometry or an extra 16-sample delay."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("pickup source must be finite mono audio")
    impulse = pickup_kernel(freq, sr, position, width, spacing)
    if not signal.size:
        return signal.copy()
    result = oaconvolve(signal, impulse)
    return result[_RADIUS:_RADIUS+len(signal)]
