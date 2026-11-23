"""Analytic modes of a generated waveguide, with independently scaled positive losses.

No sampled waveforms or measured phases are used. Residues are evaluated from
the loop and its mathematical excitation, avoiding high-order polynomial deflation.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import freqz


def decay_scales(value: object) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) > 80:
        raise ValueError('partial_t60_scales needs at most 80 finite positive factors')
    result = []
    for item in value:
        try:
            scale = float(item)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError('partial_t60_scales needs finite positive factors') from error
        if isinstance(item, (bool, np.bool_)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('partial_t60_scales needs finite positive factors')
        with np.errstate(over='ignore'):
            if not np.isfinite(1/scale):
                raise ValueError('partial_t60_scales produces a nonfinite damping rate')
        result.append(scale)
    return tuple(result)


def _response(p: np.ndarray, b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = 1/p
    numerator = np.polynomial.polynomial.polyval(q, b)
    denominator = np.polynomial.polynomial.polyval(q, a)
    bd = np.polynomial.polynomial.polyval(q, np.arange(1, len(b))*b[1:]) if len(b) > 1 else 0.
    ad = np.polynomial.polynomial.polyval(q, np.arange(1, len(a))*a[1:]) if len(a) > 1 else 0.
    # -p H'(p)/H(p); this also gives the group delay's real part on the unit circle.
    delay = q*(bd/numerator-ad/denominator)
    return numerator/denominator, delay


@lru_cache(maxsize=128)
def loop_modes(length: int, b: tuple[float, ...], a: tuple[float, ...], count: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Positive-frequency poles and residue factors, ordered from the fundamental.

    The filter must have the steel waveguide's monotone total phase and stable
    contractive loop. Real DC/Nyquist poles are deliberately not edited.
    """
    bb, aa = np.asarray(b), np.asarray(a)
    dc = float(np.sum(bb)/np.sum(aa))
    if not np.isfinite(dc) or not 0 < dc < 1:
        raise ValueError('modal loss control requires a strictly contractive waveguide loop')
    w = np.linspace(0., np.pi, 8193)
    _, h = freqz(bb, aa, worN=w)
    phase = length*w-np.unwrap(np.angle(h))
    if not np.isfinite(phase).all() or np.any(np.diff(phase) <= 0):
        raise ValueError('modal loss control needs a monotone loop phase')
    turns = 2*np.pi*np.arange(1, count+1)
    turns = turns[turns < phase[-1]-1e-9]
    omega = np.interp(turns, phase, w)
    unit = np.exp(1j*omega)
    response, delay = _response(unit, bb, aa)
    p = np.exp(np.log(abs(response))/(length+delay.real)+1j*omega)
    for _ in range(20):
        response, delay = _response(p, bb, aa)
        ratio = response/np.power(p, length)
        step = (1-ratio)/(ratio*(length+delay)/p)
        p = p-step
        if not len(step) or np.max(abs(step)) < 1e-13:
            break
    response, delay = _response(p, bb, aa)
    residual = abs(1-response/np.power(p, length))
    if (not np.isfinite(p).all() or not np.isfinite(delay).all()
            or np.any(residual > 1e-7) or np.any(abs(p) >= 1)
            or np.any(p.imag <= 0) or np.any(np.diff(np.angle(p)) <= 0)):
        raise ValueError('modal loss control could not resolve stable distinct waveguide modes')
    factors = 1/(length+delay)
    p.flags.writeable = factors.flags.writeable = False
    return p, factors


def decay_correction(burst: np.ndarray, length: int, b: np.ndarray, a: np.ndarray,
                     scales: tuple[float, ...], n: int) -> np.ndarray:
    """Change selected modal radii, retaining frequencies and initial modal coefficients.

    Each factor multiplies the native T60 of that mode, separately in each
    polarisation. Unedited modes and the direct transient remain in the caller's
    original waveguide signal. This is stable modal damping, not a proof of the
    mechanical energy or radiation model of a real string.
    """
    correction = np.zeros(n)
    if not n or all(scale == 1 for scale in scales):
        return correction
    poles, factors = loop_modes(length, tuple(b), tuple(a), len(scales))
    coefficients = np.polynomial.polynomial.polyval(1/poles, burst)*factors
    time = np.arange(n)
    for pole, coefficient, scale in zip(poles, coefficients, scales[:len(poles)], strict=True):
        if scale == 1:
            continue
        alpha = -float(np.log(abs(pole)))
        rate = alpha/scale
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('partial_t60_scales produces a nonfinite damping rate')
        angle = float(np.angle(pole))
        before = np.exp(time*(-alpha+1j*angle))
        after = np.exp(time*(-rate+1j*angle))
        correction += 2*(coefficient*(after-before)).real
    return correction
