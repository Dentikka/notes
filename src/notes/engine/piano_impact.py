"""Procedural nonharmonic piano impact: designed modes, synthetic positive pulse.

No waveform or measured impulse response is read. This is an independent output
attack component, not an identified or energy-coupled piano action mechanism.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ['impact_duration', 'piano_impact']


def _impact_controls(decay: float, color: float, velocity_power: float, delay: float) -> None:
    try:
        finite = bool(np.isfinite((decay, color, velocity_power, delay)).all())
    except TypeError as error:
        raise ValueError('impact controls must be finite numbers') from error
    if not finite:
        raise ValueError('impact controls must be finite')
    if not .03 <= decay <= .5 or not -1 <= color <= 2:
        raise ValueError('impact decay must be in [.03,.5] and color in [-1,2]')
    if not 0 <= velocity_power <= 2 or not 0 <= delay <= .06:
        raise ValueError('impact velocity power must be in [0,2] and delay in [0,.06]')


def impact_duration(decay: float, delay: float = 0.) -> float:
    """Conservative seconds of modal tail, including pulse/discretization margin."""
    _impact_controls(decay, .5, 1., delay)
    return 1.5*decay*(200/70)**.3+.005+delay


@lru_cache(maxsize=16)
def _kernel(sr: int, decay: float, color: float) -> np.ndarray:
    index = np.arange(48, dtype=float)
    frequencies = np.geomspace(70., 1350., 48)*(1+.02*np.sin(index*2.399))
    keep = frequencies < .42*sr
    index, frequencies = index[keep], frequencies[keep]
    t60 = decay*(200/frequencies)**.3
    length = max(2, int(np.ceil(1.5*t60.max()*sr)))
    time = np.arange(length)/sr
    wave = np.zeros(length)
    participation = np.sin((index+1)*2.399)+.35*np.cos((index+1)*.873)
    participation *= frequencies**2/(frequencies**2+120**2)*(frequencies/200)**(-color)
    for frequency, lifetime, weight in zip(frequencies, t60, participation, strict=True):
        wave += weight*np.sin(2*np.pi*frequency*time)*np.exp(-np.log(1000)*time/lifetime)
    fade = min(length, max(2, round(.005*sr)))
    wave[-fade:] *= np.linspace(1, 0, fade)
    pulse = np.hanning(max(4, round(.001*sr)))
    pulse /= pulse.sum()
    reference = np.convolve(wave, pulse)
    norm = np.sqrt(np.sum(reference*reference)/sr)
    wave /= max(norm, 1e-30)
    wave.flags.writeable = False
    return wave


def piano_impact(n: int, sr: int, velocity: float, rng: np.random.Generator, *,
                 decay: float = .18, color: float = .5, velocity_power: float = 1.,
                 delay: float = 0.) -> np.ndarray:
    """Return a causal modal impact, normalized once by a designed reference pulse.

    Decay is T60 at 200 Hz. Color tilts modal amplitudes; velocity_power controls
    positive MIDI velocity response. The seeded positive 1 ms pulse has unit
    area; no per-note output peak/RMS normalization is performed. Zero velocity
    and empty output consume no random draws. Prefixes are length-independent.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    if isinstance(sr, bool) or not isinstance(sr, int) or sr < 1000:
        raise ValueError('sr must be an integer >=1000')
    _impact_controls(decay, color, velocity_power, delay)
    if not np.isfinite(velocity) or not 0 <= velocity <= 1:
        raise ValueError('impact velocity must be finite and in [0,1]')
    if n == 0 or velocity == 0:
        return np.zeros(n)
    pulse = np.hanning(max(4, round(.001*sr)))
    pulse *= np.maximum(.05, 1+.35*rng.normal(size=len(pulse)))
    pulse /= pulse.sum()
    raw = np.convolve(_kernel(sr, float(decay), float(color)), pulse)
    output = np.zeros(n)
    count = min(n, int(np.ceil(len(raw)+delay*sr)))
    position = np.arange(count)-delay*sr
    output[:count] = np.interp(position, np.arange(len(raw)), raw, left=0., right=0.)
    return output*velocity**velocity_power
