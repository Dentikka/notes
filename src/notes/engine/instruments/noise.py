"""Noise for as long as a note is held: the tape hiss under a quiet intro, wind, a riser."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.dsp import adsr, filt, vel_gain
from notes.engine.registry import register_instrument

__all__ = ["Noise"]

#: Spectral slope of each colour, as the exponent of frequency in the power spectrum.
_SLOPES = {"white": 0.0, "pink": -1.0, "brown": -2.0}


def coloured_noise(n: int, sr: int, color: str, rng: np.random.Generator) -> np.ndarray:
    """n samples of unit-RMS noise whose power falls as f^0 (white), f^-1 (pink: equal energy
    per octave, as tape hiss) or f^-2 (brown), shaped in the frequency domain."""
    if color not in _SLOPES:
        raise ValueError(f"color must be one of {', '.join(_SLOPES)}, got {color!r}")
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    spec *= np.maximum(f, 20.0) ** (_SLOPES[color] / 2.0)
    y = np.fft.irfft(spec, n)
    return y / max(float(np.sqrt(np.mean(y**2))), 1e-12)


@register_instrument("noise")
@dataclass(frozen=True, kw_only=True)
class Noise(Instrument):
    """Coloured noise for the length of each note (the pitch is ignored): `color` 'white',
    'pink' or 'brown', band-limited to `low`-`high` Hz, faded in over `attack` and out over
    `release` seconds; `level` is its RMS at full velocity. Two of them panned apart give the
    uncorrelated hiss of a stereo tape."""

    color: str = "pink"
    low: float = 20.0
    high: float = 20000.0
    attack: float = 0.05
    release: float = 0.1
    level: float = 0.1

    def __post_init__(self) -> None:
        if self.color not in _SLOPES:
            raise ValueError(f"color must be one of {', '.join(_SLOPES)}, got {self.color!r}")

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        y = coloured_noise(n, ctx.sr, self.color, rng)
        if self.low > 20.0:
            y = filt(y, "highpass", self.low, ctx.sr, 0.7071)
        if self.high < 0.45 * ctx.sr:
            y = filt(y, "lowpass", self.high, ctx.sr, 0.7071)
        return self.level * vel_gain(v.vel) * y * adsr(gate, n, ctx.sr, self.attack, 1.0, 1.0, self.release)
