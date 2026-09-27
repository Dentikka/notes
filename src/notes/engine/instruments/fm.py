"""Frequency-modulation synthesis: electric pianos, bells and other bright, evolving tones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.dsp import TAU, adsr, vel_gain
from notes.engine.registry import register_instrument

__all__ = ["FM", "Bell", "EPiano"]


@register_instrument("fm")
@dataclass(frozen=True, kw_only=True)
class FM(Instrument):
    """Two-operator FM: a sine carrier phase-modulated by a sine at `ratio` × the note frequency.

    The modulation index — the brightness — starts at `index` (scaled by velocity) and decays
    with time constant `index_decay`. Integer ratios give harmonic spectra, others inharmonic
    ones (bells). An optional second pair at `tine_ratio` adds a short metallic attack.
    """

    program: int | None = 4
    ratio: float = 1.0
    index: float = 2.0
    index_decay: float = 0.5
    tine: float = 0.0
    tine_ratio: float = 14.0
    attack: float = 0.002
    decay: float = 1.5
    sustain: float = 0.0
    release: float = 0.3
    level: float = 0.4

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        t = np.arange(n) / sr
        w = TAU * v.freq * t
        index = self.index * (0.4 + 0.6 * v.vel) * np.exp(-t / max(self.index_decay, 1e-3))
        y = np.sin(w + index * np.sin(self.ratio * w))
        if self.tine:
            tine_index = 2.0 * v.vel * np.exp(-t / 0.03)
            y += self.tine * np.sin(w + tine_index * np.sin(self.tine_ratio * w)) * np.exp(-t / 0.6)
        env = adsr(gate, n, sr, self.attack, self.decay, self.sustain, self.release)
        return self.level * vel_gain(v.vel) * env * y


@register_instrument("epiano")
@dataclass(frozen=True, kw_only=True)
class EPiano(FM):
    """Tine electric piano in the DX7 tradition: warm body, bell-like bark on hard notes."""

    index: float = 1.6
    index_decay: float = 0.35
    tine: float = 0.35
    decay: float = 2.5
    release: float = 0.25


@register_instrument("bell")
@dataclass(frozen=True, kw_only=True)
class Bell(FM):
    """Inharmonic FM bell (ratio 3.5) with a long ring."""

    program: int | None = 14
    ratio: float = 3.5
    index: float = 2.5
    index_decay: float = 1.2
    decay: float = 4.0
    release: float = 1.5
    level: float = 0.3
