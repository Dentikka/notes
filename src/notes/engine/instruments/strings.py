"""Plucked strings by extended Karplus–Strong, and an electric guitar built on them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.dsp import adsr, delay, filt, karplus_strong, vel_gain
from notes.engine.effects.tone import Cabinet, GuitarAmp
from notes.engine.registry import register_instrument

__all__ = ["ElectricGuitar", "Pluck"]

_PICKUP_POSITION = {"bridge": 0.09, "middle": 0.17, "neck": 0.27}


def _t60(sustain: float, freq: float) -> float:
    """Lower strings ring longer: T60 scales mildly with pitch around A3."""
    return sustain * (220.0 / freq) ** 0.3


def _gate(gate: int, n: int, sr: int, release: float) -> np.ndarray:
    """Unity while the note is held, then a smooth damping to zero over `release` seconds."""
    return adsr(gate, n, sr, 0.001, 1.0, 1.0, release)


@register_instrument("pluck")
@dataclass(frozen=True, kw_only=True)
class Pluck(Instrument):
    """A bare plucked string with a hint of wooden body — harp, koto, nylon-ish.

    `sustain` is T60 in seconds at A3, `brightness` the pluck's spectral tilt, `pick` the
    pluck position along the string (0.5 = middle: soft, hollow).
    """

    program: int | None = 24
    sustain: float = 3.0
    brightness: float = 0.5
    pick: float = 0.2
    body: float = 0.5
    release: float = 0.08
    level: float = 0.5

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        t60 = _t60(self.sustain, v.freq)
        y = karplus_strong(v.freq, n, ctx.sr, rng, t60=t60, brightness=self.brightness, pick=self.pick)
        if self.body:
            y = filt(y, "peak", 190.0, ctx.sr, 1.2, 6.0 * self.body)
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, ctx.sr, self.release)


@register_instrument("electric_guitar")
@dataclass(frozen=True, kw_only=True)
class ElectricGuitar(Instrument):
    """Karplus–Strong strings through a magnetic pickup into a modelled amp and cabinet.

    `drive` 0 is clean and 1 heavy distortion; `tone` darkens (0) or brightens (1) the amp and
    cabinet; `pickup` is 'neck' (round), 'middle' or 'bridge' (bright, nasal). The per-note
    parameter ``palm_mute=True`` gives short, dark chugs.
    """

    drive: float = 0.5
    tone: float = 0.5
    pickup: str = "bridge"
    sustain: float = 2.5
    brightness: float = 0.7
    release: float = 0.06
    level: float = 0.5

    def __post_init__(self) -> None:
        if self.pickup not in _PICKUP_POSITION:
            raise ValueError(f"pickup must be one of {', '.join(_PICKUP_POSITION)}, got {self.pickup!r}")

    def release_time(self) -> float:
        return self.release

    def inserts(self) -> tuple[Effect, ...]:
        return (GuitarAmp(drive=self.drive, tone=self.tone), Cabinet(tone=self.tone))

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        muted = bool(v.params.get("palm_mute", False))
        t60 = 0.12 if muted else _t60(self.sustain, v.freq)
        brightness = self.brightness * (0.45 if muted else 1.0)
        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        y = karplus_strong(v.freq, n, ctx.sr, rng, t60=t60, brightness=brightness, pick=0.18)
        # The pickup senses the string at one point: a comb that shapes the harmonics.
        position = _PICKUP_POSITION[self.pickup]
        lag = int(round(position * ctx.sr / v.freq))
        if lag >= 1:
            y = (y - delay(y, lag)) / (2.0 * np.sin(np.pi * position))
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, ctx.sr, self.release)
