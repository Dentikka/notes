"""Plucked strings by extended Karplus–Strong, and an electric guitar built on them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.articulation import pitch_curve, variable_rate_read
from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.dsp import adsr, delay, filt, karplus_strong, vel_gain
from notes.engine.effects.tone import Cabinet, Filter, GuitarAmp
from notes.engine.registry import register_instrument

__all__ = ["ElectricGuitar", "Pluck"]

#: Where each pickup senses the string (fraction of its length from the bridge).
_PICKUP_POSITION = {"bridge": 0.09, "middle": 0.17, "neck": 0.27}
#: Resonance of the pickup's inductance with the cable capacitance: (Hz, Q).
_PICKUP_RESONANCE = {"bridge": (4000.0, 1.4), "middle": (3600.0, 1.3), "neck": (3000.0, 1.2)}


def _t60(sustain: float, freq: float) -> float:
    """Lower strings ring longer: T60 scales mildly with pitch around A3."""
    return sustain * (220.0 / freq) ** 0.3


def _gate(gate: int, n: int, sr: int, release: float) -> np.ndarray:
    """Unity while the note is held, then a smooth damping to zero over `release` seconds."""
    return adsr(gate, n, sr, 0.001, 1.0, 1.0, release)


def _pick_click(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """The pick leaving the string: a few milliseconds of band-limited noise."""
    k = min(n, int(0.012 * sr))
    click = np.zeros(n)
    t = np.arange(k) / sr
    click[:k] = filt(rng.uniform(-1.0, 1.0, k), "bandpass", 3500.0, sr, 0.8) * np.exp(-t / 0.0025)
    return click


def _string(v: Voice, n: int, sr: int, render_at_fixed_pitch) -> np.ndarray:
    """Render n samples, following the note's vibrato, bend or slide if it has one."""
    curve = pitch_curve(v.params, n, sr)
    return render_at_fixed_pitch(n) if curve is None else variable_rate_read(render_at_fixed_pitch, curve)


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
    level: float = 0.84

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        t60 = _t60(self.sustain, v.freq)
        y = _string(v, n, ctx.sr, lambda m: karplus_strong(
            v.freq, m, ctx.sr, rng, t60=t60, brightness=self.brightness, pick=self.pick))
        if self.body:
            y = filt(y, "peak", 190.0, ctx.sr, 1.2, 6.0 * self.body)
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, ctx.sr, self.release)


@register_instrument("electric_guitar")
@dataclass(frozen=True, kw_only=True)
class ElectricGuitar(Instrument):
    """Karplus–Strong strings through a magnetic pickup into a modelled amp and cabinet.

    `drive` 0 is clean and 1 heavy distortion; `tone` darkens (0) or brightens (1) the amp and
    cabinet; `pickup` is 'neck' (round), 'middle' or 'bridge' (bright, nasal); `pick_noise` is
    the level of the pick's click. Per-note parameters: ``palm_mute=True`` for chugs that
    ring `mute_decay` seconds (T60) with the pluck darkened to `mute_brightness`, and
    ``vibrato``, ``bend``, ``slide`` (see `notes.engine.articulation`).
    """

    drive: float = 0.5
    tone: float = 0.5
    pickup: str = "bridge"
    sustain: float = 2.5
    brightness: float = 0.7
    pick_noise: float = 0.2
    mute_decay: float = 0.5
    mute_brightness: float = 0.75
    release: float = 0.06
    level: float = 0.5

    def __post_init__(self) -> None:
        if self.pickup not in _PICKUP_POSITION:
            raise ValueError(f"pickup must be one of {', '.join(_PICKUP_POSITION)}, got {self.pickup!r}")

    def release_time(self) -> float:
        return self.release

    def inserts(self) -> tuple[Effect, ...]:
        freq, q = _PICKUP_RESONANCE[self.pickup]
        return (
            Filter(kind="lowpass", cutoff=freq, q=q),
            GuitarAmp(drive=self.drive, tone=self.tone),
            Cabinet(tone=self.tone),
        )

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        muted = bool(v.params.get("palm_mute", False))
        t60 = self.mute_decay if muted else _t60(self.sustain, v.freq)
        brightness = self.brightness * (self.mute_brightness if muted else 1.0)
        position = _PICKUP_POSITION[self.pickup]
        lag = int(round(position * sr / v.freq))

        def string(m: int) -> np.ndarray:
            y = karplus_strong(v.freq, m, sr, rng, t60=t60, brightness=brightness, pick=0.18)
            # The pickup senses the string at one point: a comb that shapes the harmonics.
            return (y - delay(y, lag)) / (2.0 * np.sin(np.pi * position)) if lag >= 1 else y

        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        y = _string(v, n, sr, string) + self.pick_noise * _pick_click(n, sr, rng)
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, sr, self.release)
