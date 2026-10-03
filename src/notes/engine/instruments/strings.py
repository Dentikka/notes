"""Plucked strings: Karplus–Strong for the plain `Pluck`, a steel-string waveguide for the
electric guitar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.articulation import pitch_curve, variable_rate_read
from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.dsp import adsr, delay, filt, karplus_strong, vel_gain
from notes.engine.effects.tone import Cabinet, Drive, Filter, GuitarAmp
from notes.engine.registry import register_instrument
from notes.engine.string_model import steel_string

__all__ = ["BassGuitar", "ElectricGuitar", "Pluck"]

#: Where each pickup senses the string (fraction of its length from the bridge).
_PICKUP_POSITION = {"bridge": 0.09, "middle": 0.17, "neck": 0.27}
#: Resonance of the pickup's inductance with the cable capacitance: (Hz, Q).
_PICKUP_RESONANCE = {"bridge": (4000.0, 1.4), "middle": (3600.0, 1.3), "neck": (3000.0, 1.2)}
#: Time constant (s) of the pitch falling back after a pluck stretches the string.
_TENSION_TAU = 0.1


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
    """Steel strings (`notes.engine.string_model`) through a magnetic pickup into a modelled
    amp and cabinet.

    `drive` 0 is clean and 1 heavy distortion; `tone` darkens (0) or brightens (1) the amp and
    cabinet; `cabinet` is the `Cabinet` model ('greenback' impulse response or the smoother
    'filters'); `pickup` is 'neck' (round), 'middle' or 'bridge' (bright, nasal); `pick_noise`
    is the level of the pick's click. The string: `sustain` is the T60 (s) of the fundamental
    at A3 (longer below, shorter above), `treble_decay` the T60 of partials near 4 kHz,
    `brightness` the hardness of the pick, `pick_position` where it strikes (fraction of the
    string from the bridge), `stiffness` the inharmonicity B, `polarization` the level of
    the string's second, slightly detuned vibration, `pitch_attack` how many cents sharp a
    full-velocity pluck starts (tension falling back over ~0.1 s).

    Per-note parameters: ``palm_mute=True`` for chugs that ring `mute_decay` seconds (T60)
    with the pluck softened to `mute_brightness`, and ``vibrato``, ``bend``, ``slide`` (see
    `notes.engine.articulation`; vibrato is pushed across the fret, only ever sharpening).
    """

    drive: float = 0.5
    tone: float = 0.5
    cabinet: str = "greenback"
    pickup: str = "bridge"
    sustain: float = 9.0
    treble_decay: float = 2.5
    brightness: float = 0.7
    pick_position: float = 0.15
    stiffness: float = 4e-5
    polarization: float = 0.3
    pitch_attack: float = 8.0
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
            Cabinet(tone=self.tone, model=self.cabinet),
        )

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        muted = bool(v.params.get("palm_mute", False))
        t60 = self.mute_decay if muted else _t60(self.sustain, v.freq)
        t60_high = min(self.treble_decay, 0.3 * t60) if muted else self.treble_decay
        hardness = self.brightness * (self.mute_brightness if muted else 1.0)
        position = _PICKUP_POSITION[self.pickup]
        lag = int(round(position * sr / v.freq))

        def string(m: int) -> np.ndarray:
            y = steel_string(v.freq, m, sr, rng, t60=t60, t60_high=t60_high, pick=self.pick_position,
                             hardness=hardness, stiffness=self.stiffness, second=self.polarization)
            # The pickup senses the string at one point: a comb that shapes the harmonics.
            return (y - delay(y, lag)) / (2.0 * np.sin(np.pi * position)) if lag >= 1 else y

        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        curve = pitch_curve(v.params, n, sr, vibrato_shape="push", rng=rng)
        if self.pitch_attack:
            glide = self.pitch_attack / 100.0 * v.vel * np.exp(-np.arange(n) / (_TENSION_TAU * sr))
            curve = glide if curve is None else curve + glide
        y = string(n) if curve is None else variable_rate_read(string, curve)
        y = y + self.pick_noise * _pick_click(n, sr, rng)
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, sr, self.release)


@register_instrument("bass_guitar")
@dataclass(frozen=True, kw_only=True)
class BassGuitar(Instrument):
    """Electric bass: steel-string waveguide (`notes.engine.string_model`), played with the
    fingers, through a pickup straight into the desk (no cabinet).

    `sustain` is the T60 (s) of the fundamental at A3 (longer on the low strings),
    `treble_decay` the T60 near 4 kHz; `brightness` is the hardness of the attack (fingers
    ~0.4, pick ~0.8); `tone` sets the low-pass of the DI from dark (0) to growly (1); `drive`
    a little preamp saturation. Per-note ``palm_mute=True`` gives a muted thump.
    """

    program: int | None = 33
    sustain: float = 5.0
    treble_decay: float = 0.6
    brightness: float = 0.4
    pick_position: float = 0.22
    stiffness: float = 5e-5
    polarization: float = 0.2
    pitch_attack: float = 4.0
    tone: float = 0.5
    drive: float = 0.1
    release: float = 0.08
    level: float = 0.19

    def release_time(self) -> float:
        return self.release

    def inserts(self) -> tuple[Effect, ...]:
        chain: tuple[Effect, ...] = (Filter(kind="lowpass", cutoff=900.0 + 3000.0 * self.tone, q=0.8),)
        if self.drive > 0:
            chain += (Drive(amount=self.drive, tone=2500.0 + 3000.0 * self.tone),)
        return chain

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        muted = bool(v.params.get("palm_mute", False))
        t60 = 0.4 if muted else _t60(self.sustain, v.freq)
        position = 0.2
        lag = int(round(position * sr / v.freq))

        def string(m: int) -> np.ndarray:
            y = steel_string(v.freq, m, sr, rng, t60=t60, t60_high=min(self.treble_decay, t60),
                             pick=self.pick_position, hardness=self.brightness * (0.7 if muted else 1.0),
                             noise=0.05, stiffness=self.stiffness, second=self.polarization)
            return (y - delay(y, lag)) / (2.0 * np.sin(np.pi * position)) if lag >= 1 else y

        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        curve = pitch_curve(v.params, n, sr, vibrato_shape="push", rng=rng)
        if self.pitch_attack:
            glide = self.pitch_attack / 100.0 * v.vel * np.exp(-np.arange(n) / (_TENSION_TAU * sr))
            curve = glide if curve is None else curve + glide
        y = string(n) if curve is None else variable_rate_read(string, curve)
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, sr, self.release)
