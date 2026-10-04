"""Plucked strings: Karplus–Strong for the plain `Pluck`, a steel-string waveguide for the
electric guitar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.articulation import pitch_curve, variable_rate_read
from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.dsp import adsr, delay, filt, karplus_strong, vel_gain
from notes.engine.effects.tone import EQ, Cabinet, Drive, Filter, GuitarAmp
from notes.engine.registry import register_instrument
from notes.engine.string_model import steel_string

__all__ = ["BassGuitar", "ElectricGuitar", "Pluck"]

#: Where each pickup senses the string (fraction of its length from the bridge).
_PICKUP_POSITION = {"bridge": 0.09, "middle": 0.17, "neck": 0.27}
#: Resonance of the pickup's inductance with the cable capacitance: (Hz, Q).
_PICKUP_RESONANCE = {"bridge": (4000.0, 1.4), "middle": (3600.0, 1.3), "neck": (3000.0, 1.2)}
#: Open strings of a guitar in standard tuning (MIDI), low to high.
_STANDARD_TUNING = (40, 45, 50, 55, 59, 64)
_MAX_FRET = 24
#: A note sounded by the fretting hand: its excitation is this much softer and quieter.
_LEGATO_SOFTNESS = 0.6
_LEGATO_LEVEL = 0.7
#: Time constant (s) of the pitch falling back after a pluck stretches the string.
_TENSION_TAU = 0.1
#: Strings 4-6 (D, A, low E) are wound on a standard electric set; G and up are plain. A note
#: without a string is taken as wound below G3, where only those strings reach.
_WOUND_STRINGS, _LOWEST_PLAIN = (4, 5, 6), 55


def _t60(sustain: float, freq: float) -> float:
    """Lower strings ring longer: T60 scales mildly with pitch around A3."""
    return sustain * (220.0 / freq) ** 0.3


def _gate(gate: int, n: int, sr: int, release: float) -> np.ndarray:
    """Unity while the note is held, then a smooth damping to zero over `release` seconds."""
    return adsr(gate, n, sr, 0.001, 1.0, 1.0, release)


def _pick_click(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """The pick striking and slipping off the string: ~5 ms of broadband noise, 0.7–9 kHz,
    with a 0.3 ms rise and a 1.5 ms decay, peak-normalised (so `pick_noise` 1 matches the
    string's own peak)."""
    k = min(n, int(0.006 * sr))
    click = np.zeros(n)
    if k < 2:
        return click
    t = np.arange(k) / sr
    burst = filt(filt(rng.uniform(-1.0, 1.0, k), "highpass", 700.0, sr, 0.7), "lowpass", 9000.0, sr, 0.7)
    burst *= (1.0 - np.exp(-t / 0.0003)) * np.exp(-t / 0.0015)
    click[:k] = burst / max(float(np.max(np.abs(burst))), 1e-9)
    return click


def fret_scale(pitch: float) -> float:
    """How much shorter than the open string the vibrating length is for this note.

    The note goes on the highest string that can play it (lowest fret), as melodies are
    usually played. The pick and the pickups sit at fixed distances from the bridge, so
    as a fretted string gets shorter they sit relatively further along it: at the 12th
    fret the neck pickup is near the middle of the string, which is why high notes on the
    neck pickup sound round and hollow.
    """
    open_string = max((s for s in _STANDARD_TUNING if s <= pitch), default=_STANDARD_TUNING[0])
    fret = float(np.clip(pitch - open_string, 0.0, _MAX_FRET))
    return 2.0 ** (-fret / 12.0)


def _along(position: float, scale: float) -> float:
    """A point `position` (fraction of the open string) as a fraction of a string `scale`
    as long, folded into (0, 0.5]: a comb at p is the same as at 1 - p."""
    p = min(position / scale, 0.99)
    return float(np.clip(min(p, 1.0 - p), 0.01, 0.5))


def _finger_drift(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A slow, smooth random wander of unit peak: the fretting finger's pressure."""
    t = np.arange(n) / sr
    wander = sum(np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi)) / k
                 for k, f in enumerate((0.6, 1.7, 3.1), start=1))
    return wander / (1.0 + 1.0 / 2 + 1.0 / 3)


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
    the string's second, slightly detuned vibration (with `prompt_decay` > 0, the long
    "aftersound" that remains once the picked vibration, ringing `prompt_decay` x the
    sustain, has died: the two-stage decay of a pluck), `pitch_attack` how many cents sharp
    a full-velocity pluck starts (tension falling back over ~0.1 s), `pitch_drift` the
    cents a fretting finger lets the pitch wander, `velocity_brightness` how much softer
    picking also rounds the pluck (0: velocity sets only the level; 1: a note at velocity
    0.3 is picked with 0.3 x the hardness, as a gently picked string sounds darker);
    `wound_treble` and `wound_brightness` scale the treble decay and the pluck's hardness on
    the wound strings (D, A, low E), which lose their highs faster than the plain ones (1:
    every string alike). With `fretting`, each note is placed on a string and fret
    (`fret_scale`) and the pick and pickup positions are taken along the shortened string;
    otherwise they are the same fractions for every note.

    Per-note parameters: ``palm_mute=True`` for chugs that ring `mute_decay` seconds (T60)
    with the pluck softened to `mute_brightness`; ``legato=True`` for a note sounded by the
    fretting hand (hammer-on, pull-off, the end of a slide): no pick click, a softer and
    quieter start, no pitch kick; ``string=1..6`` (high e to low E) for which string a note
    is on, otherwise wound below G3; and ``vibrato``, ``bend``, ``slide`` (see
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
    prompt_decay: float = 0.0
    pitch_attack: float = 8.0
    pitch_drift: float = 0.0
    fretting: bool = False
    pick_noise: float = 0.2
    velocity_brightness: float = 0.0
    wound_treble: float = 1.0
    wound_brightness: float = 1.0
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
        legato = bool(v.params.get("legato", False))
        t60 = self.mute_decay if muted else _t60(self.sustain, v.freq)
        t60_high = min(self.treble_decay, 0.3 * t60) if muted else self.treble_decay
        hardness = self.brightness * (self.mute_brightness if muted else 1.0) * (_LEGATO_SOFTNESS if legato else 1.0)
        hardness *= 1.0 - self.velocity_brightness * (1.0 - v.vel)
        string = v.params.get("string")
        if (string in _WOUND_STRINGS) if string is not None else v.pitch < _LOWEST_PLAIN:
            t60_high *= self.wound_treble
            hardness *= self.wound_brightness
        scale = fret_scale(v.pitch) if self.fretting else 1.0
        position = _along(_PICKUP_POSITION[self.pickup], scale)
        pick = _along(self.pick_position, scale)
        lag = int(round(position * sr / v.freq))

        def string(m: int) -> np.ndarray:
            prompt = self.prompt_decay * t60 if self.prompt_decay > 0 and not muted else None
            y = steel_string(v.freq, m, sr, rng, t60=t60, t60_high=t60_high, pick=pick,
                             hardness=hardness, stiffness=self.stiffness, second=self.polarization,
                             prompt_t60=prompt)
            # The pickup senses the string at one point: a comb that shapes the harmonics.
            return (y - delay(y, lag)) / (2.0 * np.sin(np.pi * position)) if lag >= 1 else y

        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        curve = pitch_curve(v.params, n, sr, vibrato_shape="push", rng=rng)
        if self.pitch_attack and not legato:
            glide = self.pitch_attack / 100.0 * v.vel * np.exp(-np.arange(n) / (_TENSION_TAU * sr))
            curve = glide if curve is None else curve + glide
        if self.pitch_drift:
            drift = _finger_drift(n, sr, rng) * self.pitch_drift / 100.0
            curve = drift if curve is None else curve + drift
        y = string(n) if curve is None else variable_rate_read(string, curve)
        # A hammered or pulled note: quieter, and no pick to click.
        y = y * _LEGATO_LEVEL if legato else y + self.pick_noise * _pick_click(n, sr, rng)
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, sr, self.release)


@register_instrument("bass_guitar")
@dataclass(frozen=True, kw_only=True)
class BassGuitar(Instrument):
    """Electric bass: steel-string waveguide (`notes.engine.string_model`), played with the
    fingers, through a pickup straight into the desk (no cabinet).

    `sustain` is the T60 (s) of the fundamental at A3 (longer on the low strings),
    `treble_decay` the T60 near 4 kHz; `brightness` is the hardness of the attack (fingers
    ~0.4, pick ~0.8); `tone` sets the low-pass of the DI from dark (0) to growly (1); `drive`
    a little preamp saturation. `clank` is the level of the wound string slapping the frets
    on a pluck; `cabinet` sends it through a bass amp's speaker (low lift, a growl around
    800 Hz, nothing above ~4 kHz) instead of straight into the desk. Per-note
    ``palm_mute=True`` gives a muted thump, ``legato=True`` a note without the pitch kick
    and the clank of a pluck.
    """

    program: int | None = 33
    sustain: float = 5.0
    treble_decay: float = 0.1
    brightness: float = 0.4
    pick_position: float = 0.22
    stiffness: float = 5e-5
    polarization: float = 0.2
    pitch_attack: float = 15.0
    tone: float = 0.5
    drive: float = 0.1
    clank: float = 0.3
    cabinet: bool = False
    release: float = 0.08
    level: float = 0.19

    def release_time(self) -> float:
        return self.release

    def inserts(self) -> tuple[Effect, ...]:
        chain: tuple[Effect, ...] = (Filter(kind="lowpass", cutoff=900.0 + 3000.0 * self.tone, q=0.8),)
        if self.drive > 0:
            chain += (Drive(amount=self.drive, tone=2500.0 + 3000.0 * self.tone),)
        if self.cabinet:
            chain += (
                Filter(kind="highpass", cutoff=38.0, q=0.7),
                EQ(low=2.0, low_freq=100.0, mid=3.0, mid_freq=800.0, mid_q=0.8),
                Filter(kind="lowpass", cutoff=4000.0, q=0.7),
            )
        return chain

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        muted = bool(v.params.get("palm_mute", False))
        legato = bool(v.params.get("legato", False))
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
        if self.pitch_attack and not legato:
            glide = self.pitch_attack / 100.0 * v.vel * np.exp(-np.arange(n) / (_TENSION_TAU * sr))
            curve = glide if curve is None else curve + glide
        y = string(n) if curve is None else variable_rate_read(string, curve)
        if self.clank and not legato:
            k = min(n, int(0.03 * sr))
            burst = filt(rng.uniform(-1.0, 1.0, k), "bandpass", 3000.0, sr, 0.9) * np.exp(-np.arange(k) / (0.005 * sr))
            y[:k] += self.clank * v.vel * burst / max(float(np.max(np.abs(burst))), 1e-9) * float(np.max(np.abs(y)))
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, sr, self.release)
