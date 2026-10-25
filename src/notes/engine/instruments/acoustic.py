"""Acoustic guitar: steel strings heard through a wooden body, six strings or twelve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.articulation import pitch_curve, variable_rate_read
from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.dsp import delay, vel_gain
from notes.engine.effects.body import Body
from notes.engine.effects.tone import Filter
from notes.engine.instruments.strings import (
    _LEGATO_SOFTNESS,
    _STANDARD_TUNING,
    _along,
    _gate,
    _pick_click,
    _t60,
    fret_scale,
)
from notes.engine.registry import register_instrument
from notes.engine.string_model import steel_string

__all__ = ["AcousticGuitar", "NylonGuitar"]

#: Strings 3-6 (G, D, A, low E) of a twelve-string carry an octave string; 1-2 a unison one.
_OCTAVE_COURSES = (3, 4, 5, 6)
_LOWEST_UNISON = 59  # B3: a note without a string below it is on an octave course


@register_instrument("acoustic_guitar")
@dataclass(frozen=True, kw_only=True)
class AcousticGuitar(Instrument):
    """Steel strings (`notes.engine.string_model`) through the guitar's body (`Body`), miked.

    `prompt_decay` optionally shortens the picked polarization relative to sustain;
    `velocity_brightness` couples picking strength to hardness (0 preserves old behaviour).
    `fretting=True` maps a fixed pick position onto the shortened fretted string;
    per-note string=1..6 selects the open string, otherwise the lowest fret is used.
    `sustain` is the T60 (s) of the fundamental at A3, `treble_decay` that of partials near
    4 kHz, `brightness` the pick's hardness, `pick_position` where it strikes (fraction of the
    string from the bridge; over the sound hole ~0.25); `body` the share of the sound that is
    the body, `body_size` its scale (1: a dreadnought), `diffuse` its dense upper modes. With
    `twelve`, each course has a second string: an octave up on the four lower courses (strings
    3-6 when a note says ``string=``, else below B3), in unison on the two upper ones,
    `course_detune` cents apart and struck `course_lag` s after the thin string (a down stroke
    meets it first), at `octave_level` of the main string. Per-note ``legato=True`` (no pick,
    softer start) and ``vibrato``, ``bend``, ``slide`` as on the electric guitar.
    """

    program: int | None = 25
    sustain: float = 7.0
    treble_decay: float = 3.0
    brightness: float = 0.9
    pick_position: float = 0.18
    fretting: bool = False
    excitation: str = "velocity"
    stiffness: float = 4e-5
    polarization: float = 0.3
    prompt_decay: float = 0.0
    velocity_brightness: float = 0.0
    body: float = 1.0
    body_size: float = 1.0
    diffuse: float = 0.8
    twelve: bool = False
    course_detune: float = 4.0
    course_lag: float = 0.006
    octave_level: float = 0.6
    pick_noise: float = 0.15
    release: float = 0.08
    level: float = 0.25

    def __post_init__(self) -> None:
        if self.excitation not in ("velocity", "displacement"):
            raise ValueError("excitation must be velocity or displacement")

    def release_time(self) -> float:
        return self.release

    def inserts(self) -> tuple[Effect, ...]:
        return (Filter(kind="highpass", cutoff=60.0), Body(size=self.body_size, diffuse=self.diffuse, mix=self.body))

    def _string(self, freq: float, n: int, sr: int, hardness: float, curve, rng: np.random.Generator,
                pick: float | None = None) -> np.ndarray:
        def render(m: int) -> np.ndarray:
            return steel_string(freq, m, sr, rng, t60=_t60(self.sustain, freq), t60_high=self.treble_decay,
                                pick=self.pick_position if pick is None else pick, hardness=hardness,
                                stiffness=self.stiffness,
                                second=self.polarization, excitation=self.excitation,
                                prompt_t60=(self.prompt_decay * _t60(self.sustain, freq)
                                            if self.prompt_decay > 0 else None))

        return render(n) if curve is None else variable_rate_read(render, curve)

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        legato = bool(v.params.get("legato", False))
        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        hardness = self.brightness * (_LEGATO_SOFTNESS if legato else 1.0)
        hardness *= 1.0 - self.velocity_brightness * (1.0 - v.vel)
        curve = pitch_curve(v.params, n, sr, vibrato_shape="push", rng=rng)
        pick = None
        if self.fretting:
            string = v.params.get("string")
            if string is not None:
                if not isinstance(string, int) or not 1 <= string <= 6:
                    raise ValueError("string must be an integer from 1 to 6")
                opened = _STANDARD_TUNING[6-string]
                scale = 2 ** (-float(np.clip(v.pitch-opened, 0, 24))/12)
            else:
                scale = fret_scale(v.pitch)
            pick = _along(self.pick_position, scale)
        y = self._string(v.freq, n, sr, hardness, curve, rng, pick)
        if self.twelve:
            string = v.params.get("string")
            octave = string in _OCTAVE_COURSES if string is not None else v.pitch < _LOWEST_UNISON
            partner = self._string(v.freq * (2.0 if octave else 1.0) * 2.0 ** (self.course_detune / 1200.0), n, sr,
                                   hardness, curve, rng, pick)
            lag = int(round(self.course_lag * sr))
            # the octave string is the thin one, met first on a down stroke; a unison pair is alike
            y = delay(y, lag) + self.octave_level * partner if octave else y + self.octave_level * delay(partner, lag)
        if not legato:
            y = y + self.pick_noise * _pick_click(n, sr, rng) * float(np.max(np.abs(y)))
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, sr, self.release)


@register_instrument("nylon_guitar")
@dataclass(frozen=True, kw_only=True)
class NylonGuitar(AcousticGuitar):
    """Experimental classical-guitar variant with displacement-like string readout.

    Uses the same mathematical waveguide and body, with softer excitation and
    lower stiffness. Fretting follows a fixed pick location along each string.
    """

    program: int | None = 24
    excitation: str = "displacement"
    fretting: bool = True
    stiffness: float = 1e-5
    brightness: float = 0.5
    treble_decay: float = 1.5
    pick_noise: float = 0.04
