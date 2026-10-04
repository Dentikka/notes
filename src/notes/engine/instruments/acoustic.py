"""Acoustic guitar: steel strings heard through a wooden body, six strings or twelve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.articulation import pitch_curve, variable_rate_read
from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.dsp import delay, vel_gain
from notes.engine.effects.body import Body
from notes.engine.effects.tone import Filter
from notes.engine.instruments.strings import _LEGATO_SOFTNESS, _gate, _pick_click, _t60
from notes.engine.registry import register_instrument
from notes.engine.string_model import steel_string

__all__ = ["AcousticGuitar"]

#: Strings 3-6 (G, D, A, low E) of a twelve-string carry an octave string; 1-2 a unison one.
_OCTAVE_COURSES = (3, 4, 5, 6)
_LOWEST_UNISON = 59  # B3: a note without a string below it is on an octave course


@register_instrument("acoustic_guitar")
@dataclass(frozen=True, kw_only=True)
class AcousticGuitar(Instrument):
    """Steel strings (`notes.engine.string_model`) through the guitar's body (`Body`), miked.

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
    stiffness: float = 4e-5
    polarization: float = 0.3
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

    def release_time(self) -> float:
        return self.release

    def inserts(self) -> tuple[Effect, ...]:
        return (Filter(kind="highpass", cutoff=60.0), Body(size=self.body_size, diffuse=self.diffuse, mix=self.body))

    def _string(self, freq: float, n: int, sr: int, hardness: float, curve, rng: np.random.Generator) -> np.ndarray:
        def render(m: int) -> np.ndarray:
            return steel_string(freq, m, sr, rng, t60=_t60(self.sustain, freq), t60_high=self.treble_decay,
                                pick=self.pick_position, hardness=hardness, stiffness=self.stiffness,
                                second=self.polarization)

        return render(n) if curve is None else variable_rate_read(render, curve)

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        legato = bool(v.params.get("legato", False))
        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        hardness = self.brightness * (_LEGATO_SOFTNESS if legato else 1.0)
        curve = pitch_curve(v.params, n, sr, vibrato_shape="push", rng=rng)
        y = self._string(v.freq, n, sr, hardness, curve, rng)
        if self.twelve:
            string = v.params.get("string")
            octave = string in _OCTAVE_COURSES if string is not None else v.pitch < _LOWEST_UNISON
            partner = self._string(v.freq * (2.0 if octave else 1.0) * 2.0 ** (self.course_detune / 1200.0), n, sr,
                                   hardness, curve, rng)
            lag = int(round(self.course_lag * sr))
            # the octave string is the thin one, met first on a down stroke; a unison pair is alike
            y = delay(y, lag) + self.octave_level * partner if octave else y + self.octave_level * delay(partner, lag)
        if not legato:
            y = y + self.pick_noise * _pick_click(n, sr, rng) * float(np.max(np.abs(y)))
        return self.level * vel_gain(v.vel) * y * _gate(gate, n, sr, self.release)
