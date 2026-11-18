"""Experimental shared-hammer piano; the accepted Piano remains separate."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

from notes.engine.articulation import pitch_curve
from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.dsp import adsr
from notes.engine.piano_unison import _number, _Parameters, struck_unisons
from notes.engine.registry import register_instrument


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be boolean')
    return bool(value)


@register_instrument('coupled_piano')
@dataclass(frozen=True, kw_only=True)
class CoupledPiano(Instrument):
    """One felt hammer and separate string coordinates, with an optional moving bridge.

    This is an experimental source, not the accepted acoustic-piano preset or a
    calibrated soundboard. `unison_cents` selects one to three explicit strings;
    identical offsets retain the aggregate one-string limit. Parameters shared
    with the source may be overridden on notes. There are no measured phase,
    carrier, impulse-response or readout-envelope inputs.

    `pitch_compensation` tunes the generated modal centroid, not a recording.
    A nonoscillatory or nonconvergent configuration raises instead of silently
    returning an unverified pitch. Dynamic pitch contours are unsupported.
    `pedal` or `undamped` notes ring for the explicit finite `ring_time` after
    their gate. This is not sympathetic resonance, continuous pedal or a physical
    key-damper model. `release` and `ring_time` also accept note overrides.
    """

    program: int | None = 0
    unison_cents: tuple[float, ...] = (-1.4, 0., 1.4)
    modes: int = 64
    mechanical_band: float = 17640.
    stiffness: float = .00012
    sustain: float = 8.
    brightness: float = .65
    hammer_mass_scale: float = 1.
    felt_stiffness_scale: float = 1.
    contact_oversample: int = 1
    radiation: bool = True
    radiation_loss_scale: float = 1.
    bridge_mass: float | None = None
    bridge_hz: float = 1500.
    bridge_q: float = 30.
    pitch_compensation: bool = True
    release: float = .18
    ring_time: float = 8.
    level: float = .45

    def __post_init__(self) -> None:
        p = self._settings({})
        for field in fields(p):
            object.__setattr__(self, field.name, getattr(p, field.name))
        object.__setattr__(self, 'pitch_compensation', _boolean(self.pitch_compensation, 'pitch_compensation'))
        for name in ['release', 'ring_time']:
            object.__setattr__(self, name, _number(getattr(self, name), name, 0, 120, positive=True))
        object.__setattr__(self, 'level', _number(self.level, 'level', 0, 16))

    def _settings(self, params: Mapping[str, object]) -> _Parameters:
        return _Parameters(**{f.name: params.get(f.name, getattr(self, f.name)) for f in fields(_Parameters)})

    def release_time(self) -> float:
        return max(self.release, self.ring_time)

    def release_time_for(self, params: Mapping[str, object]) -> float:
        return max(_number(params.get(name, getattr(self, name)), name, 0, 120, positive=True)
                   for name in ['release', 'ring_time'])

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        options = self._settings(v.params)
        release = _number(v.params.get('release', self.release), 'release', 0, 120, positive=True)
        ring = _number(v.params.get('ring_time', self.ring_time), 'ring_time', 0, 120, positive=True)
        free = _boolean(v.params.get('pedal', False), 'pedal') | _boolean(v.params.get('undamped', False), 'undamped')
        tuning = _boolean(v.params.get('pitch_compensation', self.pitch_compensation), 'pitch_compensation')
        velocity = _number(v.vel, 'velocity', 0, 1)
        gate = ctx.samples(_number(v.dur, 'gate', 0))
        n = gate+ctx.samples(max(release, ring) if free else release)
        for name in ['partial_gain_points', 'unison_strings', 'hammer_speed',
                     'hammer_speed_points', 'radiation_compensation']:
            if name in v.params:
                raise ValueError(f'{name} is not implemented for CoupledPiano')
        if gate == 0 or velocity == 0 or self.level == 0:
            return np.zeros(n)
        frequency = v.freq
        curve = pitch_curve(v.params, n, ctx.sr)
        if curve is not None:
            if np.ptp(curve) > 1e-12:
                raise ValueError('CoupledPiano supports steady pitches only')
            frequency *= 2**(float(curve[0])/12)
        wave = struck_unisons(frequency, n, ctx.sr, velocity, pitch_compensation=tuning,
                              **{f.name: getattr(options, f.name) for f in fields(options)})
        # The onset observation must not depend on whether the key has a damper
        # or the pedal is down. Key release only affects samples after the gate.
        wave *= adsr(n if free else gate, n, ctx.sr, .001, .01, 1., release)
        count = min(n-gate, ctx.samples(.01))
        if count:
            wave[-count:] *= np.linspace(1., 0., count)
        return self.level*wave
