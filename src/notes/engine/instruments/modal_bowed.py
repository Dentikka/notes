"""Experimental continuous-friction bowed source; legacy strings remain separate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import oaconvolve

from notes.engine.articulation import pitch_curve
from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.bow_contact import modal_bowed_string
from notes.engine.dsp import vel_gain
from notes.engine.effects.speaker import _minimum_phase
from notes.engine.registry import register_instrument


def _body(wave: np.ndarray, sr: int, size: float) -> np.ndarray:
    frequency = np.fft.rfftfreq(8192, 1/sr)
    magnitude = .35+sum(g/(1+((frequency-f/size)/(w/size))**2)
                        for f, w, g in ((460, 100, .9), (550, 90, .7), (2400, 700, 1.8)))
    return oaconvolve(wave, _minimum_phase(magnitude)[:2048])[:len(wave)]


@register_instrument('modal_bowed_string')
@dataclass(frozen=True, kw_only=True)
class ModalBowedString(Instrument):
    """Experimental string/bow contact with a designed feed-forward body.

    Defaults describe cello geometry; violin geometry may be supplied explicitly.
    Selects the highest open string not above the note. Per-note ``open_pitch``
    can select another string; ``bow_speed``, ``force_ratio``, ``bow_position``,
    ``bow_width`` and ``stiffness`` override physical controls. Ordinary note
    velocity scales output amplitude; it does not infer bow motion or pressure.

    Only steady pitches, including constant pitch offsets, are supported. The
    bare mode frequency is not the exact sounding pitch (about 10 cents flat in
    some tested notes). No thermal rosin, body feedback, linked legato state or
    recorded excitation is present. This does not replace Violin/Cello and is
    not an established realistic instrument preset.
    """

    program: int | None = 42
    open_pitches: tuple[float, ...] = (36., 43., 50., 57.)
    length: float = .69
    tension: float = 110.
    stiffness: float = 1e-5
    modes: int = 128
    bandwidth: float = 18000.
    oversample: int = 16
    sustain: float = 12.
    treble_decay: float = 1.
    bow_speed: float = .08
    force_ratio: float = 16.
    bow_position: float = .08
    bow_width: float = .012
    soft_a: float = 100.
    attack: float = .15
    pressure_attack: float | None = None
    body_size: float = 2.2
    release: float = .18
    level: float = .32

    def __post_init__(self) -> None:
        pitches = tuple(float(v) for v in self.open_pitches)
        if (not pitches or not np.isfinite(pitches).all() or min(pitches) < 0 or max(pitches) > 127
                or any(b <= a for a, b in zip(pitches[:-1], pitches[1:], strict=True))):
            raise ValueError('open_pitches must be increasing finite MIDI pitches')
        object.__setattr__(self, 'open_pitches', pitches)
        for name in ('length', 'tension', 'bandwidth', 'sustain', 'treble_decay', 'soft_a',
                     'attack', 'body_size', 'release'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('stiffness', 'bow_speed', 'force_ratio', 'bow_width', 'level'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if (not np.isfinite(self.bow_position) or self.bow_position-self.bow_width/2 <= 0
                or self.bow_position+self.bow_width/2 >= 1):
            raise ValueError('whole bow aperture must lie inside the open string')
        if not isinstance(self.modes, int) or isinstance(self.modes, bool) or not 8 <= self.modes <= 128:
            raise ValueError('modes must be an integer from 8 to 128')
        if (not isinstance(self.oversample, int) or isinstance(self.oversample, bool)
                or self.oversample not in (1, 2, 4, 8, 16, 32, 64)):
            raise ValueError('oversample must be a power of two from 1 to 64')
        if self.pressure_attack is not None and (not np.isfinite(self.pressure_attack) or self.pressure_attack <= 0):
            raise ValueError('pressure_attack must be finite and positive or None')

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        gate = ctx.samples(v.dur)
        n = gate+ctx.samples(self.release)
        if gate == 0 or v.vel <= 0 or self.level == 0:
            return np.zeros(n)
        curve = pitch_curve(v.params, n, ctx.sr)
        freq = v.freq
        if curve is not None:
            if np.ptp(curve) > 1e-12:
                raise ValueError('ModalBowedString supports steady pitches; use Violin/Cello for pitch curves')
            freq *= 2**(float(curve[0])/12)
        allowed = [p for p in self.open_pitches if p <= v.pitch]
        open_pitch = float(v.params.get('open_pitch', max(allowed) if allowed else self.open_pitches[0]))
        if not np.isfinite(open_pitch) or open_pitch > v.pitch:
            raise ValueError('open_pitch must be finite and not above the note')
        scale = 2**(-(v.pitch-open_pitch)/12)
        wave, _ = modal_bowed_string(freq, n, ctx.sr, gate=gate/ctx.sr,
            bow_speed=float(v.params.get('bow_speed', self.bow_speed)),
            force_ratio=float(v.params.get('force_ratio', self.force_ratio)),
            position=float(v.params.get('bow_position', self.bow_position))/scale,
            width=float(v.params.get('bow_width', self.bow_width))/scale,
            stiffness=float(v.params.get('stiffness', self.stiffness/scale**2)),
            soft_a=self.soft_a, attack=self.attack, pressure_attack=self.pressure_attack,
            modes=self.modes, bandwidth=self.bandwidth, oversample=self.oversample,
            tension=self.tension, length=self.length*scale, sustain=self.sustain, treble=self.treble_decay)
        wave = _body(wave, ctx.sr, self.body_size)
        count = min(n, max(2, ctx.samples(.002)))
        wave[-count:] *= np.linspace(1., 0., count)
        return wave*self.level*vel_gain(v.vel)
