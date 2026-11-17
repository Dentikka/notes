"""Experimental mechanical plucked string with an electromagnetic pickup readout."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.articulation import pitch_curve
from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.dsp import vel_gain
from notes.engine.effects.tone import Filter
from notes.engine.instruments.strings import _LEGATO_LEVEL, _LEGATO_SOFTNESS, _gate, _pick_click, fret_scale
from notes.engine.mechanical_string import modal_string
from notes.engine.registry import register_instrument


@register_instrument('modal_string')
@dataclass(frozen=True, kw_only=True)
class ModalString(Instrument):
    """Steady plucked notes from positive mass/stiffness/damping matrices.

    One designed bridge mode exchanges energy with the string. The pickup reads
    velocity relative to the moving bridge. ``release_motion`` (-1..1) assigns
    initial energy to a moving rather than held deflection; its sign sets the
    direction, while velocity scales energy independently.

    Per-note ``pick_position``, ``pick_hardness``, ``release_motion``,
    ``bridge_scale`` (0..1) and effective ``stiffness`` override the controls.
    ``fretting=True`` uses the guitar's string/fret geometry. ``level`` converts
    mechanical velocity to DSP amplitude, so it affects a following amplifier.
    ``pluck_force`` is the effective maximum pull in the mechanical model,
    scaled by velocity; it is separate from the synthetic pick transient.

    Optional ``palm_resistance`` enables distributed viscous contact on muted
    notes instead of ``mute_decay``. None preserves scalar muting; zero enables
    the contact mechanics with no added damping. ``palm_position/palm_width``
    describe its region as fractions of the open string (or vibrating length
    without fretting), with resistance in effective N*s/m. These three controls
    accept per-note overrides requiring ``palm_mute=True``. The independent
    ``mechanical_band`` is bounded by ``modes``; it is not an identified palm.
    ``pick_cutoff`` optionally overrides the hardness-derived force bandwidth;
    ``pluck_rounding`` selects spatial or release rounding. Defaults retain
    the original source. Neither this source nor a fit certifies timbre realism.

    This experimental direct source has no amp/cabinet. It currently supports
    steady pitches; time-varying bends, slides and vibrato raise ValueError.
    No sampled waves, reference phases or measured IRs are loaded.
    """

    program: int | None = 27
    modes: int = 64
    stiffness: float = 4e-5
    bridge_hz: float = 125.
    bridge_q: float = 20.
    bridge_mobility: float = .01
    tension: float = 70.
    length: float = .65
    sustain: float = 9.
    treble_decay: float = 2.5
    pick_position: float = .15
    brightness: float = .7
    pick_cutoff: float | None = None
    pluck_rounding: str = 'release'
    pickup_position: float = .06
    pickup_width: float = .04
    coil_spacing: float = 0.
    pickup_cutoff: float = 4000.
    pickup_q: float = .7
    release_motion: float = 0.
    fretting: bool = False
    pick_noise: float = .2
    pluck_force: float = 3.
    mute_decay: float = .5
    palm_resistance: float | None = None
    palm_position: float = .08
    palm_width: float = .04
    mechanical_band: float = 8000.
    release: float = .06
    level: float = .5

    def __post_init__(self) -> None:
        if not isinstance(self.modes, int) or isinstance(self.modes, bool) or not 8 <= self.modes <= 128:
            raise ValueError('modes must be an integer from 8 to 128')
        for name in ('bridge_hz', 'bridge_q', 'tension', 'length', 'sustain', 'treble_decay',
                     'pickup_cutoff', 'pickup_q', 'mute_decay', 'release'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('stiffness', 'bridge_mobility', 'pickup_width', 'coil_spacing',
                     'pick_noise', 'pluck_force', 'level'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not np.isfinite(self.brightness) or not 0 <= self.brightness <= 1:
            raise ValueError('brightness must be between 0 and 1')
        if not np.isfinite(self.release_motion) or not -1 <= self.release_motion <= 1:
            raise ValueError('release_motion must be between -1 and 1')
        if not np.isfinite(self.pick_position) or not 0 < self.pick_position < 1:
            raise ValueError('pick_position must be inside the string')
        if (not np.isfinite(self.pickup_position) or not 0 < self.pickup_position < 1
                or self.pickup_position-(self.pickup_width+self.coil_spacing)/2 < 0
                or self.pickup_position+(self.pickup_width+self.coil_spacing)/2 > 1):
            raise ValueError('the whole pickup aperture must lie inside the string')
        if self.pick_cutoff is not None and (not np.isfinite(self.pick_cutoff) or self.pick_cutoff <= 0):
            raise ValueError('pick_cutoff must be finite and positive')
        if self.pluck_rounding not in ('spatial', 'release'):
            raise ValueError('pluck_rounding must be spatial or release')
        if self.palm_resistance is not None and (not np.isfinite(self.palm_resistance) or self.palm_resistance < 0):
            raise ValueError('palm_resistance must be finite and nonnegative')
        if (not np.isfinite(self.palm_position) or not 0 < self.palm_position < 1
                or not np.isfinite(self.palm_width) or self.palm_width < 0
                or self.palm_position-self.palm_width/2 < 0 or self.palm_position+self.palm_width/2 > 1):
            raise ValueError('whole palm contact must lie inside the open string')
        if not np.isfinite(self.mechanical_band) or self.mechanical_band <= 0:
            raise ValueError('mechanical_band must be finite and positive')

    def release_time(self) -> float:
        return self.release

    def inserts(self) -> tuple[Effect, ...]:
        return (Filter(kind='lowpass', cutoff=self.pickup_cutoff, q=self.pickup_q),)

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        gate = ctx.samples(v.dur)
        n = gate+ctx.samples(self.release)
        if gate == 0 or v.vel <= 0 or self.level == 0 or self.pluck_force == 0:
            return np.zeros(n)
        curve = pitch_curve(v.params, n, ctx.sr)
        frequency = v.freq
        if curve is not None:
            if np.ptp(curve) > 1e-12:
                raise ValueError('ModalString currently supports steady pitches; use ElectricGuitar for pitch curves')
            frequency *= 2**(float(curve[0])/12)
        scale = fret_scale(v.pitch, v.params.get('string')) if self.fretting else 1.
        pick = float(v.params.get('pick_position', self.pick_position))/scale
        hardness = float(v.params.get('pick_hardness', self.brightness))
        motion = float(v.params.get('release_motion', self.release_motion))
        bridge = float(v.params.get('bridge_scale', 1.))
        stiffness = float(v.params.get('stiffness', self.stiffness/(scale*scale)))
        if not np.isfinite(hardness) or not 0 <= hardness <= 1:
            raise ValueError('pick_hardness must be between 0 and 1')
        if not np.isfinite(bridge) or not 0 <= bridge <= 1:
            raise ValueError('bridge_scale must be between 0 and 1')
        legato = bool(v.params.get('legato', False))
        muted = bool(v.params.get('palm_mute', False))
        if legato:
            hardness *= _LEGATO_SOFTNESS
        if not muted and any(name in v.params for name in ('palm_resistance', 'palm_position', 'palm_width')):
            raise ValueError('per-note palm controls require palm_mute=True')
        resistance = v.params.get('palm_resistance', self.palm_resistance) if muted else None
        if resistance is not None:
            resistance = float(resistance)
        contact = resistance is not None
        if not contact and any(name in v.params for name in ('palm_position', 'palm_width')):
            raise ValueError('per-note palm geometry requires palm_resistance')
        position = float(v.params.get('palm_position', self.palm_position))/scale
        contact_width = float(v.params.get('palm_width', self.palm_width))/scale
        cutoff = v.params.get('pick_cutoff', self.pick_cutoff)
        if cutoff is not None:
            cutoff = float(cutoff)
        else:
            cutoff = min(1500*2**(3*hardness), .8*self.mechanical_band if contact else ctx.sr*.45)
        sustain = self.mute_decay if muted and not contact else self.sustain
        upper = min(self.treble_decay, .3*self.mute_decay) if muted and not contact else self.treble_decay
        wave = modal_string(frequency, n, ctx.sr, modes=self.modes, stiffness=stiffness,
            body_hz=self.bridge_hz, body_q=self.bridge_q, peak=self.bridge_mobility*bridge,
            tension=self.tension, length=self.length*scale, sustain=sustain, treble_decay=upper,
            pick=pick, pickup=self.pickup_position/scale, width=self.pickup_width/scale,
            spacing=self.coil_spacing/scale, cutoff=cutoff, rounding=self.pluck_rounding,
            release_motion=motion, palm_resistance=resistance, palm_position=position,
            palm_width=contact_width, mechanical_band=self.mechanical_band)
        wave *= self.pluck_force
        if legato:
            wave *= _LEGATO_LEVEL
        else:
            wave += self.pick_noise*_pick_click(n, ctx.sr, rng)
        envelope = _gate(gate, n, ctx.sr, self.release)
        if not contact:
            envelope[:gate] = 1.  # Preserve the original source's mechanically defined attack.
        return wave*self.level*vel_gain(v.vel)*envelope
