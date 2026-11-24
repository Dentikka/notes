"""Experimental additive tones and independently generated coloured noise."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from notes.engine.articulation import pitch_curve
from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.registry import register_instrument
from notes.engine.spectral import SpectralProfile


@register_instrument('spectral_tone')
@dataclass(frozen=True, kw_only=True)
class SpectralTone(Instrument):
    """A numeric profile driven by the usual note pitch, gate, velocity and bends.

    No audio is loaded by this instrument. Phases are designed anew, and each
    noise band has its own deterministic RNG stream. Frequencies above 45% of
    sample rate are omitted (for bends, the whole component is checked against
    the maximum pitch). Noise bands remain at their absolute Hz positions.

    Profile envelopes run on note time; beyond the last knot they fade with
    this instrument's release time rather than sustain an unmeasured level.
    Velocity scales amplitude only. Neither transposition nor velocity creates
    an inferred timbral change; use independently validated profiles for that.
    The default is a designed sine pluck, not a realistic instrument preset.
    """

    profile: SpectralProfile | Mapping[str, Any] = field(default_factory=SpectralProfile)
    release: float = .12
    level: float = 1.
    velocity_reference: float = .7
    velocity_power: float = 1.7

    def __post_init__(self) -> None:
        profile = self.profile
        if isinstance(profile, Mapping):
            profile = SpectralProfile.from_dict(profile)
        if not isinstance(profile, SpectralProfile):
            raise ValueError('SpectralTone requires a SpectralProfile or its numeric dictionary')
        object.__setattr__(self, 'profile', profile)
        if (not np.isfinite([self.release, self.level, self.velocity_reference, self.velocity_power]).all()
                or self.release <= 0 or self.level < 0 or not 0 < self.velocity_reference <= 1
                or self.velocity_power <= 0):
            raise ValueError('invalid spectral release, level or velocity controls')

    def to_spec(self) -> dict[str, Any]:
        result = super().to_spec()
        result['profile'] = self.profile.to_dict()
        return result

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        if (not np.isfinite([v.freq, v.dur, v.vel]).all() or v.freq <= 0 or v.dur < 0 or not 0 <= v.vel <= 1
                or ctx.sr < 1000):
            raise ValueError('spectral voice requires finite positive frequency, valid duration/velocity/rate')
        n = ctx.samples(v.dur) + ctx.samples(self.release)
        t = np.arange(n) / ctx.sr
        curve = pitch_curve(v.params, n, ctx.sr)
        if curve is None:
            phase = 2*np.pi*v.freq*t
            maximum_hz = v.freq
        else:
            hz = v.freq*2**(curve/12)
            phase = 2*np.pi*np.r_[0., np.cumsum(hz[:-1])/ctx.sr]
            maximum_hz = float(np.max(hz))
        ratios = np.array(self.profile.ratios)
        phases = (-np.pi*ratios*.08 + np.where(np.sin(np.pi*ratios*.15) < 0, np.pi, 0.)
                  + rng.uniform(-.05, .05, len(ratios)))
        seeds = rng.integers(0, 2**64, size=len(self.profile.noise_bands), dtype=np.uint64)

        def envelope(values: tuple[float, ...]) -> np.ndarray:
            result = np.interp(t, self.profile.times, values)
            result *= np.exp(-np.maximum(t-self.profile.times[-1], 0)/self.release)
            return result

        y = np.zeros(n)
        for ratio, initial, values in zip(ratios, phases, self.profile.amplitudes, strict=True):
            if ratio*maximum_hz < .45*ctx.sr:
                y += envelope(values)*np.sin(ratio*phase+initial)
        for band, values, seed in zip(self.profile.noise_bands, self.profile.noise_rms, seeds, strict=True):
            low, high = band[0], min(band[1], .45*ctx.sr)
            if low >= high:
                continue
            sos = butter(2, [low, high], fs=ctx.sr, btype='bandpass', output='sos')
            impulse = np.zeros(ctx.sr)
            impulse[0] = 1.
            scale = np.sqrt(np.sum(sosfilt(sos, impulse)**2))
            white = np.random.default_rng(int(seed)).normal(size=n)
            y += sosfilt(sos, white)/max(scale, 1e-30)*envelope(values)
        gate = np.minimum(t/.002, 1.)*np.clip((v.dur+self.release-t)/self.release, 0, 1)
        return y*gate*self.level*(v.vel/self.velocity_reference)**self.velocity_power
