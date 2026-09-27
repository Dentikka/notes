"""Subtractive synthesis: detuned oscillators through a resonant lowpass with its own envelope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.articulation import pitch_curve
from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.dsp import adsr, filt, lowpass_sweep, osc, vel_gain
from notes.engine.registry import register_instrument

__all__ = ["Bass", "Lead", "Pad", "Synth"]


@register_instrument("synth")
@dataclass(frozen=True, kw_only=True)
class Synth(Instrument):
    """Subtractive voice: `unison` detuned oscillators (+ sub octave, + noise) into a lowpass.

    The 24 dB/oct filter opens by `env_amount` octaves at note-on (scaled by velocity) and
    falls back to `cutoff` with time constant `filter_decay`; `keytrack` = 1 makes the cutoff
    follow the pitch. `detune` is the spread in cents between the outermost unison voices.
    """

    program: int | None = 81
    wave: str = "saw"
    unison: int = 1
    detune: float = 0.0
    sub: float = 0.0
    noise: float = 0.0
    cutoff: float = 12000.0
    resonance: float = 0.7071
    env_amount: float = 0.0
    filter_decay: float = 0.3
    keytrack: float = 0.0
    attack: float = 0.005
    decay: float = 0.3
    sustain: float = 0.8
    release: float = 0.2
    drive: float = 0.0
    level: float = 0.5

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        if n == 0:
            return np.zeros(0)
        curve = pitch_curve(v.params, n, sr)
        freq = v.freq if curve is None else v.freq * 2.0 ** (curve / 12.0)
        spread = np.linspace(-0.5, 0.5, self.unison) * self.detune if self.unison > 1 else np.zeros(1)
        x = np.zeros(n)
        for cents in spread:
            start = float(rng.random()) if self.unison > 1 else 0.0
            x += osc(self.wave, freq * 2.0 ** (cents / 1200.0), n, sr, start)
        x /= np.sqrt(len(spread))
        if self.sub:
            x += self.sub * osc("square", freq / 2.0, n, sr)
        if self.noise:
            x += self.noise * rng.uniform(-1.0, 1.0, n)
        base = self.cutoff * (v.freq / 261.63) ** self.keytrack
        if self.env_amount:
            t = np.arange(n) / sr
            cutoff = base * 2.0 ** (self.env_amount * v.vel * np.exp(-t / max(self.filter_decay, 1e-3)))
            x = lowpass_sweep(x, np.minimum(cutoff, 0.45 * sr), sr, self.resonance)
        elif base < 0.45 * sr:
            x = filt(filt(x, "lowpass", base, sr, 0.5412), "lowpass", base, sr, self.resonance)
        if self.drive:
            k = 1.0 + 9.0 * self.drive
            x = np.tanh(k * x) / np.tanh(k)
        env = adsr(gate, n, sr, self.attack, self.decay, self.sustain, self.release)
        return self.level * vel_gain(v.vel) * env * x


@register_instrument("bass")
@dataclass(frozen=True, kw_only=True)
class Bass(Synth):
    """Round synth bass: saw plus a sub octave, a plucky filter envelope, a little drive."""

    program: int | None = 38
    sub: float = 0.5
    cutoff: float = 220.0
    resonance: float = 1.0
    env_amount: float = 2.5
    filter_decay: float = 0.15
    attack: float = 0.003
    decay: float = 0.4
    sustain: float = 0.7
    release: float = 0.06
    drive: float = 0.25
    level: float = 0.46


@register_instrument("pad")
@dataclass(frozen=True, kw_only=True)
class Pad(Synth):
    """Wide supersaw pad: five detuned saws, soft attack and a long release."""

    program: int | None = 89
    unison: int = 5
    detune: float = 24.0
    cutoff: float = 1600.0
    resonance: float = 0.8
    attack: float = 0.5
    decay: float = 1.0
    sustain: float = 0.85
    release: float = 1.2
    level: float = 0.4


@register_instrument("lead")
@dataclass(frozen=True, kw_only=True)
class Lead(Synth):
    """Hollow square lead with a short filter blip on every note."""

    wave: str = "square"
    unison: int = 2
    detune: float = 8.0
    cutoff: float = 2600.0
    resonance: float = 1.2
    env_amount: float = 1.5
    filter_decay: float = 0.25
    attack: float = 0.01
    sustain: float = 0.7
    release: float = 0.15
    level: float = 0.3
