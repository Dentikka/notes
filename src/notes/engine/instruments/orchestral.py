"""Procedural piano and bowed strings; no recorded samples required."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.articulation import pitch_curve
from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.dsp import adsr, filt, vel_gain
from notes.engine.registry import register_instrument


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


@register_instrument("piano")
@dataclass(frozen=True, kw_only=True)
class Piano(Instrument):
    """Hammered inharmonic strings, unison beating and velocity-dependent felt.

    sustain is fundamental T60 at A3. Per-note pedal=True lets the string ring
    after key release; it is a fixed per-note pedal, not a continuous MIDI pedal.
    brightness controls hammer hardness, detune is unison spread in cents.
    model="soundboard" enables hammer-position coloration and two-stage decay;
    model="hammer" integrates a nonlinear felt contact exciting string modes.
    model="hammer_damped" additionally softens the modal readout above A2; it is
    an experimental register correction, not a complete soundboard model.
    legacy preserves the original renderer.
    """

    program: int | None = 0
    model: str = "legacy"
    sustain: float = 8.0
    brightness: float = 0.65
    detune: float = 1.4
    stiffness: float = 0.00012
    hammer: float = 0.035
    release: float = 0.18
    level: float = 0.45

    def __post_init__(self) -> None:
        if self.model not in ("legacy", "soundboard", "hammer", "hammer_damped"):
            raise ValueError("model must be legacy, soundboard, hammer or hammer_damped")
        _positive(sustain=self.sustain, release=self.release)
        for name in ("brightness", "detune", "stiffness", "hammer", "level"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def release_time(self) -> float:
        return max(self.release, self.sustain)

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        if self.model in ("hammer", "hammer_damped"):
            from notes.engine.hammer import struck_string

            pedal = bool(v.params.get("pedal", False))
            tail = self.release_time() if pedal else self.release
            n = ctx.samples(v.dur + tail)
            y = struck_string(v.freq, n, ctx.sr, v.vel, brightness=self.brightness,
                              stiffness=self.stiffness, sustain=self.sustain, detune=self.detune,
                              radiation=self.model == "hammer_damped")
            if not pedal:
                y *= adsr(ctx.samples(v.dur), n, ctx.sr, 0.001, 0.01, 1, self.release)
            count = min(n, ctx.samples(0.01))
            if count:
                y[-count:] *= np.linspace(1, 0, count)
            return self.level * y
        if self.model == "soundboard":
            return self._soundboard(v, ctx, rng)
        pedal = bool(v.params.get("pedal", False))
        tail = self.release_time() if pedal else self.release
        n = ctx.samples(v.dur + tail)
        t = np.arange(n) / ctx.sr
        y = np.zeros(n)
        tau = self.sustain * (220 / v.freq) ** 0.3 / np.log(1000)
        hardness = max(0.1, self.brightness * (0.25 + 0.75 * v.vel))
        strings = 1 if v.pitch < 33 else (2 if v.pitch < 52 else 3)
        for cents in np.linspace(-self.detune, self.detune, strings) if strings > 1 else [0.0]:
            f0 = v.freq * 2 ** (cents / 1200)
            for k in range(1, min(65, int(ctx.sr * 0.47 / f0) + 1)):
                freq = f0 * k * np.sqrt((1 + self.stiffness * k * k) / (1 + self.stiffness))
                if freq >= ctx.sr * 0.47:
                    break
                amp = np.exp(-(k - 1) / (2 + 13 * hardness)) / k ** 0.85
                decay = np.exp(-t * (1 + 0.16 * (k - 1) ** 1.3) / tau)
                y += amp * decay * np.sin(2 * np.pi * freq * t) / strings
        y *= 1 - np.exp(-t / 0.0015)
        noise = filt(rng.normal(size=n), "lowpass", min(3500, ctx.sr * 0.4), ctx.sr, 0.7)
        y += self.hammer * v.vel * noise * np.exp(-t / 0.009) * (1 - np.exp(-t / 0.0005))
        if not pedal:
            y *= adsr(ctx.samples(v.dur), n, ctx.sr, 0.001, 0.01, 1, self.release)
        if n:
            y[-min(n, ctx.samples(0.01)):] *= np.linspace(1, 0, min(n, ctx.samples(0.01)))
        return self.level * vel_gain(v.vel) * y


    def _soundboard(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        """Designed modal variant: hammer-position comb, two-stage partial decay,
        slightly unequal unison strings and a resonant bridge response.
        """
        pedal = bool(v.params.get("pedal", False))
        tail = self.release_time() if pedal else self.release
        n = ctx.samples(v.dur + tail)
        t = np.arange(n) / ctx.sr
        y = np.zeros(n)
        hardness = max(0.08, self.brightness * (0.2 + 0.8 * v.vel))
        strings = 1 if v.pitch < 33 else (2 if v.pitch < 52 else 3)
        offsets = np.linspace(-self.detune, self.detune, strings) if strings > 1 else [0.0]
        stiffness = self.stiffness * (1 + 0.6 * abs(v.pitch - 60) / 24)
        for index, cents in enumerate(offsets):
            f0 = v.freq * 2 ** (cents / 1200)
            # A few samples between strings, rather than phase-identical oscillators.
            lag = index * 0.00013
            age = np.maximum(t - lag, 0)
            for k in range(1, min(97, int(ctx.sr * 0.46 / f0) + 1)):
                hz = f0 * k * np.sqrt((1 + stiffness * k * k) / (1 + stiffness))
                if hz >= ctx.sr * 0.46:
                    break
                # Fixed hammer position creates uneven partial strengths and missing modes.
                strike = np.sin(np.pi * k * 0.137) / np.sin(np.pi * 0.137)
                felt = np.exp(-(hz / (1600 + 6500 * hardness)) ** 1.4)
                amplitude = strike * felt / k ** 0.72
                amplitude *= 1 + 0.06 * np.sin(k * 2.17 + index)
                # Upper partials persist instead of collapsing to the fundamental.
                t60 = self.sustain * (220 / v.freq) ** 0.28 / (1 + 0.055 * (k - 1) ** 1.15)
                prompt = np.exp(-np.log(1000) * age / (0.24 * t60))
                after = np.exp(-np.log(1000) * age / t60)
                envelope = (0.65 * prompt + 0.35 * after) * (1 - np.exp(-age / 0.0018))
                y += amplitude * envelope * np.sin(2 * np.pi * hz * age) / strings
        # Short fixed bridge resonances colour the entire string, including its attack.
        dry = y.copy()
        for hz, q, gain in ((180, 1.0, 1.4), (520, 1.3, -2.0), (2600, 0.8, 2.5)):
            if hz < ctx.sr * 0.45:
                y = filt(y, "peak", hz, ctx.sr, q, gain)
        y = 0.85 * y + 0.15 * dry
        noise = filt(rng.normal(size=n), "lowpass", min(5500, ctx.sr * 0.4), ctx.sr, 0.7)
        y += self.hammer * noise * np.exp(-t / 0.012) * (1 - np.exp(-t / 0.001))
        if not pedal:
            y *= adsr(ctx.samples(v.dur), n, ctx.sr, 0.001, 0.01, 1, self.release)
        count = min(n, ctx.samples(0.01))
        if count:
            y[-count:] *= np.linspace(1, 0, count)
        return self.level * 0.55 * vel_gain(v.vel) * y


@register_instrument("bowed_string")
@dataclass(frozen=True, kw_only=True)
class BowedString(Instrument):
    """Sustained harmonic bow source shaped by fixed body resonances.

    A procedural approximation, not a nonlinear bow/string simulation.
    pressure controls upper harmonics; bow_noise the friction component.
    Per-note legato=True shortens attack; vibrato/bends/pitch_points are supported.
    """

    program: int | None = 40
    body_size: float = 1.0
    pressure: float = 0.55
    bow_noise: float = 0.025
    attack: float = 0.12
    release: float = 0.18
    vibrato: float = 0.12
    vibrato_rate: float = 5.3
    level: float = 0.32

    def __post_init__(self) -> None:
        _positive(body_size=self.body_size, attack=self.attack, release=self.release,
                  vibrato_rate=self.vibrato_rate)
        for name in ("pressure", "bow_noise", "vibrato", "level"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        n = ctx.samples(v.dur) + ctx.samples(self.release)
        t = np.arange(n) / ctx.sr
        params = {"vibrato": self.vibrato, "vibrato_rate": self.vibrato_rate, **v.params}
        curve = pitch_curve(params, n, ctx.sr, rng=rng)
        freq = np.full(n, v.freq) if curve is None else v.freq * 2 ** (curve / 12)
        phase = np.concatenate(([0.0], np.cumsum(freq[:-1] / ctx.sr)))[:n]
        y = np.zeros(n)
        highest = max(v.freq, float(np.max(freq, initial=v.freq)))
        for k in range(1, min(81, int(ctx.sr * 0.47 / highest) + 1)):
            hz = v.freq * k
            body = 0.35 + sum(g / (1 + ((hz - f / self.body_size) / (w / self.body_size)) ** 2)
                              for f, w, g in ((460, 100, 0.9), (550, 90, 0.7), (2400, 700, 1.8)))
            amp = body * np.exp(-(k - 1) / (8 + 30 * self.pressure * v.vel)) / k
            y += amp * np.sin(2 * np.pi * k * phase)
        noise = filt(rng.normal(size=n), "highpass", min(1000 / self.body_size, ctx.sr * 0.3), ctx.sr, 0.7)
        y += self.bow_noise * noise
        y *= 1 + 0.025 * np.sin(2 * np.pi * 1.4 * t + rng.uniform(0, 2 * np.pi))
        attack = 0.025 if v.params.get("legato", False) else self.attack
        env = adsr(ctx.samples(v.dur), n, ctx.sr, attack, 0.3, 0.88, self.release)
        if n:
            count = min(n, max(2, ctx.samples(0.002)))
            env[-count:] *= np.linspace(1, 0, count)
        return self.level * vel_gain(v.vel) * y * env


@register_instrument("violin")
@dataclass(frozen=True, kw_only=True)
class Violin(BowedString):
    """Solo violin body and bow preset."""


@register_instrument("cello")
@dataclass(frozen=True, kw_only=True)
class Cello(BowedString):
    """Larger body, slower bow attack and slower vibrato."""

    program: int | None = 42
    body_size: float = 2.2
    attack: float = 0.16
    vibrato_rate: float = 4.7
