"""An analog-style drum machine (TR-808/909 lineage), every sound synthesised from scratch.

Kicks and toms are sines with falling pitch, snares mix a tone with filtered noise, and
cymbals come from the 808's bank of six detuned square oscillators. Each sound is scaled
to a fixed peak so a default kit is already roughly balanced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.dsp import TAU, fade, filt, vel_gain
from notes.engine.registry import register_instrument
from notes.ir.gm import GM_DRUM_NAMES

__all__ = ["DrumKit"]

#: The TR-808 cymbal oscillator bank (Hz); the inharmonic cluster sounds metallic.
_METAL_FREQS = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def _time(seconds: float, sr: int) -> np.ndarray:
    return np.arange(max(1, int(seconds * sr))) / sr


def _metal(t: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    x = sum(np.sign(np.sin(TAU * f * scale * t + rng.uniform(0.0, TAU))) for f in _METAL_FREQS)
    return x / len(_METAL_FREQS)


def _kick(sr: int, rng: np.random.Generator, tune: float, dec: float, punch: float) -> np.ndarray:
    t = _time(0.5 * dec, sr)
    freq = 46.0 * tune + (150.0 - 46.0) * tune * np.exp(-t / 0.03)
    body = np.sin(TAU * np.cumsum(freq) / sr) * np.exp(-t / (0.22 * dec))
    click = filt(rng.uniform(-1.0, 1.0, t.size), "highpass", 2000.0, sr) * np.exp(-t / 0.002)
    drive = 1.0 + 2.0 * punch
    return np.tanh(drive * (body + 0.4 * punch * click)) / np.tanh(drive)


def _snare(sr: int, rng: np.random.Generator, tune: float, dec: float) -> np.ndarray:
    t = _time(0.35 * dec, sr)
    f = 185.0 * tune
    tone = (np.sin(TAU * f * t) + 0.5 * np.sin(TAU * 1.6 * f * t)) * np.exp(-t / 0.05)
    noise = filt(filt(rng.uniform(-1.0, 1.0, t.size), "highpass", 1200.0, sr), "lowpass", 9000.0, sr)
    return 0.55 * tone + 0.9 * noise * np.exp(-t / (0.11 * dec))


def _clap(sr: int, rng: np.random.Generator, dec: float) -> np.ndarray:
    t = _time(0.4 * dec, sr)
    env = sum(np.where(t >= o, np.exp(-(t - o) / 0.004), 0.0) for o in (0.0, 0.011, 0.022))
    env = env + np.where(t >= 0.03, 0.8 * np.exp(-(t - 0.03) / (0.13 * dec)), 0.0)
    return filt(rng.uniform(-1.0, 1.0, t.size), "bandpass", 1300.0, sr, 1.5) * env


def _hat(sr: int, rng: np.random.Generator, tune: float, tau: float) -> np.ndarray:
    t = _time(min(6.0 * tau, 2.5), sr)
    x = 0.6 * _metal(t, tune, rng) + 0.4 * rng.uniform(-1.0, 1.0, t.size)
    x = filt(filt(x, "bandpass", 9000.0, sr, 0.8), "highpass", 7000.0, sr)
    return x * np.exp(-t / tau)


def _cymbal(sr: int, rng: np.random.Generator, tune: float, tau: float, cutoff: float) -> np.ndarray:
    t = _time(min(5.0 * tau, 5.0), sr)
    x = 0.5 * _metal(t, 1.47 * tune, rng) + 0.5 * rng.uniform(-1.0, 1.0, t.size)
    return fade(filt(x, "highpass", cutoff, sr) * np.exp(-t / tau), sr, fade_in=0.001)


def _tom(sr: int, rng: np.random.Generator, freq: float, dec: float) -> np.ndarray:
    t = _time(0.6 * dec, sr)
    f = freq * (1.0 + 0.5 * np.exp(-t / 0.05))
    body = np.sin(TAU * np.cumsum(f) / sr) * np.exp(-t / (0.25 * dec))
    return body + 0.1 * filt(rng.uniform(-1.0, 1.0, t.size), "bandpass", 3000.0, sr) * np.exp(-t / 0.01)


def _rim(sr: int, rng: np.random.Generator) -> np.ndarray:
    t = _time(0.08, sr)
    click = filt(rng.uniform(-1.0, 1.0, t.size), "bandpass", 1700.0, sr, 3.0) * np.exp(-t / 0.006)
    return click + 0.4 * np.sin(TAU * 500.0 * t) * np.exp(-t / 0.01)


def _cowbell(sr: int, rng: np.random.Generator, tune: float, dec: float) -> np.ndarray:
    t = _time(0.5 * dec, sr)
    x = np.sign(np.sin(TAU * 540.0 * tune * t)) + np.sign(np.sin(TAU * 800.0 * tune * t))
    return filt(x, "bandpass", 1000.0, sr, 1.2) * np.exp(-t / (0.12 * dec))


@register_instrument("drums")
@dataclass(frozen=True, kw_only=True)
class DrumKit(Instrument):
    """Synthesised kit addressed by General MIDI drum numbers or ``sound`` names.

    `tune` shifts every tonal drum in semitones, `decay` scales all decay times, `punch`
    adds click and saturation to the kick. Closed hats choke the open hat.
    """

    program: int | None = 0
    tune: float = 0.0
    decay: float = 1.0
    punch: float = 0.5
    level: float = 0.9

    def release_time(self) -> float:
        return 0.0

    def choke_groups(self) -> Mapping[int, str]:
        return {42: "hat", 44: "hat", 46: "hat"}

    def _sounds(self, sr: int) -> dict[str, tuple[Callable[[np.random.Generator], np.ndarray], float]]:
        k, d = 2.0 ** (self.tune / 12.0), self.decay
        return {
            "kick": (lambda r: _kick(sr, r, k, d, self.punch), 0.95),
            "snare": (lambda r: _snare(sr, r, k, d), 0.75),
            "clap": (lambda r: _clap(sr, r, d), 0.6),
            "rim": (lambda r: _rim(sr, r), 0.5),
            "hat": (lambda r: _hat(sr, r, k, 0.045 * d), 0.3),
            "pedal_hat": (lambda r: _hat(sr, r, k, 0.03 * d), 0.25),
            "open_hat": (lambda r: _hat(sr, r, k, 0.4 * d), 0.3),
            "crash": (lambda r: _cymbal(sr, r, k, 0.9 * d, 4500.0), 0.35),
            "ride": (lambda r: _cymbal(sr, r, k, 0.6 * d, 6000.0), 0.28),
            "tom_lo": (lambda r: _tom(sr, r, 90.0 * k, d), 0.7),
            "tom_mid": (lambda r: _tom(sr, r, 130.0 * k, d), 0.7),
            "tom_hi": (lambda r: _tom(sr, r, 180.0 * k, d), 0.7),
            "cowbell": (lambda r: _cowbell(sr, r, k, d), 0.45),
        }

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        name = v.params.get("sound") or GM_DRUM_NAMES.get(int(round(v.pitch)))
        sounds = self._sounds(ctx.sr)
        if name not in sounds:
            raise ValueError(f"The drum kit has no sound for {name or v.pitch!r}; sounds: {', '.join(sounds)}")
        make, peak = sounds[name]
        y = make(rng)
        y = y / max(float(np.max(np.abs(y))), 1e-9) * peak
        return self.level * vel_gain(v.vel) * fade(y, ctx.sr, fade_out=0.005)
