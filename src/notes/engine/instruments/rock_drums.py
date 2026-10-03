"""An acoustic rock kit by modal synthesis (`notes.engine.drum_physics`): 22" kick, 14"
snare with wires, three toms, hi-hats, crash and ride.

Unlike the drum machine (`DrumKit`), every hit is a little different (fresh mode
scatter, phases and noise per hit), and velocity changes the timbre, not just the level:
a harder hit is brighter, clicks more and starts sharper in pitch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.drum_physics import impact, membrane, metal, wash
from notes.engine.dsp import fade, filt, vel_gain
from notes.engine.instruments.drums import _clap, _cowbell
from notes.engine.registry import register_instrument
from notes.ir.gm import GM_DRUM_NAMES

__all__ = ["RockKit"]

#: Fundamentals (Hz) of the kit's heads at tune 0.
_KICK_HZ, _SNARE_HZ = 55.0, 195.0
_TOM_HZ = {"tom_lo": 82.0, "tom_mid": 118.0, "tom_hi": 156.0}


def _time(seconds: float, sr: int) -> np.ndarray:
    return np.arange(max(1, int(seconds * sr))) / sr


def _kick(sr: int, rng: np.random.Generator, vel: float, k: float, d: float, click: float) -> np.ndarray:
    t = _time(0.9 * d, sr)
    head = membrane(t, _KICK_HZ * k, 0.16 * d, rng, strike=0.15, glide=0.35 * vel, n_modes=8, damping=1.3)
    # The shell and the ported front head: a short boom an octave up.
    boom = membrane(t, 2.0 * _KICK_HZ * k, 0.05 * d, rng, strike=0.0, n_modes=3)
    beater = impact(t.size, sr, rng, hardness=0.4 + 0.5 * vel, length=0.0012)
    y = head + 0.3 * boom + click * (0.6 + 0.8 * vel) * beater
    # Miked inside: boxiness dipped, the beater's click lifted.
    y = filt(y, "peak", 380.0, sr, 1.0, -6.0)
    return filt(filt(y, "peak", 3500.0, sr, 1.0, 4.0 * click), "lowpass", 9000.0, sr, 0.7)


def _snare(sr: int, rng: np.random.Generator, vel: float, k: float, d: float, wires: float, rimshot: float):
    t = _time(0.7 * d, sr)
    head = membrane(t, _SNARE_HZ * k, 0.09 * d, rng, strike=0.45, glide=0.12 * vel, n_modes=10, damping=1.2)
    # The wires buzz against the snare-side head: noise gated by the head's own motion.
    motion = filt(np.abs(head), "lowpass", 60.0, sr, 0.7)
    motion /= max(float(np.max(motion)), 1e-9)
    buzz = filt(filt(rng.uniform(-1.0, 1.0, t.size), "highpass", 1800.0, sr, 0.7), "lowpass", 11000.0, sr, 0.7)
    body = filt(rng.uniform(-1.0, 1.0, t.size), "bandpass", 450.0, sr, 0.8)
    rattle = (0.85 * buzz + 0.3 * body) * (0.4 * motion + np.exp(-t / (0.16 * d)))
    stick = impact(t.size, sr, rng, hardness=0.7 + 0.3 * vel, length=0.0008)
    y = head + wires * (0.5 + 0.5 * vel) * rattle + 0.7 * stick
    if rimshot:
        # Stick across head and rim at once: the shell and rim ring, a hard crack on top.
        ring = metal(t, rng, n_modes=12, low=420.0, high=1900.0, tau_low=0.06, tau_high=0.03)
        y = y + rimshot * (0.8 * ring + 0.6 * impact(t.size, sr, rng, hardness=1.0, length=0.0005))
    return y


def _tom(sr: int, rng: np.random.Generator, vel: float, f0: float, d: float) -> np.ndarray:
    t = _time(1.4 * d, sr)
    head = membrane(t, f0, 0.32 * d, rng, strike=0.35, glide=0.18 * vel, n_modes=9, damping=1.4)
    return head + 0.5 * impact(t.size, sr, rng, hardness=0.6 + 0.3 * vel, length=0.0008)


def _norm(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.sqrt(np.mean(x[: max(1, x.size // 8)] ** 2))), 1e-12)


def _hat(sr: int, rng: np.random.Generator, vel: float, k: float, tau: float, length: float) -> np.ndarray:
    t = _time(length, sr)
    body = wash(t, sr, rng, low=500.0 * k, high=17000.0, tau_low=tau, tau_high=0.7 * tau, tilt=2.5 + vel, bloom=0.002)
    shimmer = metal(t, rng, n_modes=60, low=3000.0 * k, high=14000.0, tau_low=tau, tau_high=0.6 * tau, tilt=-1.0)
    stick = impact(t.size, sr, rng, hardness=0.8 + 0.2 * vel, length=0.0005)
    return _norm(body) + 0.25 * _norm(shimmer) + 1.5 * filt(stick, "highpass", 2000.0, sr, 0.7)


def _pedal_hat(sr: int, rng: np.random.Generator) -> np.ndarray:
    """The two plates clapped shut by the foot: a short 'chick'."""
    t = _time(0.12, sr)
    chick = filt(rng.uniform(-1.0, 1.0, t.size), "bandpass", 4500.0, sr, 1.2) * np.exp(-t / 0.012)
    return chick + 0.5 * _hat(sr, rng, 0.4, 1.0, 0.02, 0.12)


def _crash(sr: int, rng: np.random.Generator, vel: float, k: float, d: float) -> np.ndarray:
    t = _time(4.5 * d, sr)
    body = wash(t, sr, rng, low=300.0 * k, high=18000.0, tau_low=1.6 * d, tau_high=0.6 * d,
                tilt=1.5 + 1.5 * vel, bloom=0.06)
    shimmer = metal(t, rng, n_modes=200, low=300.0 * k, high=12000.0, tau_low=1.5 * d, tau_high=0.6 * d,
                    tilt=0.5, bloom=0.06)
    stick = impact(t.size, sr, rng, hardness=0.9, length=0.0006)
    return fade(_norm(body) + 0.3 * _norm(shimmer) + 2.0 * stick, sr, fade_out=0.3)


def _ride(sr: int, rng: np.random.Generator, vel: float, k: float, d: float) -> np.ndarray:
    t = _time(3.0 * d, sr)
    body = wash(t, sr, rng, low=500.0 * k, high=15000.0, tau_low=2.2 * d, tau_high=1.0 * d, tilt=0.5, bloom=0.03)
    # The stick's 'ping': a bright cluster of modes that dies quickly, over a quieter wash.
    ping = metal(t, rng, n_modes=40, low=2500.0 * k, high=6500.0 * k, tau_low=0.25, tau_high=0.12)
    stick = impact(t.size, sr, rng, hardness=1.0, length=0.0004)
    return fade(0.35 * _norm(body) + (0.8 + 0.4 * vel) * _norm(ping) + 1.5 * stick, sr, fade_out=0.3)


def _cross_stick(sr: int, rng: np.random.Generator) -> np.ndarray:
    """Stick laid across the snare, struck on the rim: a woody knock."""
    t = _time(0.12, sr)
    knock = metal(t, rng, n_modes=8, low=700.0, high=2600.0, tau_low=0.03, tau_high=0.015)
    return knock + 0.4 * membrane(t, _SNARE_HZ, 0.04, rng, n_modes=4) + 0.4 * impact(t.size, sr, rng, hardness=0.8)


@register_instrument("rock_drums")
@dataclass(frozen=True, kw_only=True)
class RockKit(Instrument):
    """Acoustic rock kit by modal synthesis, addressed like `DrumKit` (GM numbers or
    ``sound`` names: kick, snare, rim, tom_lo/mid/hi, hat, pedal_hat, open_hat, crash, ride,
    clap, cowbell).

    `tune` shifts the heads and cymbals in semitones, `decay` scales ring times (a damped
    70s kit ~0.8, an open 80s kit ~1.3), `click` the kick beater's attack, `wires` the
    snare's buzz, `rimshot` how much of each backbeat lands on the rim too. Velocity
    shapes the timbre; closed hats choke the open hat.
    """

    program: int | None = 0
    tune: float = 0.0
    decay: float = 1.0
    click: float = 0.6
    wires: float = 0.8
    rimshot: float = 0.3
    level: float = 0.9

    def release_time(self) -> float:
        return 4.5 * self.decay + 0.1  # sounds ignore the gate; the crash rings longest

    def choke_groups(self) -> Mapping[int, str]:
        return {42: "hat", 44: "hat", 46: "hat"}

    def _sounds(self, sr: int, vel: float) -> dict[str, tuple[Callable[[np.random.Generator], np.ndarray], float]]:
        k, d = 2.0 ** (self.tune / 12.0), self.decay
        sounds = {
            "kick": (lambda r: _kick(sr, r, vel, k, d, self.click), 0.95),
            "snare": (lambda r: _snare(sr, r, vel, k, d, self.wires, self.rimshot), 0.8),
            "rim": (lambda r: _cross_stick(sr, r), 0.45),
            "hat": (lambda r: _hat(sr, r, vel, k, 0.035 * d, 0.25), 0.28),
            "pedal_hat": (lambda r: _pedal_hat(sr, r), 0.22),
            "open_hat": (lambda r: _hat(sr, r, vel, k, 0.45 * d, 2.0 * d), 0.3),
            "crash": (lambda r: _crash(sr, r, vel, k, d), 0.42),
            "ride": (lambda r: _ride(sr, r, vel, k, d), 0.3),
            "clap": (lambda r: _clap(sr, r, d), 0.6),
            "cowbell": (lambda r: _cowbell(sr, r, k, d), 0.45),
        }
        for name, f0 in _TOM_HZ.items():
            sounds[name] = (lambda r, f=f0: _tom(sr, r, vel, f * k, d), 0.75)
        return sounds

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        name = v.params.get("sound") or GM_DRUM_NAMES.get(int(round(v.pitch)))
        sounds = self._sounds(ctx.sr, float(np.clip(v.vel, 0.0, 1.0)))
        if name not in sounds:
            raise ValueError(f"The rock kit has no sound for {name or v.pitch!r}; sounds: {', '.join(sounds)}")
        make, peak = sounds[name]
        y = make(rng)
        y = y / max(float(np.max(np.abs(y))), 1e-9) * peak
        return self.level * vel_gain(v.vel) * fade(y, ctx.sr, fade_out=0.005)
