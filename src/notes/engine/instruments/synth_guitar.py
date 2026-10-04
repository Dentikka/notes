"""The first engine's electric guitar, kept for its synthetic tone.

A Karplus–Strong string seen through a pickup comb, one asymmetric tanh stage for the amp and
a smooth biquad cabinet: the guitar of First Groove (`examples/first_groove.py`, engine
cbd8640), before `ElectricGuitar` moved to the steel-string waveguide and a miked cabinet.
CLAP hears it as an organ or a chiptune lead rather than a guitar, and that is its use: it
sits with the drum machine, the synth bass and the FM keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.dsp import db_to_gain, dc_block, delay, filt, karplus_strong, vel_gain
from notes.engine.instruments.strings import _PICKUP_POSITION, _gate, _string, _t60
from notes.engine.registry import register_instrument

__all__ = ["SynthGuitar"]

_PICK = 0.18
#: A palm-muted chug: T60 (s) and the share of the pluck's brightness it keeps.
_MUTE_T60, _MUTE_BRIGHTNESS = 0.25, 0.45


@dataclass(frozen=True, kw_only=True)
class _OneStageAmp(Effect):
    """The first amp: tightening high-pass, mid push, one asymmetric tanh stage from +3 dB
    (clean) to +42 dB (heavy), not oversampled, a tone tilt, and up to -12 dB of make-down as
    the drive rises."""

    drive: float = 0.5
    tone: float = 0.5
    level: float = 0.5

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        sr, d = ctx.sr, float(np.clip(self.drive, 0.0, 1.0))
        y = filt(x, "highpass", 60.0 + 90.0 * d, sr, 0.7)
        y = filt(y, "peak", 800.0, sr, 0.8, 5.0 * d)
        bias = 0.15 * d
        y = dc_block(np.tanh(db_to_gain(3.0 + 39.0 * d) * y + bias) - np.tanh(bias), sr)
        tilt = (self.tone - 0.5) * 12.0
        y = filt(y, "lowshelf", 250.0, sr, 0.7071, -0.5 * tilt)
        y = filt(y, "highshelf", 2500.0, sr, 0.7071, tilt)
        return self.level * db_to_gain(-12.0 * (1.0 - np.exp(-d / 0.25))) * y


@dataclass(frozen=True, kw_only=True)
class _BiquadCabinet(Effect):
    """The first cabinet: a low bump, a presence peak and a 24 dB/oct roll-off from
    3.2 kHz (`tone` 0) to 5.8 kHz (`tone` 1)."""

    tone: float = 0.5

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        sr = ctx.sr
        top = 3200.0 + 2600.0 * self.tone
        y = filt(x, "highpass", 80.0, sr, 0.8)
        y = filt(y, "peak", 110.0, sr, 1.0, 2.0)
        y = filt(y, "peak", 2300.0, sr, 1.2, 3.0)
        return filt(filt(y, "lowpass", top, sr, 0.9), "lowpass", 1.1 * top, sr, 0.6)


@register_instrument("synth_guitar")
@dataclass(frozen=True, kw_only=True)
class SynthGuitar(Instrument):
    """Karplus–Strong strings through a pickup comb into a one-stage amp and a biquad cabinet.

    `drive` 0 is clean and 1 heavy distortion; `tone` darkens (0) or brightens (1) the amp and
    cabinet; `pickup` is 'neck' (round), 'middle' or 'bridge' (bright, nasal); `sustain` is
    the T60 (s) at A3 and `brightness` the pluck's spectral tilt. Per-note ``palm_mute=True``
    gives short, dark chugs; ``vibrato``, ``bend`` and ``slide`` move the string's pitch.
    """

    drive: float = 0.5
    tone: float = 0.5
    pickup: str = "bridge"
    sustain: float = 2.5
    brightness: float = 0.7
    release: float = 0.06
    level: float = 0.5

    def __post_init__(self) -> None:
        if self.pickup not in _PICKUP_POSITION:
            raise ValueError(f"pickup must be one of {', '.join(_PICKUP_POSITION)}, got {self.pickup!r}")

    def release_time(self) -> float:
        return self.release

    def inserts(self) -> tuple[Effect, ...]:
        return (_OneStageAmp(drive=self.drive, tone=self.tone), _BiquadCabinet(tone=self.tone))

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        sr = ctx.sr
        muted = bool(v.params.get("palm_mute", False))
        t60 = _MUTE_T60 if muted else _t60(self.sustain, v.freq)
        brightness = self.brightness * (_MUTE_BRIGHTNESS if muted else 1.0)
        position = _PICKUP_POSITION[self.pickup]
        lag = int(round(position * sr / v.freq))

        def string(m: int) -> np.ndarray:
            y = karplus_strong(v.freq, m, sr, rng, t60=t60, brightness=brightness, pick=_PICK)
            # The pickup senses the string at one point: a comb that shapes the harmonics.
            return (y - delay(y, lag)) / (2.0 * np.sin(np.pi * position)) if lag >= 1 else y

        gate = ctx.samples(v.dur)
        n = gate + ctx.samples(self.release)
        return self.level * vel_gain(v.vel) * _string(v, n, sr, string) * _gate(gate, n, sr, self.release)
