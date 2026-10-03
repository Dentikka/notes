"""Tone-shaping effects: filters, EQ, saturation, and the guitar amp and cabinet models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import oaconvolve, resample_poly

from notes.engine.base import Effect, RenderContext
from notes.engine.dsp import db_to_gain, dc_block, filt
from notes.engine.effects.speaker import speaker_ir
from notes.engine.registry import register_effect

__all__ = ["EQ", "Cabinet", "Drive", "Filter", "GuitarAmp", "saturate"]

_OVERSAMPLE = 4


def saturate(x: np.ndarray, gain: float, bias: float = 0.0, oversample: int = _OVERSAMPLE) -> np.ndarray:
    """tanh(gain * x + bias) - tanh(bias) evaluated at `oversample` times the sample rate.

    The polyphase resampler's low-pass removes the new harmonics above the original Nyquist
    before decimation, so they do not alias back into the audible band.
    """
    if oversample <= 1:
        return np.tanh(gain * x + bias) - np.tanh(bias)
    up = resample_poly(x, oversample, 1, axis=-1)
    return resample_poly(np.tanh(gain * up + bias) - np.tanh(bias), 1, oversample, axis=-1)[..., : x.shape[-1]]


@register_effect("filter")
@dataclass(frozen=True, kw_only=True)
class Filter(Effect):
    """A single biquad: lowpass, highpass, bandpass, notch, peak, lowshelf or highshelf."""

    kind: str = "lowpass"
    cutoff: float = 2000.0
    q: float = 0.7071
    gain_db: float = 0.0

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        return filt(x, self.kind, self.cutoff, ctx.sr, self.q, self.gain_db)


@register_effect("eq")
@dataclass(frozen=True, kw_only=True)
class EQ(Effect):
    """Three-band EQ: low shelf, mid peak, high shelf; gains in dB."""

    low: float = 0.0
    low_freq: float = 120.0
    mid: float = 0.0
    mid_freq: float = 1000.0
    mid_q: float = 0.8
    high: float = 0.0
    high_freq: float = 6000.0

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        y = x
        if self.low:
            y = filt(y, "lowshelf", self.low_freq, ctx.sr, 0.7071, self.low)
        if self.mid:
            y = filt(y, "peak", self.mid_freq, ctx.sr, self.mid_q, self.mid)
        if self.high:
            y = filt(y, "highshelf", self.high_freq, ctx.sr, 0.7071, self.high)
        return y


@register_effect("drive")
@dataclass(frozen=True, kw_only=True)
class Drive(Effect):
    """Symmetric tanh saturation followed by a lowpass that tames the new harmonics."""

    amount: float = 0.3
    tone: float = 6000.0
    mix: float = 1.0

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        k = 1.0 + 24.0 * self.amount
        wet = filt(saturate(x, k) / np.tanh(k), "lowpass", self.tone, ctx.sr)
        return (1.0 - self.mix) * x + self.mix * wet


@register_effect("guitar_amp")
@dataclass(frozen=True, kw_only=True)
class GuitarAmp(Effect):
    """Two-stage guitar amplifier, clipped at 4x the sample rate.

    Preamp: a tightening high-pass and a mid push feed asymmetric tanh clipping (the bias
    adds a tube stage's even harmonics) with a gain from +3 dB (clean) to +36 dB (heavy).
    Tone stack: a mid scoop that deepens with gain and a tilt set by `tone`. Power amp: a
    gentle symmetric squash, then presence. Clipping runs oversampled, so harmonics above
    Nyquist are filtered out instead of folding back as the inharmonic fizz of digital
    distortion. The output is trimmed from -4.6 dB (clean) to about -13 dB (heavy), so that
    gain changes the character, not the loudness.
    """

    drive: float = 0.5
    tone: float = 0.5
    level: float = 0.5

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        sr, d = ctx.sr, float(np.clip(self.drive, 0.0, 1.0))
        y = filt(x, "highpass", 60.0 + 90.0 * d, sr, 0.7)
        y = filt(y, "peak", 800.0, sr, 0.8, 5.0 * d)
        y = dc_block(saturate(y, db_to_gain(3.0 + 33.0 * d), 0.15 * d), sr)
        tilt = (self.tone - 0.5) * 12.0
        y = filt(y, "peak", 650.0, sr, 0.7, -4.0 * d)
        y = filt(y, "lowshelf", 250.0, sr, 0.7071, -0.5 * tilt)
        y = filt(y, "highshelf", 2500.0, sr, 0.7071, tilt)
        power = 1.0 + 1.5 * d
        y = saturate(y, power) / np.tanh(power)
        y = filt(y, "highshelf", 3500.0, sr, 0.7071, 2.0 * self.tone)
        return self.level * db_to_gain(-4.6 - 8.7 * (1.0 - np.exp(-d / 0.28))) * y


@register_effect("cabinet")
@dataclass(frozen=True, kw_only=True)
class Cabinet(Effect):
    """Closed-back 4x12 cabinet, miked.

    `model='greenback'` convolves with a designed impulse response (`notes.engine.effects.speaker`):
    jagged cone-breakup response, presence hump, a cliff above 5 kHz and, scaled by `room`,
    the floor and room reflections the mic hears. `model='filters'` is the earlier smooth
    biquad cabinet: low resonance, low-mid dip, presence and breakup peaks, then a steep
    roll-off. `tone` brightens either: presence and the corner of the roll-off.
    """

    tone: float = 0.5
    model: str = "greenback"
    room: float = 0.2

    def __post_init__(self) -> None:
        if self.model not in ("greenback", "filters"):
            raise ValueError(f"cabinet model must be 'greenback' or 'filters', got {self.model!r}")

    def tail(self, ctx: RenderContext) -> float:
        return 0.05 if self.model == "greenback" else 0.0

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        if self.model == "greenback":
            ir = speaker_ir(ctx.sr, round(float(self.tone), 4), round(float(self.room), 4))
            return oaconvolve(x, ir.reshape((1,) * (x.ndim - 1) + (-1,)), axes=-1)[..., : x.shape[-1]]
        sr = ctx.sr
        top = 4200.0 + 1800.0 * self.tone
        y = filt(x, "highpass", 75.0, sr, 0.7)
        y = filt(y, "peak", 110.0, sr, 1.2, 3.0)
        y = filt(y, "peak", 500.0, sr, 1.0, -3.0)
        y = filt(y, "peak", 2400.0, sr, 1.5, 4.0)
        y = filt(y, "peak", 3800.0, sr, 2.0, 3.0)
        for q in (0.54, 1.31):
            y = filt(y, "lowpass", top, sr, q)
        return filt(y, "lowpass", 1.4 * top, sr, 0.7)
