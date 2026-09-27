"""Tone-shaping effects: filters, EQ, saturation, and the guitar amp and cabinet models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.base import Effect, RenderContext
from notes.engine.dsp import db_to_gain, dc_block, filt
from notes.engine.registry import register_effect

__all__ = ["EQ", "Cabinet", "Drive", "Filter", "GuitarAmp"]


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
        wet = filt(np.tanh(k * x) / np.tanh(k), "lowpass", self.tone, ctx.sr)
        return (1.0 - self.mix) * x + self.mix * wet


@register_effect("guitar_amp")
@dataclass(frozen=True, kw_only=True)
class GuitarAmp(Effect):
    """Guitar amplifier: tightening high-pass, mid push, asymmetric soft clipping, tone tilt.

    `drive` sets the pre-gain from +3 dB (clean) to +42 dB (heavy); the bias makes clipping
    asymmetric, which adds the even harmonics of a tube stage.
    """

    drive: float = 0.5
    tone: float = 0.5
    level: float = 0.5

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        sr, d = ctx.sr, float(np.clip(self.drive, 0.0, 1.0))
        y = filt(x, "highpass", 60.0 + 90.0 * d, sr, 0.7)
        y = filt(y, "peak", 800.0, sr, 0.8, 5.0 * d)
        bias = 0.15 * d
        y = np.tanh(db_to_gain(3.0 + 39.0 * d) * y + bias) - np.tanh(bias)
        y = dc_block(y, sr)
        tilt = (self.tone - 0.5) * 12.0
        y = filt(y, "lowshelf", 250.0, sr, 0.7071, -0.5 * tilt)
        y = filt(y, "highshelf", 2500.0, sr, 0.7071, tilt)
        return self.level * y


@register_effect("cabinet")
@dataclass(frozen=True, kw_only=True)
class Cabinet(Effect):
    """Closed-back speaker cabinet: band-limits and colours the amp (24 dB/oct top roll-off)."""

    tone: float = 0.5

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        sr = ctx.sr
        top = 3200.0 + 2600.0 * self.tone
        y = filt(x, "highpass", 80.0, sr, 0.8)
        y = filt(y, "peak", 110.0, sr, 1.0, 2.0)
        y = filt(y, "peak", 2300.0, sr, 1.2, 3.0)
        return filt(filt(y, "lowpass", top, sr, 0.9), "lowpass", 1.1 * top, sr, 0.6)
