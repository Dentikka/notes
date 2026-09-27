"""Time-based effects: tempo-synced delay, chorus and an algorithmic reverb."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import oaconvolve

from notes.engine.base import Effect, RenderContext
from notes.engine.dsp import TAU, biquad, delay, feedback_delay
from notes.engine.registry import register_effect

__all__ = ["Chorus", "Delay", "Reverb"]

# Freeverb tunings (samples at 44.1 kHz) and the right-channel offset that decorrelates it.
_COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASSES = (556, 441, 341, 225)
_STEREO_SPREAD = 23
# Freeverb's input gain for the summed channels; the wet scaling is calibrated so that at the
# default size mix=1 gives the tail as much energy as the dry signal (mix=0.25 -> about -12 dB).
_INPUT_GAIN = 0.03
_WET_GAIN = 1.4
_MAX_IR_SECONDS = 15.0


@lru_cache(maxsize=32)
def _freeverb_ir(sr: int, feedback: float, damp: float, length: int) -> np.ndarray:
    """Impulse response of the Freeverb network, one row per output channel.

    The network is linear and time-invariant, so rendering it once for an impulse and then
    convolving is exact up to the truncation at `length` (chosen at about -90 dB).
    """
    scale = sr / 44100.0
    impulse = np.zeros(length)
    impulse[0] = 1.0
    b, a = np.array([1.0 - damp]), np.array([1.0, -damp])
    rows = []
    for spread in (0, _STEREO_SPREAD):
        acc = np.zeros(length)
        for size in _COMBS:
            m = max(1, round((size + spread) * scale))
            acc += delay(feedback_delay(impulse, m, feedback, b, a), m)
        for size in _ALLPASSES:
            m = max(1, round((size + spread) * scale))
            acc = delay(feedback_delay(acc, m, 0.5), m) - acc
        rows.append(acc)
    return np.stack(rows)


def _stereo(x: np.ndarray) -> np.ndarray:
    x2 = np.atleast_2d(x)
    return np.repeat(x2, 2, axis=0) if x2.shape[0] == 1 else x2


@register_effect("delay")
@dataclass(frozen=True, kw_only=True)
class Delay(Effect):
    """Tempo-synced echo: `time` in beats (0.75 = dotted eighth); repeats get darker via `tone`."""

    time: float = 0.75
    feedback: float = 0.35
    mix: float = 0.25
    tone: float = 3500.0

    def _seconds(self, ctx: RenderContext) -> float:
        return self.time * 60.0 / ctx.bpm

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        d = max(1, ctx.samples(self._seconds(ctx)))
        b, a = biquad("lowpass", self.tone, ctx.sr)
        x2 = np.atleast_2d(x)
        wet = np.stack([delay(feedback_delay(ch, d, self.feedback, b, a), d) for ch in x2])
        return x2 + self.mix * wet

    def tail(self, ctx: RenderContext) -> float:
        fb = min(max(self.feedback, 1e-3), 0.99)
        repeats = np.log(1e-3) / np.log(fb)
        return float(min(10.0, self._seconds(ctx) * (repeats + 1.0)))


@register_effect("chorus")
@dataclass(frozen=True, kw_only=True)
class Chorus(Effect):
    """Modulated short delay per channel with LFOs in quadrature — widens and thickens."""

    rate: float = 0.8
    depth: float = 0.003
    base_delay: float = 0.012
    mix: float = 0.5

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        x2 = _stereo(x)
        n = x2.shape[1]
        idx = np.arange(n, dtype=float)
        t = (idx + ctx.offset) / ctx.sr
        out = np.empty_like(x2, dtype=float)
        for ch in range(2):
            lfo = 0.5 * (1.0 + np.sin(TAU * self.rate * t + ch * np.pi / 2.0))
            lag = (self.base_delay + self.depth * lfo) * ctx.sr
            wet = np.interp(idx - lag, idx, x2[ch], left=0.0)
            out[ch] = (x2[ch] + self.mix * wet) / (1.0 + 0.5 * self.mix)
        return out

    def tail(self, ctx: RenderContext) -> float:
        return self.base_delay + self.depth


@register_effect("reverb")
@dataclass(frozen=True, kw_only=True)
class Reverb(Effect):
    """Freeverb (Schroeder–Moorer): eight damped combs into four allpasses per channel.

    `size` sets the comb feedback (0.70–0.98), `damp` how fast the highs die, `width` the
    stereo spread of the tail; the dry signal passes through untouched. `mix` is the wet level:
    1 puts as much energy in the tail as in the dry signal (at the default size).
    Rendered as a convolution with the network's cached impulse response.
    """

    size: float = 0.6
    damp: float = 0.4
    mix: float = 0.25
    width: float = 1.0
    predelay: float = 0.01

    def _feedback(self) -> float:
        return 0.7 + 0.28 * float(np.clip(self.size, 0.0, 1.0))

    def _t60(self) -> float:
        """Decay time at DC: the comb loop gain `feedback` per mean comb period."""
        loop = float(np.mean(_COMBS)) / 44100.0
        return 3.0 * loop / -np.log10(self._feedback())

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        dry = _stereo(x)
        inp = delay(dry.mean(axis=0), ctx.samples(self.predelay)) * _INPUT_GAIN
        damp = round(0.4 * float(np.clip(self.damp, 0.0, 1.0)), 6)
        length = ctx.samples(min(1.5 * self._t60(), _MAX_IR_SECONDS))
        ir = _freeverb_ir(ctx.sr, round(self._feedback(), 6), damp, length)
        left, right = (oaconvolve(inp, h)[: inp.size] for h in ir)
        w1, w2 = self.width / 2.0 + 0.5, (1.0 - self.width) / 2.0
        tail = np.stack([w1 * left + w2 * right, w1 * right + w2 * left])
        return dry + self.mix * _WET_GAIN * tail

    def tail(self, ctx: RenderContext) -> float:
        return float(min(_MAX_IR_SECONDS, self.predelay + 1.5 * self._t60()))
