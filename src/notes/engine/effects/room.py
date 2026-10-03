"""A recording room: what a pair of room mics hears besides the instrument itself.

Drums on a record are never heard in a vacuum: overheads and room mics pick up the first
reflections from the walls, floor and ceiling within tens of milliseconds, then the
room's diffuse tail. The early reflections give a sense of space and size without a long
tail — the opposite of a big algorithmic reverb, which on drums sounds like a hall or a
plate rather than a room.

The impulse response is built once per setting and convolved:

- **early reflections** by the image-source method for a shoebox room (up to third-order
  images), two mics 60 cm apart; each bounce loses energy to the wall and its top end to
  absorption, so later reflections are darker;
- **diffuse tail**: decorrelated noise per mic, its decay frequency dependent (highs die
  faster), faded in after the first reflections (`notes.engine.drum_physics.wash`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import oaconvolve

from notes.engine.base import Effect, RenderContext
from notes.engine.drum_physics import wash
from notes.engine.dsp import filt
from notes.engine.registry import register_effect

__all__ = ["Room"]

_SPEED_OF_SOUND = 343.0
#: A mid-sized live room at size 0.5 (m), source and the centre of the mic pair in it.
_ROOM = np.array([7.0, 5.5, 3.2])
_SOURCE = np.array([2.6, 2.4, 1.0])
_MICS = np.array([4.6, 2.75, 1.8])
_MIC_SPACING = 0.6
_TAIL_SHARE = 0.55  # share of the reverberant energy in the diffuse tail


def _early(sr: int, dims: np.ndarray, src: np.ndarray, mic: np.ndarray, absorb: float, damp: float) -> np.ndarray:
    """Early reflections (orders 1–3) at one mic, darker with each bounce."""
    beta = np.sqrt(1.0 - absorb)
    direct = float(np.linalg.norm(src - mic))
    by_order: dict[int, list[tuple[float, float]]] = {}
    for u in itertools.product((0, 1), repeat=3):
        for i in itertools.product((-1, 0, 1), repeat=3):
            order = sum(abs(2 * ii - uu) for ii, uu in zip(i, u, strict=True))
            if not 1 <= order <= 3:
                continue
            image = (1 - 2 * np.array(u)) * src + 2 * np.array(i) * dims
            dist = float(np.linalg.norm(image - mic))
            by_order.setdefault(order, []).append(((dist - direct) / _SPEED_OF_SOUND, beta**order * direct / dist))
    length = int(0.12 * sr)
    out = np.zeros(length)
    for order, taps in by_order.items():
        layer = np.zeros(length)
        for delay_s, gain in taps:
            k = int(round(delay_s * sr))
            if 0 < k < length:
                layer[k] += gain
        cutoff = 16000.0 * (1.0 - 0.5 * damp) ** order
        out += filt(layer, "lowpass", min(cutoff, 0.45 * sr), sr, 0.7)
    return out


@lru_cache(maxsize=16)
def room_ir(sr: int, size: float, decay: float, damp: float) -> np.ndarray:
    """Stereo room response (2, n), without the direct sound, unit energy per channel."""
    scale = 0.5 + float(size)
    dims, src, mic = _ROOM * scale, _SOURCE * scale, _MICS * scale
    absorb = float(np.clip(0.161 * np.prod(dims) / (2 * (dims[0] * dims[1] + dims[0] * dims[2]
                                                        + dims[1] * dims[2]) * max(decay, 0.05)), 0.05, 0.95))
    n = int((0.05 + 1.2 * decay) * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1977)
    onset = 2.0 * np.linalg.norm(dims) / 3.0 / _SPEED_OF_SOUND  # roughly the mean free path in time
    rows = []
    for side in (-1.0, 1.0):
        early = np.zeros(n)
        e = _early(sr, dims, src, mic + np.array([0.0, side * _MIC_SPACING / 2.0, 0.0]), absorb, damp)
        early[: min(n, e.size)] = e[: min(n, e.size)]
        tau_high = decay * (1.0 - 0.8 * damp) / 6.9  # T60 = 6.9 tau
        tail = wash(t, sr, rng, low=60.0, high=0.45 * sr, tau_low=decay / 6.9, tau_high=tau_high)
        tail *= 1.0 - np.exp(-t / max(onset, 1e-3))
        early /= max(float(np.sqrt(np.sum(early**2))), 1e-12)
        tail /= max(float(np.sqrt(np.sum(tail**2))), 1e-12)
        rows.append(np.sqrt(1.0 - _TAIL_SHARE) * early + np.sqrt(_TAIL_SHARE) * tail)
    return np.stack(rows)


@register_effect("room")
@dataclass(frozen=True, kw_only=True)
class Room(Effect):
    """Room mics: early reflections of a shoebox room plus its short diffuse tail.

    `size` 0 small booth .. 1 large live room, `decay` the RT60 in seconds (a studio live
    room 0.4–0.9), `damp` how much darker each bounce and the tail get, `mix` the room's
    level: 1 gives the room as much energy as the dry signal. The dry signal passes untouched.
    """

    size: float = 0.5
    decay: float = 0.6
    damp: float = 0.5
    mix: float = 0.3

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        dry = np.atleast_2d(x)
        dry = np.repeat(dry, 2, axis=0) if dry.shape[0] == 1 else dry
        ir = room_ir(ctx.sr, round(float(self.size), 4), round(float(self.decay), 4), round(float(self.damp), 4))
        mono = dry.mean(axis=0)
        wet = np.stack([oaconvolve(mono, h)[: mono.size] for h in ir])
        return dry + np.sqrt(self.mix) * wet

    def tail(self, ctx: RenderContext) -> float:
        return 0.05 + 1.2 * float(self.decay)
