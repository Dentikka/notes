"""Dynamics: a feed-forward compressor with a soft knee."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from notes.engine.base import Effect, RenderContext
from notes.engine.registry import register_effect

__all__ = ["Compressor"]

_BLOCK = 32


@register_effect("compressor")
@dataclass(frozen=True, kw_only=True)
class Compressor(Effect):
    """Peak compressor: level tracked per 32-sample block with separate attack and release."""

    threshold: float = -18.0
    ratio: float = 3.0
    attack: float = 0.01
    release: float = 0.15
    knee: float = 6.0
    makeup: float = 0.0

    def _reduction(self, level_db: np.ndarray) -> np.ndarray:
        over = level_db - self.threshold
        slope = 1.0 / self.ratio - 1.0
        k = max(self.knee, 1e-6)
        soft = slope * (over + k / 2.0) ** 2 / (2.0 * k)
        return np.where(2.0 * over < -k, 0.0, np.where(2.0 * np.abs(over) <= k, soft, slope * over))

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        x2 = np.atleast_2d(x)
        n = x2.shape[1]
        blocks = -(-n // _BLOCK)
        level = np.zeros(blocks * _BLOCK)
        level[:n] = np.max(np.abs(x2), axis=0)
        peaks_db = 20.0 * np.log10(np.maximum(level.reshape(blocks, _BLOCK).max(axis=1), 1e-9))
        att = float(np.exp(-_BLOCK / (max(self.attack, 1e-4) * ctx.sr)))
        rel = float(np.exp(-_BLOCK / (max(self.release, 1e-4) * ctx.sr)))
        env = np.empty(blocks)
        e = -120.0
        for i, target in enumerate(peaks_db):
            coef = att if target > e else rel
            e = coef * e + (1.0 - coef) * target
            env[i] = e
        centers = (np.arange(blocks) + 0.5) * _BLOCK
        gain_db = np.interp(np.arange(n), centers, self._reduction(env)) + self.makeup
        return x2 * 10.0 ** (gain_db / 20.0)
