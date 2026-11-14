"""Designed spatial variation of contact material, independent of render clocks.

Gaussian AR(1) knots at length/4 spacing share a stationary origin, with separate
positive/negative innovation streams. Linear interpolation and clipping at +/-3
are explicit modelling approximations, not measured surface data. Values are
generated on demand in fixed blocks, preserving already visited material.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter


class SpatialGrain:
    """A bounded material value at a signed position in metres.

    Each direction caches at most 2**20 knots. Queries outside this range fail
    before allocating an unbounded material history.
    """

    def __init__(self, length: float = 1e-4, seed: int = 17) -> None:
        try:
            length = float(length)
        except (TypeError, ValueError) as error:
            raise ValueError('positive finite material correlation length required') from error
        if not math.isfinite(length) or length/4 <= 0:
            raise ValueError('positive finite material correlation length required')
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise ValueError('grain seed must be an unsigned 64-bit integer')
        self.spacing = length/4
        self._rho = float(np.exp(-.25))
        self._scale = float(np.sqrt(1-self._rho**2))
        streams = np.random.SeedSequence(seed).spawn(3)
        origin = float(np.random.default_rng(streams[0]).normal())
        self._values = [[origin], [origin]]
        self._generators = [np.random.default_rng(stream) for stream in streams[1:]]

    def _knot(self, index: int) -> float:
        side = int(index < 0)
        index = abs(index)
        if index >= 2**20:
            raise ValueError('material coordinate exceeds the spatial cache; increase grain length')
        values = self._values[side]
        while len(values) <= index:
            innovation = self._generators[side].standard_normal(min(256, 2**20-len(values)))
            block, _ = lfilter([self._scale], [1., -self._rho], innovation, zi=[self._rho*values[-1]])
            values.extend(block.tolist())
        return values[index]

    def __call__(self, position: float) -> float:
        try:
            coordinate = float(position)/self.spacing
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('material coordinate must be finite') from error
        if not math.isfinite(coordinate):
            raise ValueError('material coordinate must be finite')
        left = math.floor(coordinate)
        fraction = coordinate-left
        a, b = self._knot(left), self._knot(left+1)
        return min(3., max(-3., a+(b-a)*fraction))
