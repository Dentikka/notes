"""Bounded magnitude-only timbre data, without audio carriers or recorded phases."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpectralProfile:
    """At most 32 tones, 16 envelope knots and 8 independent noise bands.

    Times are seconds, amplitudes are linear peak amplitudes, noise_rms is RMS.
    Ratios refer to the fundamental. pitch/reference_hz describe the analyzed
    anchor; rendering uses the score's tuning. This is a conditional timbre
    description, not a physical model or a cross-pitch/velocity realism claim.
    """

    pitch: float = 69.
    reference_hz: float = 440.
    ratios: tuple[float, ...] = (1.,)
    times: tuple[float, ...] = (0., .01, .3, 1.)
    amplitudes: tuple[tuple[float, ...], ...] = ((0., .2, .08, 0.),)
    noise_rms: tuple[tuple[float, ...], ...] = ()
    noise_bands: tuple[tuple[float, float], ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError('unsupported spectral profile version')
        if not np.isfinite([self.pitch, self.reference_hz]).all() or self.reference_hz <= 0:
            raise ValueError('spectral anchor must have finite pitch and positive frequency')
        arrays = {}
        for name in ('ratios', 'times', 'amplitudes', 'noise_rms', 'noise_bands'):
            value = getattr(self, name)
            if np.iscomplexobj(value):
                raise ValueError('spectral profile values must be real')
            try:
                arrays[name] = np.asarray(value, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError('invalid numeric spectral profile') from error
            if not np.isfinite(arrays[name]).all():
                raise ValueError('spectral profile values must be finite')
        ratios, times, amplitude = (arrays[k] for k in ('ratios', 'times', 'amplitudes'))
        if (ratios.ndim != 1 or not 1 <= len(ratios) <= 32 or np.any(ratios <= 0)
                or np.any(np.diff(ratios) <= 0)):
            raise ValueError('spectral ratios require 1..32 positive increasing values')
        if (times.ndim != 1 or not 2 <= len(times) <= 16 or times[0] != 0 or np.any(np.diff(times) <= 0)):
            raise ValueError('spectral times require 2..16 increasing knots starting at zero')
        if amplitude.shape != (len(ratios), len(times)) or np.any(amplitude < 0):
            raise ValueError('spectral amplitudes must be nonnegative, one curve per ratio')
        noise, bands = arrays['noise_rms'], arrays['noise_bands']
        if noise.size == 0 and bands.size == 0:
            noise, bands = np.empty((0, len(times))), np.empty((0, 2))
        if (noise.ndim != 2 or not 0 <= len(noise) <= 8 or noise.shape[1] != len(times)
                or bands.shape != (len(noise), 2) or np.any(noise < 0)
                or np.any(bands <= 0) or np.any(bands[:, 1] <= bands[:, 0])):
            raise ValueError('spectral noise needs up to 8 positive frequency bands and matching RMS curves')
        object.__setattr__(self, 'ratios', tuple(map(float, ratios)))
        object.__setattr__(self, 'times', tuple(map(float, times)))
        for name, value in [('amplitudes', amplitude), ('noise_rms', noise), ('noise_bands', bands)]:
            object.__setattr__(self, name, tuple(tuple(map(float, row)) for row in value))
        object.__setattr__(self, 'pitch', float(self.pitch))
        object.__setattr__(self, 'reference_hz', float(self.reference_hz))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SpectralProfile:
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError('unknown or malformed spectral profile fields') from error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
