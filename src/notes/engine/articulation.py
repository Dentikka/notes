"""Expressive pitch: vibrato, bends and slides, read from per-note parameters.

Parameters (all optional; set them with ``Music.vibrato/bend/slide`` or the melody
notation ``~`` and ``^n``):

    vibrato  depth in semitones (peak)          vibrato_rate Hz (5.5)   vibrato_delay s (0.2)
    bend     semitones the bend reaches         bend_start s (0.0)      bend_time s (0.15)
    slide    semitones away the note glides in from at its onset        slide_time s (0.08)
    pitch_points  [(seconds from onset, semitone offset), ...], linearly interpolated

`vibrato_shape` chooses how the pitch moves: 'sine' swings evenly around the note (voice,
violin, synth); 'push' only ever raises it, the way a guitarist's finger pushes the string
across the fret and lets it back, with the same peak-to-peak swing and a rate that wanders
by a few per cent (given an `rng`).

Instruments with oscillators turn the curve into a frequency track. Strings are rendered
at a fixed pitch and read back at a varying rate, which moves the string's harmonics with
the pitch — what bending a real string does — while the amp and cabinet after it stay put.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = ["pitch_curve", "variable_rate_read"]

_VIBRATO_FADE_S = 0.3
_RATE_WANDER = 0.08


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _vibrato_phase(rate: float, t: np.ndarray, sr: int, rng: np.random.Generator | None) -> np.ndarray:
    """Phase (radians) of a vibrato starting at t = 0, its rate wandering slowly given an rng."""
    inst = np.full(t.shape, rate)
    if rng is not None:
        slow = sum(np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi)) for f in (0.37, 0.83))
        inst *= 1.0 + 0.5 * _RATE_WANDER * slow
    inst[t <= 0.0] = 0.0
    return 2.0 * np.pi * np.cumsum(inst) / sr


def pitch_curve(
    params: Mapping[str, Any],
    n: int,
    sr: int,
    *,
    vibrato_shape: str = "sine",
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Pitch offset in semitones for each of the `n` samples of a voice; None for a plain note."""
    vibrato = float(params.get("vibrato", 0.0))
    bend = float(params.get("bend", 0.0))
    slide = float(params.get("slide", 0.0))
    points = params.get("pitch_points")
    if not (vibrato or bend or slide) and points is None:
        return None
    t = np.arange(n) / sr
    curve = np.zeros(n)
    if points is not None:
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 1
                or not np.isfinite(points).all() or points[0, 0] < 0 or np.any(np.diff(points[:, 0]) <= 0)):
            raise ValueError("pitch_points must be finite (nonnegative increasing seconds, semitones) pairs")
        curve += np.interp(t, points[:, 0], points[:, 1])
    if slide:
        curve += slide * (1.0 - _smoothstep(t / max(float(params.get("slide_time", 0.08)), 1e-3)))
    if bend:
        start, time = float(params.get("bend_start", 0.0)), max(float(params.get("bend_time", 0.15)), 1e-3)
        curve += bend * _smoothstep((t - start) / time)
    if vibrato:
        delay, rate = float(params.get("vibrato_delay", 0.2)), float(params.get("vibrato_rate", 5.5))
        fade = np.clip((t - delay) / _VIBRATO_FADE_S, 0.0, 1.0)
        if vibrato_shape == "push":
            phase = _vibrato_phase(rate, t - delay, sr, rng)
            curve += vibrato * fade * (1.0 - np.cos(phase))
        elif vibrato_shape == "sine":
            curve += vibrato * fade * np.sin(2.0 * np.pi * rate * (t - delay))
        else:
            raise ValueError(f"vibrato_shape must be 'sine' or 'push', got {vibrato_shape!r}")
    return curve


def variable_rate_read(render: Callable[[int], np.ndarray], curve: np.ndarray, *,
                       interpolation: str = "linear") -> np.ndarray:
    """Play a fixed-pitch signal back at a varying rate so its pitch follows `curve` (semitones).

    `render(m)` must return m samples of the source; it is asked for exactly as many as the
    readout consumes, plus filter lookahead in sinc mode. The original linear
    interpolation remains the default. Optional sinc interpolation suppresses
    fractional-sample coloration and applies a rate-dependent anti-alias filter.
    Its upward rate is limited to 16x (four octaves) to bound the filter support.
    """
    if interpolation not in ("linear", "sinc"):
        raise ValueError("pitch interpolation must be linear or sinc")
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 1 or not np.isfinite(curve).all():
        raise ValueError("pitch curve must be a finite one-dimensional array")
    if not len(curve):
        return np.zeros(0)
    ratio = 2.0 ** (curve / 12.0)
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("pitch curve produces an invalid playback rate")
    position = np.concatenate(([0.0], np.cumsum(ratio[:-1])))
    if interpolation == "sinc" and np.any(curve):
        if ratio.max() > 16:
            raise ValueError("sinc pitch interpolation supports at most four octaves upward")
        return _sinc_read(render, position, ratio)
    source = render(int(np.ceil(position[-1])) + 2)
    return np.interp(position, np.arange(source.size), source)


def _sinc_read(render: Callable[[int], np.ndarray], position: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Blackman-windowed sinc, 16 zero crossings on either side at the local rate.

    Positions remain centered on the requested times: filter lookahead introduces
    no extra event delay. Samples before the source begins are zero. Chunking
    bounds temporary arrays independently of the note length.
    """
    cutoff = np.minimum(1.0, 1.0/ratio)
    radius = int(np.ceil(16.0/cutoff.min()))
    count = int(np.ceil(position[-1])) + radius + 1
    source = np.asarray(render(count), dtype=float)
    if source.shape != (count,) or not np.isfinite(source).all():
        raise ValueError("pitch source must return the requested number of finite mono samples")
    offsets = np.arange(-radius+1, radius+1)
    output = np.empty(len(position))
    for start in range(0, len(position), 4096):
        stop = min(len(position), start+4096)
        at = position[start:stop, None]
        indices = np.floor(at).astype(np.int64) + offsets
        distance = (at-indices) * cutoff[start:stop, None]
        inside = abs(distance) < 16
        window = .42 + .5*np.cos(np.pi*distance/16) + .08*np.cos(2*np.pi*distance/16)
        weights = np.sinc(distance) * window * inside
        weights /= weights.sum(axis=1, keepdims=True)
        values = source[np.clip(indices, 0, count-1)] * (indices >= 0)
        output[start:stop] = np.sum(weights*values, axis=1)
    return output
