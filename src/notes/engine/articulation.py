"""Expressive pitch: vibrato, bends and slides, read from per-note parameters.

Parameters (all optional; set them with ``Music.vibrato/bend/slide`` or the melody
notation ``~`` and ``^n``):

    vibrato  depth in semitones (peak)          vibrato_rate Hz (5.5)   vibrato_delay s (0.2)
    bend     semitones the bend reaches         bend_start s (0.0)      bend_time s (0.15)
    slide    semitones away the note glides in from at its onset        slide_time s (0.08)

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


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pitch_curve(params: Mapping[str, Any], n: int, sr: int) -> np.ndarray | None:
    """Pitch offset in semitones for each of the `n` samples of a voice; None for a plain note."""
    vibrato = float(params.get("vibrato", 0.0))
    bend = float(params.get("bend", 0.0))
    slide = float(params.get("slide", 0.0))
    if not (vibrato or bend or slide):
        return None
    t = np.arange(n) / sr
    curve = np.zeros(n)
    if slide:
        curve += slide * (1.0 - _smoothstep(t / max(float(params.get("slide_time", 0.08)), 1e-3)))
    if bend:
        start, time = float(params.get("bend_start", 0.0)), max(float(params.get("bend_time", 0.15)), 1e-3)
        curve += bend * _smoothstep((t - start) / time)
    if vibrato:
        delay, rate = float(params.get("vibrato_delay", 0.2)), float(params.get("vibrato_rate", 5.5))
        fade = np.clip((t - delay) / _VIBRATO_FADE_S, 0.0, 1.0)
        curve += vibrato * fade * np.sin(2.0 * np.pi * rate * (t - delay))
    return curve


def variable_rate_read(render: Callable[[int], np.ndarray], curve: np.ndarray) -> np.ndarray:
    """Play a fixed-pitch signal back at a varying rate so its pitch follows `curve` (semitones).

    `render(m)` must return m samples of the source; it is asked for exactly as many as the
    readout consumes.
    """
    ratio = 2.0 ** (curve / 12.0)
    position = np.concatenate(([0.0], np.cumsum(ratio[:-1])))
    source = render(int(np.ceil(position[-1])) + 2)
    return np.interp(position, np.arange(source.size), source)
