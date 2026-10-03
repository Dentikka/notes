"""A plucked steel string for the electric guitar: digital waveguide with physically shaped
excitation, frequency-dependent losses, stiffness dispersion and two polarisations.

Compared with plain Karplus–Strong (`notes.engine.dsp.karplus_strong`), every piece here
removes one reason a synthetic string sounds synthetic:

- **pluck** — the excitation is the velocity profile of a string pulled at the pick point
  (spectrum ~ sin(pi h p) / h), rounded by the pick's hardness, with a little noise. A
  noise burst instead gives each note a random timbre and often a weak fundamental;
- **losses** — a one-pole loop filter fitted to two decay times: `t60` for the fundamental
  and `t60_high` at 4 kHz. The two-tap average of Karplus–Strong ties the treble decay to
  the pitch and strips a high note to a sine within a second;
- **stiffness** — first-order allpasses in the loop stretch the partials roughly as
  f_h = h f0 sqrt(1 + B h^2) (fitted near 1.5 kHz), so the waveform keeps changing shape
  instead of repeating;
- **polarisations** — a second, slightly detuned and longer-ringing string, mixed in
  quietly: the slow beating of the partials that every real string has. Given
  `prompt_t60`, the picked polarisation dies fast (the "prompt sound") while the other
  carries the long "aftersound": the two-stage decay of a real pluck instead of an
  organ-like steady tone.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from notes.engine.dsp import dc_block, feedback_delay, filt

__all__ = ["steel_string"]

_F_HIGH = 4000.0
_DISPERSION_STAGES = 2
_F_FIT = 1500.0


def _lowpass_delay(a: float, w: float) -> float:
    """Phase delay (samples) of the one-pole g(1+a)/(1+a z^-1) at angular frequency w."""
    return float(-np.arctan2(a * np.sin(w), 1.0 + a * np.cos(w)) / w)


def _allpass_delay(c: float, w: float) -> float:
    """Phase delay (samples) of the allpass (c + z^-1)/(1 + c z^-1), unwrapped."""
    return float(1.0 - 2.0 * np.arctan2(c * np.sin(w), 1.0 + c * np.cos(w)) / w)


def _loss_filter(f0: float, sr: int, t60: float, t60_high: float) -> tuple[float, float]:
    """(g, a) of the one-pole g(1+a)/(1+a z^-1) whose loss per period gives the two decay times."""
    w0, wh = 2 * np.pi * f0 / sr, 2 * np.pi * min(_F_HIGH, 0.45 * sr) / sr
    per_period = lambda t: 10.0 ** (-3.0 / (f0 * max(t, 1e-3)))  # noqa: E731
    mag = lambda a, w: (1 + a) / abs(1 + a * np.exp(-1j * w))  # noqa: E731
    ratio = per_period(min(t60_high, t60)) / per_period(t60)
    a = 0.0
    if wh > w0 and ratio < mag(0.0, wh) / mag(0.0, w0) - 1e-12:
        # A one-pole can only dip so far between f0 and 4 kHz; past that, take its steepest.
        steepest = mag(-0.9999, wh) / mag(-0.9999, w0)
        a = -0.9999 if ratio <= steepest else brentq(lambda x: mag(x, wh) / mag(x, w0) - ratio, -0.9999, 0.0)
    return per_period(t60) / mag(a, w0), a


def _fraction_coefficient(frac: float, w0: float) -> float:
    """First-order allpass coefficient with phase delay `frac` samples exactly at w0."""
    try:
        return brentq(lambda c: _allpass_delay(c, w0) - frac, -0.999, 0.999)
    except ValueError:
        return (1.0 - frac) / (1.0 + frac)


def _loop(f0: float, sr: int, t60: float, t60_high: float, stiffness: float):
    """Integer delay and loop filter (losses, dispersion, fine tuning) that resonate at f0.

    The partials of the loop sit where its total phase delay D(w) satisfies w D(w) = 2 pi h.
    With stiffness, the dispersion allpasses' coefficient is chosen so that the partial
    nearest 1.5 kHz lands at h f0 sqrt(1 + B h^2) (or as near as the loop gets: the low-pass
    and the fine-tuning allpass stretch a little on their own); the fractional allpass then
    puts h = 1 on f0.
    """
    period = sr / f0
    w0 = 2 * np.pi * f0 / sr
    g, a = _loss_filter(f0, sr, t60, t60_high)
    stages = _DISPERSION_STAGES if stiffness > 0 else 0

    def design(d: float):
        rest = period - _lowpass_delay(a, w0) - stages * _allpass_delay(d, w0)
        length = int(np.floor(rest - 0.1))
        return length, _fraction_coefficient(rest - length, w0) if length >= 2 else 0.0

    d = 0.0
    if stiffness > 0:
        h = max(2, int(_F_FIT / f0))

        def partial_error(d: float) -> float:
            length, c = design(d)
            if length < 2:
                return 1.0

            def phase(w: float) -> float:
                delay = length + _lowpass_delay(a, w) + stages * _allpass_delay(d, w)
                return w * (delay + _allpass_delay(c, w)) - 2 * np.pi * h

            wh = brentq(phase, 0.5 * h * w0, min(1.5 * h * w0, 0.999 * np.pi))
            return wh / (h * w0) - np.sqrt(1 + stiffness * h * h)

        try:
            d = brentq(partial_error, -0.95, 0.0)
        except ValueError:
            d = 0.0
    length, c = design(d)
    if length < 2:
        raise ValueError(f"{f0:.0f} Hz is too high for the string model at {sr} Hz")
    loop_b, loop_a = np.array([g * (1 + a)]), np.array([1.0, a])
    for coef in [d] * stages + [c]:
        loop_b, loop_a = np.convolve(loop_b, [coef, 1.0]), np.convolve(loop_a, [1.0, coef])
    return length, loop_b, loop_a


def _pluck(length: int, sr: int, pick: float, hardness: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    """One period of string velocity right after the pick lets go."""
    p = int(np.clip(round(pick * length), 1, length - 1))
    shape = np.where(np.arange(length) < p, 1.0 / p, -1.0 / (length - p))
    shape /= np.max(np.abs(shape))
    shape += noise * rng.uniform(-1.0, 1.0, length)
    # Round the corners: filter the periodic extension, keep one settled period.
    cutoff = min(1500.0 * 2.0 ** (3.0 * float(np.clip(hardness, 0.0, 1.0))), 0.45 * sr)
    y = filt(np.tile(shape, 4), "lowpass", cutoff, sr, 0.6)[-length:]
    y -= y.mean()
    return y / max(float(np.max(np.abs(y))), 1e-9)


def steel_string(
    freq: float,
    n: int,
    sr: int,
    rng: np.random.Generator,
    *,
    t60: float = 6.0,
    t60_high: float = 1.0,
    pick: float = 0.15,
    hardness: float = 0.6,
    noise: float = 0.1,
    stiffness: float = 4e-5,
    detune: float = 0.7,
    second: float = 0.3,
    prompt_t60: float | None = None,
) -> np.ndarray:
    """n samples of a string plucked at fraction `pick` of its length from the bridge.

    `t60` / `t60_high`: decay (s) of the fundamental and of partials near 4 kHz;
    `hardness` 0..1 rounds the pluck from a soft thumb to a hard pick; `noise` is the share
    of noise in the pluck; `stiffness` is the inharmonicity B; the second polarisation is
    `detune` cents sharp, rings 1.4x longer and is mixed at `second`. With `prompt_t60`
    the first polarisation decays in `prompt_t60` instead (its treble no slower) and the
    second one in `t60`.
    """
    excitation = np.zeros(n)
    out = np.zeros(n)
    detuned = freq * 2.0 ** (detune / 1200.0)
    if prompt_t60 is None:
        strings = ((1.0, freq, t60, t60_high), (second, detuned, 1.4 * t60, t60_high))
    else:
        prompt = min(prompt_t60, t60)
        strings = ((1.0, freq, prompt, min(t60_high, prompt)), (second, detuned, t60, t60_high))
    for weight, f0, decay, decay_high in strings:
        if weight <= 0:
            continue
        length, b, a = _loop(f0, sr, decay, decay_high, stiffness)
        burst = _pluck(length, sr, pick, hardness, noise, rng)
        excitation[:] = 0.0
        excitation[: min(length, n)] = burst[: min(length, n)]
        out += weight * feedback_delay(excitation, length, 1.0, b, a)
    return dc_block(out, sr)
