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

from notes.engine.articulation import _partial_gains
from notes.engine.dsp import dc_block, feedback_delay, filt
from notes.engine.waveguide_modes import decay_correction, decay_scales

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


def _loop(f0: float, sr: int, t60: float, t60_high: float, stiffness: float,
          dispersion_fit: str = "legacy"):
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

            if dispersion_fit == "bracketed":
                # Trial dispersion coefficients can move this mode far outside
                # 0.5..1.5 times its nominal frequency. A failed inner bracket
                # used to abort the outer coefficient fit and disable stiffness.
                wh = brentq(phase, w0, 0.999 * np.pi)
            else:
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


def _spectral_shape(length: int, pick: float, excitation: str) -> np.ndarray:
    """Finite Fourier series of a continuous triangular displacement or its derivative."""
    if length < 3:
        raise ValueError("spectral pluck requires at least three delay samples")
    position = float(np.clip(pick, 1/length, 1-1/length))
    h = np.arange(1, length//2+1, dtype=float)
    spectrum = np.zeros(length//2+1, dtype=complex)
    jump = np.expm1(-2j*np.pi*h*position)/(position*(1-position))
    if excitation == "velocity":
        spectrum[1:] = -length*jump/(2j*np.pi*h)
    else:
        spectrum[0] = length*.5
        spectrum[1:] = length*jump/(4*np.pi**2*h*h)
    return np.fft.irfft(spectrum, n=length)


def _pluck(length: int, sr: int, pick: float, hardness: float, noise: float, rng: np.random.Generator,
           excitation: str = "velocity", sampling: str = "grid") -> np.ndarray:
    """One band-limited period of the selected string variable after release."""
    p = int(np.clip(round(pick * length), 1, length - 1))
    if sampling == "spectral":
        shape = _spectral_shape(length, pick, excitation)
    elif excitation == "displacement":
        # A triangular initial displacement, rounded by the same band-limit filter.
        position = np.arange(length)
        shape = np.where(position < p, position / p, (length-position) / (length-p))
    else:
        shape = np.where(np.arange(length) < p, 1.0 / p, -1.0 / (length - p))
    shape /= np.max(np.abs(shape))
    shape += noise * rng.uniform(-1.0, 1.0, length)
    # Round the corners: filter the periodic extension, keep one settled period.
    cutoff = min(1500.0 * 2.0 ** (3.0 * float(np.clip(hardness, 0.0, 1.0))), 0.45 * sr)
    y = filt(np.tile(shape, 4), "lowpass", cutoff, sr, 0.6)[-length:]
    y -= y.mean()
    return y / max(float(np.max(np.abs(y))), 1e-9)


def _shape_excitation(burst: np.ndarray, gains: tuple[float, ...]) -> np.ndarray:
    count = min(len(gains), len(burst)//2)
    if not count or all(gain == 1 for gain in gains[:count]):
        return burst
    spectrum = np.fft.rfft(burst)
    spectrum[1:count+1] *= gains[:count]
    return np.fft.irfft(spectrum, n=len(burst))


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
    excitation: str = "velocity",
    dispersion_fit: str = "legacy",
    excitation_gains: tuple[float, ...] = (),
    pluck_sampling: str = "grid",
    partial_t60_scales: tuple[float, ...] = (),
) -> np.ndarray:
    """n samples of a string plucked at fraction `pick` of its length from the bridge.

    `excitation` selects the original velocity readout or a triangular-displacement
    variant; both are designed mathematically without recorded excitation.
    `t60` / `t60_high`: decay (s) of the fundamental and of partials near 4 kHz;
    `hardness` 0..1 rounds the pluck from a soft thumb to a hard pick; `noise` is the share
    of noise in the pluck; `stiffness` is the inharmonicity B; the second polarisation is
    `detune` cents sharp, rings 1.4x longer and is mixed at `second`. With `prompt_t60`
    the first polarisation decays in `prompt_t60` instead (its treble no slower) and the
    second one in `t60`.

    `dispersion_fit="bracketed"` expands the inner resonance search so extreme
    trial coefficients do not silently disable the stiffness fit. It retains
    the two-allpass approximation and its 1.5 kHz reference, not an exact modal
    string. "legacy" preserves the original waveform.

    ``excitation_gains`` optionally scales successive non-DC DFT bins of each
    mathematically generated excitation period before it enters the loop.
    Values are in 0..16; omitted bins remain unchanged and unavailable bins
    are ignored. This preserves generated bin phases, not recorded phases.
    No peak renormalization follows the shaping. Dispersion/fractional delays
    mean these are excitation-bin controls, not exact resonant-partial gains.
    Empty or unity controls retain the original samples and RNG stream exactly.

    ``pluck_sampling="spectral"`` evaluates a finite Fourier series at a continuous
    pick coordinate, avoiding integer-grid notches and jumps as the position moves.
    It requires at least three delay samples and retains pick softening by the
    lowpass and the existing peak normalization;
    it does not identify a real pick position or certify a realistic timbre.
    "grid" preserves the original excitation and random-number stream.

    ``partial_t60_scales`` optionally multiplies individual oscillatory modes'
    native T60s after the generated excitation period ends. Positive factors
    retain frequencies and the mathematical modal state at that point; both
    polarisations keep their original lifetime ratios. The excitation period
    itself is unchanged before subsequent pickup/filter/pitch readout.
    Empty/unity factors preserve the original signal/RNG exactly. The optional
    modal path requires a contractive loop. Times refer to the source clock,
    before any subsequent pitch readout or note-release envelope.
    """
    if excitation not in ("velocity", "displacement"):
        raise ValueError("excitation must be velocity or displacement")
    if dispersion_fit not in ("legacy", "bracketed"):
        raise ValueError("dispersion_fit must be legacy or bracketed")
    if pluck_sampling not in ("grid", "spectral"):
        raise ValueError("pluck_sampling must be grid or spectral")
    gains = _partial_gains(excitation_gains, name='excitation_gains')
    scales = decay_scales(partial_t60_scales)
    impulse = np.zeros(n)
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
        length, b, a = (_loop(f0, sr, decay, decay_high, stiffness) if dispersion_fit == "legacy"
                        else _loop(f0, sr, decay, decay_high, stiffness, dispersion_fit))
        burst = (_pluck(length, sr, pick, hardness, noise, rng, excitation) if pluck_sampling == "grid"
                 else _pluck(length, sr, pick, hardness, noise, rng, excitation, sampling=pluck_sampling))
        burst = _shape_excitation(burst, gains)
        impulse[:] = 0.0
        impulse[: min(length, n)] = burst[: min(length, n)]
        wave = feedback_delay(impulse, length, 1.0, b, a)
        if scales and any(scale != 1 for scale in scales):
            wave += decay_correction(burst, length, b, a, scales, n)
        out += weight * wave
    return dc_block(out, sr)
