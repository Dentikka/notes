"""Signal-processing primitives: oscillators, envelopes, filters and delay loops.

Everything is vectorised with numpy; recursions (resonant filters, comb and string loops)
run through ``scipy.signal.lfilter``. A feedback loop with a long delay is computed in
blocks no longer than the delay — every sample it feeds back is already known — which is
exact and avoids the cost of a dense high-order filter.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import lfilter

__all__ = [
    "TAU",
    "adsr",
    "biquad",
    "db_to_gain",
    "dc_block",
    "decay",
    "delay",
    "fade",
    "feedback_delay",
    "filt",
    "gain_to_db",
    "karplus_strong",
    "lowpass_sweep",
    "osc",
    "phase",
    "stable_seed",
    "vel_gain",
]

TAU = 2.0 * np.pi
_DENSE_DELAY_MAX = 128


def stable_seed(*parts: object) -> int:
    """A seed that is identical across runs and machines (unlike Python's salted ``hash``)."""
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def gain_to_db(gain: float, floor: float = -120.0) -> float:
    return float(20.0 * np.log10(gain)) if gain > 0 else floor


def vel_gain(vel: float) -> float:
    """Velocity to amplitude: 0.8 -> -2.9 dB, 0.5 -> -9 dB."""
    return float(max(vel, 0.0)) ** 1.5


# --- oscillators ------------------------------------------------------------------------
def phase(freq: float | np.ndarray, n: int, sr: int, start: float = 0.0) -> np.ndarray:
    """Normalised phase in [0, 1) of an oscillator with a constant or per-sample frequency."""
    if np.ndim(freq) == 0:
        ph = start + np.arange(n) * (float(freq) / sr)
    else:
        inc = np.asarray(freq, dtype=float)[:n] / sr
        ph = start + np.concatenate(([0.0], np.cumsum(inc[:-1])))
    return np.mod(ph, 1.0)


def _blep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Polynomial band-limited step residual (PolyBLEP) around phase wraps."""
    out = np.zeros_like(t)
    lo = t < dt
    x = t[lo] / dt[lo]
    out[lo] = x + x - x * x - 1.0
    hi = t > 1.0 - dt
    x = (t[hi] - 1.0) / dt[hi]
    out[hi] = x * x + x + x + 1.0
    return out


def osc(
    wave: str, freq: float | np.ndarray, n: int, sr: int, start: float = 0.0, width: float = 0.5
) -> np.ndarray:
    """Band-limited oscillator: 'sine', 'triangle', 'saw', 'square' / 'pulse' (duty `width`)."""
    ph = phase(freq, n, sr, start)
    if wave == "sine":
        return np.sin(TAU * ph)
    if wave == "triangle":
        return 1.0 - 4.0 * np.abs(ph - 0.5)
    dt = np.broadcast_to(np.asarray(freq, dtype=float) / sr, (n,)) if np.ndim(freq) == 0 else freq[:n] / sr
    if wave == "saw":
        return 2.0 * ph - 1.0 - _blep(ph, dt)
    if wave in ("square", "pulse"):
        naive = np.where(ph < width, 1.0, -1.0)
        return naive + _blep(ph, dt) - _blep(np.mod(ph - width, 1.0), dt)
    raise ValueError(f"Unknown wave {wave!r}: use sine, triangle, saw, square or pulse")


# --- envelopes --------------------------------------------------------------------------
def _held(t: np.ndarray, attack: float, dec: float, sustain: float) -> np.ndarray:
    return np.where(t < attack, t / attack, sustain + (1.0 - sustain) * np.exp(-(t - attack) * 4.0 / dec))


def adsr(
    gate: int, total: int, sr: int, attack: float, dec: float, sustain: float, release: float
) -> np.ndarray:
    """ADSR over `total` samples with the key held for `gate` samples.

    Linear attack, exponential decay (98% settled after `dec`), and a release that reaches
    exactly zero after `release` seconds from whatever level the gate closed at.
    """
    a, d, r = max(attack, 1e-3), max(dec, 1e-3), max(release, 1e-3)
    t = np.arange(total) / sr
    g = gate / sr
    level = float(_held(np.array([g]), a, d, sustain)[0]) if gate > 0 else 0.0
    x = np.clip((t - g) / r, 0.0, 1.0)
    rel = level * (np.exp(-5.0 * x) - np.exp(-5.0)) / (1.0 - np.exp(-5.0))
    return np.where(t < g, _held(t, a, d, sustain), rel)


def decay(n: int, sr: int, tau: float) -> np.ndarray:
    """Exponential decay with time constant `tau` seconds."""
    return np.exp(-np.arange(n) / (sr * max(tau, 1e-5)))


def fade(x: np.ndarray, sr: int, fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
    """Linear fades at the ends of the last axis (returns a copy)."""
    y = np.array(x, dtype=float)
    n = y.shape[-1]
    if (k := min(n, int(fade_in * sr))) > 0:
        y[..., :k] *= np.linspace(0.0, 1.0, k)
    if (k := min(n, int(fade_out * sr))) > 0:
        y[..., n - k :] *= np.linspace(1.0, 0.0, k)
    return y


# --- filters ----------------------------------------------------------------------------
def biquad(kind: str, freq: float, sr: int, q: float = 0.7071, gain_db: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """RBJ-cookbook biquad (b, a), normalised to a[0] = 1.

    kind: lowpass, highpass, bandpass, notch, peak, lowshelf, highshelf.
    """
    f = min(max(float(freq), 10.0), 0.45 * sr)
    w = TAU * f / sr
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2.0 * max(q, 1e-3))
    amp = 10.0 ** (gain_db / 40.0)
    if kind == "lowpass":
        b, a = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2], [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b, a = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2], [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b, a = [alpha, 0.0, -alpha], [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b, a = [1.0, -2 * cw, 1.0], [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b, a = [1 + alpha * amp, -2 * cw, 1 - alpha * amp], [1 + alpha / amp, -2 * cw, 1 - alpha / amp]
    elif kind in ("lowshelf", "highshelf"):
        s = 1.0 if kind == "lowshelf" else -1.0
        sa = 2.0 * np.sqrt(amp) * alpha
        b = [
            amp * ((amp + 1) - s * (amp - 1) * cw + sa),
            2 * s * amp * ((amp - 1) - s * (amp + 1) * cw),
            amp * ((amp + 1) - s * (amp - 1) * cw - sa),
        ]
        a = [
            (amp + 1) + s * (amp - 1) * cw + sa,
            -2 * s * ((amp - 1) + s * (amp + 1) * cw),
            (amp + 1) + s * (amp - 1) * cw - sa,
        ]
    else:
        raise ValueError(f"Unknown filter kind {kind!r}")
    b_arr, a_arr = np.asarray(b, dtype=float), np.asarray(a, dtype=float)
    return b_arr / a_arr[0], a_arr / a_arr[0]


def filt(x: np.ndarray, kind: str, freq: float, sr: int, q: float = 0.7071, gain_db: float = 0.0) -> np.ndarray:
    """Apply a biquad along the last axis."""
    b, a = biquad(kind, freq, sr, q, gain_db)
    return lfilter(b, a, x, axis=-1)


def dc_block(x: np.ndarray, sr: int, freq: float = 15.0) -> np.ndarray:
    pole = float(np.exp(-TAU * freq / sr))
    return lfilter([1.0, -1.0], [1.0, -pole], x, axis=-1)


def lowpass_sweep(
    x: np.ndarray, cutoff: np.ndarray, sr: int, q: float = 0.7071, poles: int = 4, block: int = 64
) -> np.ndarray:
    """Resonant lowpass whose cutoff (Hz, per sample) is updated every `block` samples.

    With 4 poles the first stage is Butterworth-damped and the resonance `q` sits on the second.
    """
    qs = [q] if poles <= 2 else [0.5412, q]
    states = [np.zeros(2) for _ in qs]
    out = np.empty(len(x))
    for i in range(0, len(x), block):
        seg = x[i : i + block]
        fc = float(cutoff[i])
        for s, stage_q in enumerate(qs):
            b, a = biquad("lowpass", fc, sr, stage_q)
            seg, states[s] = lfilter(b, a, seg, zi=states[s])
        out[i : i + block] = seg
    return out


# --- delay loops ------------------------------------------------------------------------
def delay(x: np.ndarray, samples: int) -> np.ndarray:
    """Pure delay along the last axis, same length (the end falls off)."""
    if samples <= 0:
        return np.array(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    if samples < x.shape[-1]:
        y[..., samples:] = x[..., :-samples]
    return y


def feedback_delay(
    x: np.ndarray, samples: int, gain: float, b: np.ndarray | None = None, a: np.ndarray | None = None
) -> np.ndarray:
    """y[n] = x[n] + gain * (H y)[n - samples], H = b/a (identity when omitted); 1-D input."""
    if samples < 1:
        raise ValueError("A feedback delay needs at least one sample")
    x = np.asarray(x, dtype=float)
    bb = np.array([1.0]) if b is None else np.asarray(b, dtype=float)
    aa = np.array([1.0]) if a is None else np.asarray(a, dtype=float)
    if samples <= _DENSE_DELAY_MAX:
        den = np.zeros(samples + max(len(aa), len(bb)))
        den[: len(aa)] += aa
        den[samples : samples + len(bb)] -= gain * bb
        return lfilter(aa, den, x)
    y = np.zeros_like(x)
    zi = np.zeros(max(len(aa), len(bb)) - 1)
    for start in range(0, len(x), samples):
        end = min(start + samples, len(x))
        src = y[start - samples : end - samples] if start else np.zeros(end - start)
        if zi.size:
            src, zi = lfilter(bb, aa, src, zi=zi)
        elif bb[0] != 1.0:
            src = bb[0] * src
        y[start:end] = x[start:end] + gain * src
    return y


def karplus_strong(
    freq: float,
    n: int,
    sr: int,
    rng: np.random.Generator,
    *,
    t60: float = 3.0,
    brightness: float = 0.6,
    pick: float = 0.2,
) -> np.ndarray:
    """Plucked string (extended Karplus–Strong) tuned exactly with a fractional-delay allpass.

    Loop: delay L, two-tap average (half a sample), first-order allpass for the remaining
    fraction, and a gain that sets the decay to `t60` seconds. The excitation is a noise
    burst, low-passed by `brightness` and comb-filtered by the pick position `pick`.
    """
    period = sr / freq
    target = period - 0.5
    length = int(np.floor(target))
    frac = target - length
    if frac < 0.1:
        length, frac = length - 1, frac + 1.0
    if length < 2:
        raise ValueError(f"{freq:.0f} Hz is too high for the string model at {sr} Hz")
    c = (1.0 - frac) / (1.0 + frac)
    rho = 10.0 ** (-3.0 / (freq * max(t60, 1e-3)))
    b = rho * np.convolve([0.5, 0.5], [c, 1.0])
    a = np.array([1.0, c])
    burst = rng.uniform(-1.0, 1.0, length)
    burst = filt(burst, "lowpass", 400.0 * 2.0 ** (5.0 * float(np.clip(brightness, 0.0, 1.0))), sr)
    p = int(round(pick * length))
    if 0 < p < length:
        burst = burst - delay(burst, p)
    burst -= burst.mean()
    burst /= max(float(np.max(np.abs(burst))), 1e-9)
    excitation = np.zeros(n)
    excitation[: min(length, n)] = burst[: min(length, n)]
    return dc_block(feedback_delay(excitation, length, 1.0, b, a), sr)
