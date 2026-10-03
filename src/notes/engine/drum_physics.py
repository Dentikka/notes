"""Building blocks for acoustic drums: struck membranes, metal plates and stick impacts.

Modal synthesis: a struck body rings as a sum of exponentially decaying sinusoids, one per
vibration mode. What makes it sound acoustic rather than electronic is which modes, how
they decay, and how they change with the strength of the hit:

- **membranes** (kick, snare, toms): the modes of an ideal circular membrane, at the
  zeros of the Bessel functions (1, 1.59, 2.14, 2.30, ...), slightly scattered; a hard hit
  stretches the head and raises its pitch, which falls back within tens of milliseconds;
  higher modes die faster;
- **metal** (cymbals, hi-hats): thousands of inharmonic modes, so dense at the top that
  they merge into noise. That body is noise whose spectrum evolves (`wash`): each band
  decays at its own rate, and the upper bands take tens of milliseconds to fill in (energy
  cascades upward through the plate's nonlinearity). A few hundred explicit modes on top
  (`metal`) give the shimmer of close neighbours beating; alone they sound like a
  synthesiser's metal (the 808's six square waves are the extreme case);
- **impacts**: the stick or beater itself, a ~1 ms broadband click and a softer thump.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

from notes.engine.dsp import filt

__all__ = ["MEMBRANE_RATIOS", "impact", "membrane", "metal", "modes", "wash"]

#: Mode frequencies of an ideal circular membrane relative to the fundamental (j_mn / j_01).
MEMBRANE_RATIOS = (1.000, 1.594, 2.136, 2.296, 2.653, 2.918, 3.156, 3.501, 3.600, 3.652, 4.060, 4.154)
_CHUNK = 48  # modes per vectorised block: bounds memory for long cymbals


def modes(
    t: np.ndarray,
    freqs: np.ndarray,
    amps: np.ndarray,
    taus: np.ndarray,
    rng: np.random.Generator,
    glide: float = 0.0,
    glide_tau: float = 0.03,
    bloom: np.ndarray | None = None,
) -> np.ndarray:
    """Sum of decaying sinusoids a e^{-t/tau} sin(phase), random start phases.

    `glide`: every frequency starts (1 + glide) times higher and falls back with time
    constant `glide_tau` (tension modulation of a hard hit). `bloom`: per-mode rise time (s),
    the mode's amplitude grows as 1 - e^{-t/bloom}.
    """
    y = np.zeros(t.size)
    warp = t + glide * glide_tau * (1.0 - np.exp(-t / glide_tau)) if glide else t
    for i in range(0, len(freqs), _CHUNK):
        f = freqs[i : i + _CHUNK, None]
        a = amps[i : i + _CHUNK, None]
        tau = taus[i : i + _CHUNK, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, (f.shape[0], 1))
        env = a * np.exp(-t[None, :] / tau)
        if bloom is not None:
            env = env * (1.0 - np.exp(-t[None, :] / np.maximum(bloom[i : i + _CHUNK, None], 1e-5)))
        y += np.sum(env * np.sin(2.0 * np.pi * f * warp[None, :] + phase), axis=0)
    return y


def membrane(
    t: np.ndarray,
    f0: float,
    tau: float,
    rng: np.random.Generator,
    *,
    strike: float = 0.3,
    glide: float = 0.0,
    n_modes: int = 10,
    damping: float = 1.6,
) -> np.ndarray:
    """A struck drum head. `strike` 0 = dead centre (only the round modes), 1 = near the
    edge (the off-centre modes too); `damping` > 1 makes higher modes decay faster
    (tau_k = tau / ratio^damping)."""
    ratios = np.array(MEMBRANE_RATIOS[:n_modes]) * (1.0 + rng.normal(0.0, 0.01, n_modes))
    axisymmetric = np.isin(np.arange(n_modes), (0, 3, 8))  # the (0, n) modes
    amps = np.where(axisymmetric, 1.0, strike) / ratios
    amps *= 1.0 + rng.normal(0.0, 0.15, n_modes)
    return modes(t, f0 * ratios, amps, tau / ratios**damping, rng, glide=glide)


def metal(
    t: np.ndarray,
    rng: np.random.Generator,
    *,
    n_modes: int,
    low: float,
    high: float,
    tau_low: float,
    tau_high: float,
    tilt: float = 0.0,
    bloom: float = 0.0,
) -> np.ndarray:
    """A struck metal plate: `n_modes` inharmonic modes spread log-uniformly from `low` to
    `high` Hz, decaying from `tau_low` to `tau_high` across that range; amplitude tilted by
    `tilt` dB per octave; upper modes fill in over up to `bloom` seconds."""
    freqs = np.sort(np.exp(rng.uniform(np.log(low), np.log(high), n_modes)))
    octave = np.log2(freqs / low) / max(np.log2(high / low), 1e-9)  # 0 at low, 1 at high
    taus = tau_low * (tau_high / tau_low) ** octave
    amps = 10.0 ** (tilt * np.log2(freqs / low) / 20.0) * rng.uniform(0.3, 1.0, n_modes)
    rise = bloom * octave**1.5 if bloom else None
    return modes(t, freqs, amps, taus, rng, bloom=rise)


def wash(
    t: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    *,
    low: float,
    high: float,
    tau_low: float,
    tau_high: float,
    tilt: float = 0.0,
    bloom: float = 0.0,
) -> np.ndarray:
    """Noise shaped in time and frequency like a struck plate's dense modes: band-limited to
    `low`..`high` Hz (soft edges), each frequency decaying with its own time constant
    (log-interpolated from `tau_low` to `tau_high`), tilted `tilt` dB/octave, the top
    filling in over `bloom` seconds."""
    nper = 1024
    _, _, z = stft(rng.standard_normal(t.size + nper), fs=sr, nperseg=nper, noverlap=3 * nper // 4)
    freqs = np.fft.rfftfreq(nper, 1.0 / sr)[:, None]
    times = np.arange(z.shape[1])[None, :] * (nper // 4) / sr
    octave = np.clip(np.log2(np.maximum(freqs, 1.0) / low) / max(np.log2(high / low), 1e-9), 0.0, 1.0)
    band = 1.0 / (1.0 + (low / np.maximum(freqs, 1.0)) ** 4) / (1.0 + (freqs / high) ** 6)
    amp = band * 10.0 ** (tilt * np.log2(np.maximum(freqs, low) / low) / 20.0)
    tau = tau_low * (tau_high / tau_low) ** octave
    env = amp * np.exp(-times / tau)
    if bloom:
        env = env * (1.0 - np.exp(-times / np.maximum(bloom * octave**1.5, 1e-4)))
    _, y = istft(z * env, fs=sr, nperseg=nper, noverlap=3 * nper // 4)
    return y[: t.size]


def impact(n: int, sr: int, rng: np.random.Generator, *, hardness: float, length: float = 0.001) -> np.ndarray:
    """A stick or beater hitting: a click of `length` seconds whose brightness follows
    `hardness` (0 felt .. 1 wood on metal), plus a thump of twice the length."""
    k = max(2, int(length * sr))
    click = np.zeros(n)
    m = min(n, 4 * k)
    noise = rng.uniform(-1.0, 1.0, m) * np.exp(-np.arange(m) / k)
    cutoff = min(1500.0 * 2.0 ** (3.5 * hardness), 0.45 * sr)
    click[:m] = filt(noise, "lowpass", cutoff, sr, 0.7)
    j = min(n, 2 * k)
    click[:j] += 0.6 * np.sin(np.pi * np.arange(j) / j)  # the thump: a half-sine pulse
    return click
