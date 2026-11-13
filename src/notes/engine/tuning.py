"""Phase-based tuning of a stable isolated note with a known nominal pitch.

This measures mean tuning inside an explicit window, not an unknown melody.
Reliability checks are diagnostic heuristics, not proof of instrument realism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class KnownNoteTuning:
    f0: float | None
    partial_f0: tuple[float | None, ...]
    relative_level_db: tuple[float, ...]
    phase_residual_rms: tuple[float | None, ...]
    harmonic_spread_cents: float | None
    reliable: bool


def known_note_tuning(audio: np.ndarray, sr: int, expected: float, *,
                      start: float = .15, end: float = .75, harmonics: int = 3,
                      harmonic_bandwidth: bool = False) -> KnownNoteTuning:
    """Mean fundamental from demodulated phase; prior must name the correct note.

    The analysis window needs 100 ms of audio on either side and at least four
    cycles. Inputs may be mono or channels-first, downmixed for measurement.
    Read ``reliable`` before using ``f0``: the fundamental must be present,
    stable in phase (RMS <= 0.35 rad) and within 50 cents of the prior; detected
    harmonics within 35 dB must agree within 15 cents. Vibrato, bends, noise and
    missing fundamentals may fail these checks. Existing YIN behavior is unchanged.

    ``harmonic_bandwidth=True`` gives the 50-cent prior enough demodulation
    bandwidth at high pitches and scales that bandwidth with each harmonic.
    The corner is capped below half a fundamental spacing to reject neighbors.
    False retains the original narrow-band estimator bit for bit. Neither mode
    turns a moving fundamental into a reliable constant-frequency reference.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim not in (1, 2) or not x.size or not np.isfinite(x).all():
        raise ValueError("tuning needs finite mono or channels-first audio")
    if not np.isfinite(sr) or sr < 1000 or int(sr) != sr or not 20 <= expected < sr/4:
        raise ValueError("invalid sample rate or nominal frequency")
    if not isinstance(harmonics, int) or not 1 <= harmonics <= 8:
        raise ValueError("harmonics must be an integer from 1 to 8")
    if not isinstance(harmonic_bandwidth, bool):
        raise ValueError("harmonic_bandwidth must be a boolean")
    x = np.atleast_2d(x).mean(axis=0)
    if (not np.isfinite(start) or not np.isfinite(end) or start < .1 or end > len(x)/sr-.1
            or end-start < max(.1, 4/expected)):
        raise ValueError("tuning window needs four cycles and 100 ms edge margins")
    if np.std(x) < 1e-8:
        raise ValueError("silent or DC-only tuning input")
    time = np.arange(len(x))/sr
    window = (time > start) & (time < end)
    cutoff = min(18., expected*.15)
    if harmonic_bandwidth:
        cutoff = min(expected*.15, max(18., expected*.05))
    lowpass = butter(4, cutoff, fs=sr, output="sos")
    estimates, residuals, levels = [], [], []
    for harmonic in range(1, harmonics+1):
        band = min(harmonic*cutoff, expected*.45) if harmonic_bandwidth else cutoff
        if expected*harmonic+band >= sr/2:
            break
        if harmonic_bandwidth:
            lowpass = butter(4, band, fs=sr, output="sos")
        carrier = sosfiltfilt(lowpass, x*np.exp(-2j*np.pi*expected*harmonic*time))
        amplitude = abs(carrier)
        peak = float(amplitude[window].max())
        levels.append(float(np.sqrt(np.mean(amplitude[window]**2))))
        use = window & (amplitude > peak*.1)
        if peak < 1e-12 or use.sum() < 4:
            estimates.append(None)
            residuals.append(None)
            continue
        phase = np.unwrap(np.angle(carrier))
        trend = np.polyfit(time[use], phase[use], 1, w=amplitude[use])
        estimates.append(float(expected+trend[0]/(2*np.pi*harmonic)))
        residuals.append(float(np.sqrt(np.mean((phase[use]-np.polyval(trend, time[use]))**2))))
    relative = 20*np.log10(np.maximum(levels, 1e-12)/max(max(levels), 1e-12))
    active = [value for value, level in zip(estimates, relative, strict=True) if value is not None and level >= -35]
    spread = float(1200*np.log2(max(active)/min(active))) if active and min(active) > 0 else None
    f0, phase_rms = estimates[0], residuals[0]
    reliable = (f0 is not None and f0 > 0 and phase_rms is not None and phase_rms <= .35
                and relative[0] >= -35 and spread is not None and spread <= 15
                and abs(1200*np.log2(f0/expected)) <= 50)
    return KnownNoteTuning(f0, tuple(estimates), tuple(float(v) for v in relative),
                          tuple(residuals), spread, bool(reliable))
