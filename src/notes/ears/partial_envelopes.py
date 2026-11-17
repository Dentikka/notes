"""Local sinusoid amplitudes on an explicitly supplied frequency trajectory.

Unlike spectral-band energy, regression amplitudes exclude much of the nearby
noise. Their uncertainty model assumes independent white noise; coloured noise,
wrong trajectories and unresolved modes still require independent checks.
Recorded sinusoid phases are deliberately not returned.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PartialEnvelopes:
    time: np.ndarray
    ratios: np.ndarray
    amplitude: np.ndarray
    white_noise_floor: np.ndarray
    above_white_noise_floor: np.ndarray
    residual_rms: np.ndarray
    design_condition: np.ndarray
    fundamental_hz: np.ndarray
    window_seconds: float
    sample_rate: int


@dataclass(frozen=True)
class PartialGainFit:
    """Bounded static shape controls and the support used to estimate them."""

    gains: tuple[float, ...]
    used_frames: int
    supported_cells: int
    total_cells: int
    upper_clipped_cells: int


def fit_partial_gains(reference: PartialEnvelopes, candidate: PartialEnvelopes, *,
                      start: float = .2, end: float = 1.2) -> PartialGainFit:
    """Estimate static readout gains from corresponding modal magnitudes.

    Measure each signal on its own supported frequency trajectory/ratios using
    ``partial_envelopes`` first. Equal-length mode lists must correspond by index;
    their frequencies need not coincide. Frame clocks must match exactly.

    Each frame is normalized by harmonic L2 amplitude, so this fits shape rather
    than loudness. Unsupported reference magnitudes are bounded by four white
    coefficient floors or the candidate amplitude, whichever is smaller. This
    floor is not a general tone/noise classifier. Use at least five frames with
    three supported reference modes (or all modes when fewer than three exist).
    Median frame ratios are bounded to [0, 16] for ``ModalString.partial_gains``.
    No recorded phase/carrier is returned; conditional fit is not transfer proof.
    """
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError('partial gain window needs increasing nonnegative seconds')
    for profile in (reference, candidate):
        t, a, floor = profile.time, profile.amplitude, profile.white_noise_floor
        if (t.ndim != 1 or not len(t) or not np.isfinite(t).all() or t[0] < 0
                or np.any(np.diff(t) <= 0) or a.ndim != 2 or a.shape[0] != len(t)
                or not 1 <= a.shape[1] <= 80 or len(profile.ratios) != a.shape[1]
                or floor.shape != a.shape or profile.above_white_noise_floor.shape != a.shape
                or not np.isfinite(a).all() or not np.isfinite(floor).all()
                or np.any(a < 0) or np.any(floor < 0)):
            raise ValueError('partial gain fit needs valid finite nonnegative envelope arrays')
    if (reference.amplitude.shape != candidate.amplitude.shape
            or not np.array_equal(reference.time, candidate.time)):
        raise ValueError('partial gain fit needs corresponding modes and identical frame clocks')
    if end > reference.time[-1]:
        raise ValueError('partial gain window exceeds measured frames')
    target = np.where(reference.above_white_noise_floor, reference.amplitude,
                      np.minimum(candidate.amplitude, 4*reference.white_noise_floor))
    target_norm = np.linalg.norm(target, axis=1)
    source_norm = np.linalg.norm(candidate.amplitude, axis=1)
    active = ((reference.time >= start) & (reference.time <= end)
              & (reference.above_white_noise_floor.sum(axis=1) >= min(3, target.shape[1]))
              & (target_norm > 1e-20) & (source_norm > 1e-20))
    if active.sum() < 5:
        raise ValueError('partial gain fit needs at least five supported frames')
    a = target[active]/target_norm[active, None]
    b = candidate.amplitude[active]/source_norm[active, None]
    ratios = a/np.maximum(b, 1e-6)
    gains = np.median(np.clip(ratios, 0., 16.), axis=0)
    return PartialGainFit(tuple(float(x) for x in gains), int(active.sum()),
                          int(reference.above_white_noise_floor[active].sum()), int(ratios.size),
                          int(np.sum(ratios > 16)))


def partial_envelopes(audio: np.ndarray, sr: int, fundamental_hz: float | np.ndarray, *,
                      partials: int | Sequence[float] = 20, step: float = .04,
                      minimum_window: float = .08, cycles: float = 6.) -> PartialEnvelopes:
    """Fit local sine/cosine magnitudes on a known single-note trajectory.

    ``fundamental_hz`` is a scalar or one frequency per input sample. An integer
    ``partials`` requests that many harmonics (1..80), truncated below .45*sr;
    an explicit increasing sequence supplies modal frequency ratios and must
    fit that output band. Ratios can describe known string inharmonicity, but
    do not resolve unison strings without sufficient time/frequency separation.

    Hann-weighted least squares fits all modes jointly, plus a local DC term.
    Frame times are absolute clip seconds; edge half-windows are excluded.
    Windows cover at least ``cycles`` of the slowest supplied fundamental or
    ``minimum_window`` seconds. Amplitudes are peak sinusoid magnitudes, not RMS.
    The returned noise floor is the RMS coefficient magnitude under an iid white
    residual model. ``above_white_noise_floor`` means amplitude >4 times that
    estimate; it is not a general tone/noise classifier or realism verdict.
    Neither carrier phases nor a recording-dependent excitation are returned.
    """
    x = np.asarray(audio, dtype=float)
    f = np.asarray(fundamental_hz, dtype=float)
    if (x.ndim != 1 or not x.size or not np.isfinite(x).all()
            or not isinstance(sr, int) or isinstance(sr, bool) or sr < 1000
            or f.ndim > 1 or (f.ndim == 1 and f.shape != x.shape)
            or not np.isfinite(f).all() or np.any(f < 20) or np.any(f >= sr/4)):
        raise ValueError('finite mono audio, integer rate and a known scalar/per-sample frequency required')
    if (not np.isfinite([step, minimum_window, cycles]).all() or step <= 0
            or minimum_window < .01 or cycles < 3):
        raise ValueError('positive frame step, window >=.01 seconds and at least three cycles required')
    if isinstance(partials, int) and not isinstance(partials, bool):
        if not 1 <= partials <= 80:
            raise ValueError('partial count must lie from 1 to 80')
        ratios = np.arange(1, partials+1, dtype=float)
        ratios = ratios[ratios*float(f.max()) < sr*.45]
    else:
        ratios = np.asarray(partials, dtype=float)
        if (ratios.ndim != 1 or not 1 <= len(ratios) <= 80 or not np.isfinite(ratios).all()
                or ratios[0] <= 0 or np.any(np.diff(ratios) <= 0) or ratios[-1]*float(f.max()) >= sr*.45):
            raise ValueError('1..80 increasing positive modal ratios below .45*sr required')
    size = max(round(sr*minimum_window), round(cycles*sr/float(f.min())))
    size += 1-size % 2
    hop = max(1, round(sr*step))
    if len(x) < size+2*hop:
        raise ValueError('partial envelopes need at least three full analysis windows')
    centers = np.arange(size//2, len(x)-size//2, hop)
    weights = np.hanning(size)
    track = np.broadcast_to(f, x.shape)
    phase = np.concatenate(([0.], np.cumsum(track[:-1]/sr)))*2*np.pi
    amplitudes, floors, residuals, conditions = [], [], [], []
    count = len(ratios)
    for center in centers:
        region = slice(center-size//2, center+size//2+1)
        angles = phase[region, None]*ratios[None, :]
        design = np.column_stack((np.ones(size), np.sin(angles), np.cos(angles)))
        gram = design.T@(weights[:, None]*design)
        eigenvalues = np.linalg.eigvalsh(gram)
        condition = float(np.sqrt(eigenvalues[-1]/max(eigenvalues[0], 1e-30)))
        if condition > 1e4:
            raise ValueError('partial modes are unresolved in the selected analysis window')
        inverse = np.linalg.solve(gram, np.eye(2*count+1))
        coefficients = inverse@(design.T@(weights*x[region]))
        error = x[region]-design@coefficients
        rss = float(weights@(error*error))
        second = design.T@((weights*weights)[:, None]*design)
        degrees = max(1., float(weights.sum()-np.trace(inverse@second)))
        covariance = (rss/degrees)*(inverse@second@inverse)
        variance = np.maximum(0., np.diag(covariance))
        amplitudes.append(np.hypot(coefficients[1:count+1], coefficients[count+1:]))
        numerical_floor = 10*np.finfo(float).eps*condition*np.sqrt(float(weights@(x[region]**2))/weights.sum())
        floors.append(np.maximum(numerical_floor, np.sqrt(variance[1:count+1]+variance[count+1:])))
        residuals.append(np.sqrt(rss/weights.sum()))
        conditions.append(condition)
    amplitude, floor = np.asarray(amplitudes), np.asarray(floors)
    return PartialEnvelopes(centers/sr, ratios, amplitude, floor, amplitude > 4*floor,
                            np.asarray(residuals), np.asarray(conditions), track[centers], size/sr, sr)
