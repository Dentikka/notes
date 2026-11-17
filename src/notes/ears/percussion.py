"""Windowed dominant band peaks: a mode guard, not fundamental identification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class BandPeakTrajectory:
    windows: np.ndarray
    frequency_hz: np.ndarray
    contrast_db: np.ndarray
    relative_rms_db: np.ndarray
    supported: np.ndarray
    band_hz: tuple[float, float]
    sample_rate: int


def band_peak_trajectory(audio: np.ndarray, sr: int, *, band: tuple[float, float] = (60., 220.),
                         windows: tuple[tuple[float, float], ...] = ((.05, .15), (.15, .3), (.3, .5)),
                         minimum_contrast_db: float = 6., minimum_level_db: float = -45.) -> BandPeakTrajectory:
    """Measure the strongest peak in each specified absolute clip window.

    Supply onset-cropped finite mono audio covering every window. Each Hann
    spectrum is zero-padded and its peak log-parabolically interpolated; padding
    does not improve the physical window-limited resolution. Support requires
    an interior band maximum, contrast over the band's median power and enough
    RMS relative to the clip's strongest20ms region. It is a heuristic, not a
    coloured-noise classifier or proof that the same physical mode was followed.
    Peaks may move with tension or swap dominance. No carrier phase is returned.
    """
    x = np.asarray(audio, dtype=float)
    bounds = np.asarray(band, dtype=float)
    spans = np.asarray(windows, dtype=float)
    if (not isinstance(sr, int) or sr < 1000 or x.ndim != 1 or not len(x)
            or not np.isfinite(x).all()):
        raise ValueError('peak trajectory needs finite mono audio and an integer rate >=1000')
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] < bounds[1] < sr/2):
        raise ValueError('peak band must lie inside (0, Nyquist)')
    if (spans.ndim != 2 or spans.shape[1] != 2 or not 1 <= len(spans) <= 512
            or not np.isfinite(spans).all() or np.any(spans[:, 0] < 0)
            or np.any(spans[:, 1] <= spans[:, 0]) or np.any(np.diff(spans[:, 0]) <= 0)
            or np.any(spans[:, 1] > len(x)/sr) or np.any((spans[:, 1]-spans[:, 0])*sr < 16)):
        raise ValueError('peak windows need increasing starts, >=16 samples and complete audio coverage')
    if (not np.isfinite([minimum_contrast_db, minimum_level_db]).all()
            or minimum_contrast_db < 0 or minimum_level_db > 0):
        raise ValueError('peak support needs nonnegative contrast and nonpositive relative level')
    scale = float(np.max(abs(x)))
    if scale:
        x = x/scale  # Relative analysis: avoid squaring extremely large/small gains.
    size = min(len(x), max(1, round(.02*sr)))
    peak_rms = float(np.sqrt(np.max(np.convolve(x*x, np.ones(size)/size, mode='valid'))))
    frequencies, contrasts, levels, support = [], [], [], []
    for start, end in spans:
        y = x[round(start*sr):round(end*sr)]
        nfft = 1 << (max(32768, 4*len(y))-1).bit_length()
        hz, power = welch(y, sr, nperseg=len(y), nfft=nfft)
        indices = np.flatnonzero((hz >= bounds[0]) & (hz <= bounds[1]))
        if len(indices) < 3:
            raise ValueError('peak band needs at least three frequency bins')
        k = int(indices[np.argmax(power[indices])])
        a, b, c = np.log(np.maximum(power[k-1:k+2], 1e-30))
        denominator = a-2*b+c
        offset = float(np.clip(.5*(a-c)/denominator, -.5, .5)) if denominator < 0 else 0.
        frequency = float(np.clip((k+offset)*sr/nfft, *bounds))
        contrast = float(10*np.log10(max(power[k], 1e-30)/max(float(np.median(power[indices])), 1e-30)))
        level = (float(20*np.log10(max(float(np.sqrt(np.mean(y*y))), 1e-20)/peak_rms))
                 if peak_rms > 0 else -400.)
        frequencies.append(frequency)
        contrasts.append(contrast)
        levels.append(level)
        support.append(bool(indices[0] < k < indices[-1] and contrast >= minimum_contrast_db
                            and level >= minimum_level_db))
    return BandPeakTrajectory(spans, np.array(frequencies), np.array(contrasts), np.array(levels),
                              np.array(support), tuple(float(v) for v in bounds), sr)


def compare_peak_trajectories(reference: BandPeakTrajectory, candidate: BandPeakTrajectory, *,
                              maximum_relative_error: float = .1, minimum_windows: int = 2) -> dict[str, object]:
    """Require candidate support and a frequency match in every supported target window.

    Compare descriptors from the same window clock/band. Different sample rates
    are allowed. A passing guard says nothing about extra modes, timbre, room,
    attack or whether the target peak is the membrane's fundamental.
    """
    if (not np.isfinite(maximum_relative_error) or maximum_relative_error < 0
            or not isinstance(minimum_windows, int) or minimum_windows < 1):
        raise ValueError('peak comparison needs a nonnegative tolerance and positive window count')
    if reference.band_hz != candidate.band_hz or not np.array_equal(reference.windows, candidate.windows):
        raise ValueError('peak comparison needs the same band and window clock')
    n = len(reference.windows)
    for profile in (reference, candidate):
        if (profile.frequency_hz.shape != (n,) or profile.supported.shape != (n,)
                or profile.supported.dtype != bool or not np.isfinite(profile.frequency_hz).all()
                or np.any(profile.frequency_hz <= 0)):
            raise ValueError('peak comparison needs valid frequency/support arrays')
    keep = reference.supported
    missing = int(np.sum(keep & ~candidate.supported))
    errors = abs(candidate.frequency_hz[keep]/reference.frequency_hz[keep]-1)
    worst = float(errors.max()) if len(errors) else None
    count = int(keep.sum())
    return {'passed': bool(count >= minimum_windows and missing == 0 and worst is not None
                           and worst <= maximum_relative_error),
            'reference_supported_windows': count, 'missing_candidate_windows': missing,
            'maximum_relative_error': worst, 'per_window_relative_error': errors.tolist()}
