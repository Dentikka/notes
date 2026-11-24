"""Estimate a coarse magnitude profile; recordings never become render carriers."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from notes.ears.note_profile import profile_note
from notes.engine.spectral import SpectralProfile


def spectral_profile(audio: np.ndarray, sr: int, expected_hz: float, *, pitch: float,
                     harmonics: int = 32, normalize_rms: float | None = .02) -> SpectralProfile:
    """Analyze one known steady mono note into <=16 magnitude knots per component.

    Complex demodulation estimates magnitudes and residual noise energy only;
    neither measured phases nor residual waveform leave this function. Default
    normalization uses RMS .1-.3s; None retains input amplitude. The four broad
    noise bands are approximate, not a calibrated model of an instrument body.
    No automatic onset correction or cross-note/velocity transfer is inferred.
    """
    if np.iscomplexobj(audio):
        raise ValueError('spectral analysis requires real mono audio')
    x = np.asarray(audio, dtype=float)
    if (x.ndim != 1 or not np.isfinite(x).all() or not isinstance(sr, (int, np.integer))
            or isinstance(sr, bool) or sr < 8000 or len(x) < .4*sr
            or not np.isfinite([expected_hz, pitch]).all() or not 20 <= expected_hz < sr/4
            or not isinstance(harmonics, int) or isinstance(harmonics, bool) or not 1 <= harmonics <= 32):
        raise ValueError('spectral analysis requires >=.4s finite mono audio, valid pitch/rate and 1..32 harmonics')
    level = float(np.sqrt(np.mean(x[round(.1*sr):round(.3*sr)]**2)))
    if level < 1e-8:
        raise ValueError('spectral analysis requires an audible .1-.3s anchor')
    if normalize_rms is not None:
        if not np.isfinite(normalize_rms) or normalize_rms <= 0:
            raise ValueError('normalize_rms must be positive or None')
        x = x*(normalize_rms/level)
    measured = profile_note(x, sr, expected_hz, harmonics=harmonics, partial_mode='tracked', analysis_window=.12)
    if not measured.frequency_supported:
        raise ValueError('spectral analysis could not support the intended fundamental')
    frequencies = np.unique(np.r_[measured.f0, measured.partial_hz[1:]])
    end = len(x)/sr-.05
    knots = np.array([0., .008, .02, .05, .1, .2, .4, .8, 1.6, 3.2, 5., 7., 9., 11., 13.])
    knots = np.r_[knots[knots < end], end]

    def values(signal: np.ndarray) -> tuple[float, ...]:
        result = []
        for time in knots:
            lo = max(0, round((time-.012)*sr))
            hi = min(len(signal), max(lo+1, round((time+.012)*sr)))
            result.append(float(np.median(signal[lo:hi])))
        result[0] = 0.
        return tuple(result)

    lowpass = butter(4, min(25., .15*measured.f0), fs=sr, output='sos')
    t = np.arange(len(x))/sr
    reconstruction = np.zeros(len(x))
    amplitudes = []
    for frequency in frequencies:
        carrier = np.exp(2j*np.pi*frequency*t)
        envelope = sosfiltfilt(lowpass, x*carrier.conj())
        reconstruction += 2*(envelope*carrier).real
        amplitudes.append(values(2*abs(envelope)))
    residual = x-reconstruction
    bands, noise = [], []
    for low, upper in ((40., 250.), (250., 1000.), (1000., 4000.), (4000., 9000.)):
        high = min(upper, .45*sr)
        if low >= high:
            continue
        band = sosfiltfilt(butter(2, [low, high], fs=sr, btype='bandpass', output='sos'), residual)
        power = sosfiltfilt(butter(2, 20., fs=sr, output='sos'), band*band)
        noise.append(values(np.sqrt(np.maximum(power, 0.))))
        bands.append((low, high))
    return SpectralProfile(pitch=pitch, reference_hz=measured.f0,
                           ratios=tuple(frequencies/measured.f0), times=tuple(knots),
                           amplitudes=tuple(amplitudes), noise_rms=tuple(noise), noise_bands=tuple(bands))
