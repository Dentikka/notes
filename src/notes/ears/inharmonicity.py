"""Known-note plucked-string inharmonicity from simultaneous partial phases.

Audio is an analysis input only. Reliability is a diagnostic heuristic, not a
certificate of instrument realism or a material-property measurement in situ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class InharmonicPartial:
    number: int
    ratio: float | None
    relative_level_db: float
    prominence_db: float
    phase_residual_rms: float | None
    coverage: float
    accepted: bool


@dataclass(frozen=True)
class StringInharmonicity:
    f0: float | None
    stiffness: float | None
    shape_rms_cents: float | None
    partials: tuple[InharmonicPartial, ...]
    reliable: bool
    reason: str | None


def string_inharmonicity(audio: np.ndarray, sr: int, expected: float, *,
                         start: float = .15, end: float = .65,
                         harmonics: int = 12) -> StringInharmonicity:
    """Fit B in f_h/(h*f_1) = sqrt((1+B*h*h)/(1+B)) for a known mono note.

    The nominal pitch must be correct within 50 cents. Read ``reliable`` before
    using B: at least three stable higher partials must fit the shape within
    1 cent RMS and 2.5 cents maximum error. Missing fundamentals, noise, coupled
    modes, unisons or strong inharmonicity can fail. Slightly negative estimates
    are retained for diagnosis instead of silently clipping them to zero.

    Subtracting h times the fundamental phase before fitting avoids attributing
    a common pitch glide to bending stiffness. Partials are searched within a
    quarter of f0 from nominal, up to 8 kHz. This is a moderate-inharmonicity
    diagnostic, not an unknown-note transcriber or a general bowed-string model.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.std(x) < 1e-8:
        raise ValueError("inharmonicity needs finite non-silent mono audio")
    if not np.isfinite(sr) or sr < 1000 or int(sr) != sr or not np.isfinite(expected) or not 20 <= expected < sr/4:
        raise ValueError("invalid sample rate or nominal frequency")
    if not isinstance(harmonics, int) or not 3 <= harmonics <= 32:
        raise ValueError("harmonics must be an integer from 3 to 32")
    if (not np.isfinite(start) or not np.isfinite(end) or start < .1
            or end > len(x)/sr-.1 or end-start < .2):
        raise ValueError("inharmonicity window needs 200 ms and 100 ms edge margins")
    time = np.arange(len(x))/sr
    window = (time >= start) & (time < end)
    wave = x[window]*np.hanning(window.sum())
    size = 1 << (max(32768, len(wave)*16)-1).bit_length()
    spectrum = abs(np.fft.rfft(wave, size))
    frequencies = np.fft.rfftfreq(size, 1/sr)

    def peak(center: float, width: float, background_width: float | None = None) -> tuple[float, float]:
        indices = np.flatnonzero(abs(frequencies-center) < width)
        at = int(indices[np.argmax(spectrum[indices])])
        a, b, c = np.log(spectrum[at-1:at+2]+1e-30)
        denominator = a-2*b+c
        offset = .5*(a-c)/denominator if denominator < 0 else 0.
        background = indices if background_width is None else np.flatnonzero(abs(frequencies-center) < background_width)
        prominence = 20*np.log10((spectrum[at]+1e-30)/(np.median(spectrum[background])+1e-30))
        return float((at+np.clip(offset, -.5, .5))*sr/size), float(prominence)

    # At low notes the narrow tuning search can contain mostly the Hann main
    # lobe. Estimate background over a wider region that still excludes h=2.
    fundamental, prominence = peak(expected, expected*.05, expected*.25)
    if prominence <= 12:
        return StringInharmonicity(None, None, None, (), False, "missing_fundamental")
    lowpass = butter(4, min(50, fundamental*.2), fs=sr, output="sos")

    def phase(center: float) -> tuple[np.ndarray, np.ndarray]:
        carrier = sosfiltfilt(lowpass, x*np.exp(-2j*np.pi*center*time))
        return np.unwrap(np.angle(carrier))+2*np.pi*center*time, abs(carrier)

    first, first_amp = phase(fundamental)
    f0 = float(np.polyfit(time[window], first[window], 1)[0]/(2*np.pi))
    first_level = float(np.sqrt(np.mean(first_amp[window]**2)))
    if f0 <= 0 or abs(1200*np.log2(f0/expected)) > 50 or first_level < np.std(x)*10**(-45/20):
        return StringInharmonicity(f0, None, None, (), False, "missing_fundamental")
    partials = []
    for number in range(2, harmonics+1):
        if number*f0+f0*.25 > min(8000, sr*.45):
            break
        center, prominence = peak(number*f0, f0*.25)
        value, amplitude = phase(center)
        level = float(20*np.log10((np.sqrt(np.mean(amplitude[window]**2))+1e-20)/first_level))
        valid = window & (amplitude > amplitude[window].max()*.1) & (first_amp > first_amp[window].max()*.1)
        coverage = float(valid.sum()/window.sum())
        if valid.sum() < 4:
            partials.append(InharmonicPartial(number, None, level, prominence, None, coverage, False))
            continue
        difference = value-number*first
        trend = np.polyfit(time[valid], difference[valid], 1)
        first_trend = np.polyfit(time[valid], first[valid], 1)
        ratio = float(1+trend[0]/(number*first_trend[0]))
        residual = float(np.sqrt(np.mean((difference[valid]-np.polyval(trend, time[valid]))**2)))
        accepted = level > -45 and prominence > 12 and coverage > .8 and residual < .35
        partials.append(InharmonicPartial(number, ratio, level, prominence, residual, coverage, bool(accepted)))
    used = [p for p in partials if p.accepted and p.ratio is not None and p.ratio > 0]
    if len(used) < 3:
        return StringInharmonicity(f0, None, None, tuple(partials), False, "few_stable_partials")
    numbers = np.array([p.number for p in used])
    observed = np.array([p.ratio for p in used])
    weights = np.clip(10**(np.array([p.relative_level_db for p in used])/40), .2, 1)

    def errors(parameter: np.ndarray) -> np.ndarray:
        b = parameter[0]
        predicted = np.sqrt((1+b*numbers*numbers)/(1+b))
        return 1200*np.log2(observed/predicted)

    fit = least_squares(lambda p: errors(p)*weights, [1e-5], bounds=([-.0005], [.01]),
                        loss="soft_l1", f_scale=.5, xtol=1e-12, ftol=1e-12, gtol=1e-10)
    stiffness = float(fit.x[0])
    residual = errors(fit.x)
    rms = float(np.sqrt(np.mean(residual**2)))
    reliable = bool(fit.success and stiffness >= -2e-6 and rms < 1 and abs(residual).max() < 2.5 and max(numbers) >= 4)
    return StringInharmonicity(f0, stiffness, rms, tuple(partials), reliable, None if reliable else "shape_mismatch")
