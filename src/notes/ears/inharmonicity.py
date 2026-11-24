"""Known-note plucked-string inharmonicity from simultaneous partial phases.

Audio is an analysis input only. Reliability is a diagnostic heuristic, not a
certificate of instrument realism or a material-property measurement in situ.
"""
from __future__ import annotations

from collections.abc import Sequence
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


@dataclass(frozen=True)
class PartialPitchFit:
    """Supported relative-frequency corrections; no recorded phase or carrier."""

    offsets: tuple[float, ...]
    deltas: tuple[float, ...]
    support_counts: tuple[int, ...]
    spans_cents: tuple[float | None, ...]
    accepted: tuple[bool, ...]
    reasons: tuple[str, ...]


def fit_partial_pitch_cents(
    observations: Sequence[tuple[StringInharmonicity, StringInharmonicity]],
    *, current: Sequence[float] = (), harmonics: int = 12, numbers: Sequence[int] | None = None,
    min_observations: int = 2, max_span_cents: float = 1.5, max_delta_cents: float = 50.,
    min_coverage: float = .9, max_phase_residual: float = .15, min_level_db: float = -35.,
) -> PartialPitchFit:
    """Pool paired(reference, model) relative-frequency measurements of one known note.

    Estimate each StringInharmonicity on its own signal with matching time windows.
    Individual accepted partials can be usable even when the global B-law fit is
    unreliable. Fundamental anchors must be within50 cents across usable pairs;
    this is not a cross-pitch material calibration or an unknown-note transcriber.

    Each supported delta is1200*log2(reference_ratio/model_ratio), pooled by median
    only when support count and total span pass the declared guards. Add it to
    current to obtain candidate partial_pitch_cents controls. The fundamental
    correction is preserved; unsupported/unselected components keep their current
    settings, not an inferred zero. numbers optionally limits the updated indices
    (1-based harmonic numbers). Limits are [-100,100] cents without silent clipping.

    Output describes effective relative frequency, not uniquely identified poles
    or stiffness. Phase values are never returned. Validate actual source root
    order/Nyquist, held windows and the complete nonlinear rig separately. Repeated
    windows are consistency checks, not independent statistical replications.
    """
    if (isinstance(harmonics, bool) or not isinstance(harmonics, int) or not 2 <= harmonics <= 32
            or isinstance(min_observations, bool) or not isinstance(min_observations, int) or min_observations < 1):
        raise ValueError('2..32 harmonics and positive integer observation count required')
    if (not np.isfinite([max_span_cents, max_delta_cents, min_coverage, max_phase_residual, min_level_db]).all()
            or max_span_cents < 0 or max_delta_cents <= 0 or not 0 < min_coverage <= 1 or max_phase_residual <= 0):
        raise ValueError('finite valid frequency-consistency guards required')
    if np.iscomplexobj(current):
        raise ValueError('real current pitch offsets required')
    old = np.asarray(current, dtype=float)
    if old.ndim != 1 or len(old) > harmonics or not np.isfinite(old).all() or np.any(abs(old) > 100):
        raise ValueError('current offsets must fit harmonic count and[-100,100] cents')
    old = np.pad(old, (0, harmonics - len(old)))
    selected = tuple(range(2, harmonics + 1)) if numbers is None else tuple(numbers)
    if any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or not 2 <= h <= harmonics for h in selected):
        raise ValueError('selected harmonic numbers must be unique integers2..harmonics')
    selected = tuple(map(int, selected))
    if len(set(selected)) != len(selected):
        raise ValueError('selected harmonic numbers must be unique integers2..harmonics')
    if not observations:
        raise ValueError('at least one paired measurement required')
    cells: list[list[float]] = [[] for _ in range(harmonics)]
    anchors = []

    def usable(p: InharmonicPartial) -> bool:
        if not p.accepted:
            return False
        if (p.ratio is None or p.ratio <= 0 or p.phase_residual_rms is None
                or not np.isfinite([p.ratio, p.phase_residual_rms, p.coverage,
                                    p.relative_level_db, p.prominence_db]).all()
                or p.phase_residual_rms < 0 or not 0 <= p.coverage <= 1):
            raise ValueError('accepted component has invalid frequency/support values')
        return (p.coverage >= min_coverage and p.phase_residual_rms <= max_phase_residual
                and p.relative_level_db >= min_level_db and p.prominence_db > 12)

    for pair in observations:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not isinstance(p, StringInharmonicity) for p in pair)):
            raise ValueError('paired reference/model StringInharmonicity measurements required')
        reference, model = pair
        if reference.f0 is None or model.f0 is None:
            continue
        if not np.isfinite([reference.f0, model.f0]).all() or min(reference.f0, model.f0) <= 0:
            raise ValueError('positive finite fundamental anchors required')
        anchors.extend((reference.f0, model.f0))
        maps = []
        for profile in pair:
            by_number = {p.number: p for p in profile.partials}
            if len(by_number) != len(profile.partials):
                raise ValueError('duplicate component numbers in one measurement')
            maps.append(by_number)
        for h in selected:
            a, b = maps[0].get(h), maps[1].get(h)
            if a is not None and b is not None and usable(a) and usable(b):
                cells[h - 1].append(float(1200 * np.log2(a.ratio / b.ratio)))
    if anchors and 1200 * np.log2(max(anchors) / min(anchors)) > 50:
        raise ValueError('observations must refer to the same known anchor note')
    offsets, deltas = old.copy(), np.zeros(harmonics)
    spans: list[float | None] = [None] * harmonics
    accepted = [False] * harmonics
    reasons = ['anchor_preserved'] + ['not_selected'] * (harmonics - 1)
    for h in selected:
        values = cells[h - 1]
        spans[h - 1] = float(np.ptp(values)) if values else None
        reason = 'few_supported_observations'
        if len(values) >= min_observations:
            delta = float(np.median(values))
            reason = 'inconsistent_relative_frequency'
            if spans[h - 1] <= max_span_cents:
                reason = 'delta_limit'
                if abs(delta) <= max_delta_cents:
                    reason = 'offset_limit'
                    if abs(old[h - 1] + delta) <= 100:
                        offsets[h - 1] += delta
                        deltas[h - 1] = delta
                        accepted[h - 1] = True
                        reason = 'supported'
        reasons[h - 1] = reason
    return PartialPitchFit(tuple(map(float, offsets)), tuple(map(float, deltas)), tuple(map(len, cells)),
                           tuple(spans), tuple(accepted), tuple(reasons))


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
