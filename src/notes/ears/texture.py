"""Known-note residual-band diagnostics and a bounded one-step texture fit.

Residual energy includes modulation, inharmonic components and window leakage.
It does not identify friction noise or certify realism. No recording, phase or
waveform is transferred to the renderer; only a scalar control is estimated.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


@dataclass(frozen=True)
class InterharmonicEnergy:
    fraction: float
    db: float
    median_fraction: float
    tracked_f0_min: float
    tracked_f0_max: float
    band_relative_db: float
    window_relative_db: float
    band_audible: bool
    sample_rate: int
    expected_f0: float
    start: float
    end: float
    band_low: float
    band_high: float


def _db(fraction: float) -> float:
    return float(10*np.log10(max(fraction, 1e-20)))


def interharmonic_energy(audio: np.ndarray, sr: int, f0: float, *, start: float = 1., end: float = 2.5,
                        band: tuple[float, float] = (1000., 10000.)) -> InterharmonicEnergy:
    """Measure energy outside +/-quarter-fundamental harmonic zones.

    Mono isolated note, an explicit correct pitch prior and absolute clip-clock
    window are required. Frame tuning follows a salience grid within +/-4% of
    the prior. Blackman windows cover at least 16 cycles or 60 ms. Silent upper
    bands and quiet median window energy are flagged relative to full-band or
    peak-frame energy at -60 dB; their ratios must not drive fitting. FM can
    produce residual energy without random noise.
    """
    x = np.asarray(audio, dtype=float)
    if (x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.std(x) < 1e-8
            or not isinstance(sr, int) or isinstance(sr, bool) or sr < 1000
            or not np.isfinite(f0) or not 20 <= f0 < sr/4):
        raise ValueError('finite non-silent mono audio, integer sample rate and known f0 required')
    if (not isinstance(band, (tuple, list)) or len(band) != 2 or not np.isfinite(band).all()
            or not 0 <= band[0] < band[1] or not np.isfinite([start, end]).all()
            or start < 0 or end <= start or end > len(x)/sr):
        raise ValueError('valid spectral band and clip-clock window required')
    window = max(round(sr*.06), round(16*sr/f0))
    hop = round(sr*.02)
    if len(x) < window+2*hop:
        raise ValueError('texture measurement needs at least three full spectral frames')
    nfft = max(8192, 1 << (window-1).bit_length())
    hz, time, z = stft(x, fs=sr, window='blackman', nperseg=window, noverlap=window-hop,
                       nfft=nfft, boundary='zeros', padded=True)
    use = (time >= start) & (time <= end)
    if use.sum() < 3:
        raise ValueError('texture window needs at least three spectral frames')
    whole_energy = np.sum(abs(z)**2, axis=0)
    power = abs(z[:, use])**2
    grid = f0*np.linspace(.96, 1.04, 101)
    harmonics = np.arange(1, min(6, int((sr/2)/grid.max()))+1)
    scores = np.asarray([np.interp((grid[:, None]*harmonics).ravel(), hz, column)
                         .reshape(len(grid), -1)@(1/harmonics) for column in power.T])
    tracked = grid[np.argmax(scores, axis=1)]
    nearest = np.maximum(1, np.rint(hz[:, None]/tracked[None, :]))
    outside = abs(hz[:, None]-nearest*tracked) >= .25*tracked
    low, high = float(band[0]), float(min(band[1], sr/2))
    bins = (hz >= low) & (hz < high)
    if not bins.any():
        raise ValueError('spectral band lies outside the available frequencies')
    total = float(power[bins].sum())
    full = float(power.sum())
    if total <= 0 or full <= 0:
        raise ValueError('texture window has no measurable energy')
    residual = float(power[bins][outside[bins]].sum())
    per_frame = np.sum(power[bins]*outside[bins], axis=0)/np.maximum(1e-30, np.sum(power[bins], axis=0))
    relative = _db(total/full)
    window_relative = _db(float(np.median(np.sum(power, axis=0)))/max(1e-30, float(whole_energy.max())))
    fraction = min(1., max(0., residual/total))
    return InterharmonicEnergy(fraction, _db(fraction), float(np.median(per_frame)),
        float(tracked.min()), float(tracked.max()), relative, window_relative,
        bool(relative >= -60 and window_relative >= -60), sr, float(f0),
        float(start), float(end), low, high)


@dataclass(frozen=True)
class RoughnessEstimate:
    roughness: float | None
    reason: str
    limited: bool
    target_fraction: float
    baseline_fraction: float
    probe_fraction: float | None


def _control_options(probe_roughness: float, max_roughness: float, tolerance_db: float) -> None:
    try:
        finite = bool(np.isfinite([probe_roughness, max_roughness, tolerance_db]).all())
    except (TypeError, ValueError) as error:
        raise ValueError('finite scalar texture controls required') from error
    if (not finite
            or not 0 < probe_roughness <= .25 or not 0 < max_roughness <= .25 or tolerance_db < 0):
        raise ValueError('positive roughness controls <=.25 and nonnegative tolerance required')


def estimate_roughness(reference: InterharmonicEnergy, baseline: InterharmonicEnergy,
                       probes: Sequence[InterharmonicEnergy] = (), *, probe_roughness: float = .03,
                       max_roughness: float = .25, tolerance_db: float = 1.) -> RoughnessEstimate:
    """Predict a scalar using weak-response quadratic scaling, never MERT.

    An already sufficient/excessive baseline needs no added grain. Inaudible
    bands and an unresolvable or nonincreasing probe response cannot identify a
    control. Positive predictions require a new render; they are not fitted
    physical material constants. The default 1 dB tolerance reflects the observed
    numerical/seed variation of the experimental bowed source, not a standard.
    """
    _control_options(probe_roughness, max_roughness, tolerance_db)
    fields = ('sample_rate', 'expected_f0', 'start', 'end', 'band_low', 'band_high')
    for measurement in (baseline, *probes):
        if any(getattr(measurement, key) != getattr(reference, key) for key in fields):
            raise ValueError('texture measurements must use the same protocol')
    target, base = reference.fraction, baseline.fraction
    if any(not np.isfinite(m.fraction) or not 0 <= m.fraction <= 1 for m in (reference, baseline, *probes)):
        raise ValueError('residual fractions must be finite and lie from zero to one')
    if not reference.band_audible or not baseline.band_audible:
        return RoughnessEstimate(None, 'inaudible_band', False, target, base, None)
    target_db, base_db = _db(target), _db(base)
    if base_db >= target_db-tolerance_db:
        reason = 'baseline_within_tolerance' if abs(base_db-target_db) <= tolerance_db else 'baseline_excess'
        return RoughnessEstimate(0., reason, False, target, base, None)
    if not probes:
        raise ValueError('a deficient baseline needs at least one positive probe')
    if not all(m.band_audible for m in probes):
        return RoughnessEstimate(None, 'inaudible_probe_band', False, target, base, None)
    probed = float(np.mean([m.fraction for m in probes]))
    if _db(probed)-base_db <= tolerance_db:
        return RoughnessEstimate(None, 'probe_below_resolution', False, target, base, probed)
    predicted = float(probe_roughness*np.sqrt((target-base)/(probed-base)))
    return RoughnessEstimate(min(predicted, max_roughness), 'prediction', predicted > max_roughness,
                             target, base, probed)


@dataclass(frozen=True)
class TextureTrial:
    roughness: float
    seed: int
    measurement: InterharmonicEnergy


@dataclass(frozen=True)
class TextureFitResult:
    roughness: float
    audio: np.ndarray
    estimate: RoughnessEstimate
    history: tuple[TextureTrial, ...]
    verified_within_tolerance: bool
    verification_regressed: bool


def fit_roughness(reference: np.ndarray, render: Callable[[float, int], np.ndarray], sr: int, f0: float, *,
                  probe_roughness: float = .03, max_roughness: float = .25, tolerance_db: float = 1.,
                  seeds: tuple[int, ...] = (17, 29), verification_seed: int = 41,
                  start: float = 1., end: float = 2.5, band: tuple[float, float] = (1000., 10000.),
                  on_trial: Callable[[TextureTrial], None] | None = None) -> TextureFitResult:
    """Render baseline, seeded probes and one independent verification.

    The renderer takes roughness and a material seed, returning a non-silent
    mono waveform. Notes, timing and all other controls must stay fixed. A
    sufficient baseline returns immediately without probes. A positive estimate
    is retained only if verification does not worsen this texture axis beyond
    tolerance or overshoot the target. Pitch, timbre and performance accuracy
    need separate checks; a failed one-step prediction is not a successful fit.
    """
    _control_options(probe_roughness, max_roughness, tolerance_db)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**64
                   for s in (*seeds, verification_seed)) or verification_seed in seeds):
        raise ValueError('distinct unsigned seeds and a separate verification seed required')
    target = interharmonic_energy(reference, sr, f0, start=start, end=end, band=band)
    history = []

    def evaluate(roughness: float, seed: int) -> tuple[np.ndarray, InterharmonicEnergy]:
        wave = np.asarray(render(roughness, seed), dtype=float).copy()
        measured = interharmonic_energy(wave, sr, f0, start=start, end=end, band=band)
        trial = TextureTrial(roughness, seed, measured)
        history.append(trial)
        if on_trial is not None:
            on_trial(trial)
        return wave, measured

    base_audio, baseline = evaluate(0., seeds[0])
    if (not target.band_audible or not baseline.band_audible
            or _db(baseline.fraction) >= _db(target.fraction)-tolerance_db):
        estimate = estimate_roughness(target, baseline, probe_roughness=probe_roughness,
                                      max_roughness=max_roughness, tolerance_db=tolerance_db)
        return TextureFitResult(0., base_audio, estimate, tuple(history),
                                bool(target.band_audible and baseline.band_audible
                                     and abs(target.db-baseline.db) <= tolerance_db), False)
    probes = [evaluate(probe_roughness, seed)[1] for seed in seeds]
    estimate = estimate_roughness(target, baseline, probes, probe_roughness=probe_roughness,
                                  max_roughness=max_roughness, tolerance_db=tolerance_db)
    if estimate.roughness is None or estimate.roughness == 0:
        return TextureFitResult(0., base_audio, estimate, tuple(history), False, False)
    audio, verified = evaluate(estimate.roughness, verification_seed)
    error = abs(verified.db-target.db)
    regressed = (not verified.band_audible or verified.db > target.db+tolerance_db
                 or error > abs(baseline.db-target.db)+tolerance_db)
    return TextureFitResult(0. if regressed else estimate.roughness, base_audio if regressed else audio,
                            estimate, tuple(history), bool(error <= tolerance_db and not regressed), bool(regressed))
