"""Known-note timbre measurements for procedural synthesis calibration.

Recordings are inputs to analysis only. These features do not certify perceptual realism.
Pitch is supplied explicitly, preventing octave/harmonic mistakes from being hidden.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft


@dataclass(frozen=True)
class NoteProfile:
    onset: float
    f0: float
    time: np.ndarray
    rms_db: np.ndarray
    partial_db: np.ndarray
    partial_hz: np.ndarray
    attack_seconds: float
    band_db: np.ndarray


def infer_expression_points(
    reference: NoteProfile,
    candidate: NoteProfile,
    *,
    times: Sequence[float] = (0., .08, .16, .3, .5, .8, 1.2, 1.8, 2.4, 2.8),
    gain_floor: float = .02,
) -> tuple[tuple[float, float], ...]:
    """Infer coarse expression gains for one known-note reconstruction.

    Uses RMS envelopes on the original clip clock, including their onset offsets.
    The first two knots do not set the overall gain scale, avoiding amplification
    from near-silent attack samples. The first gain is zero; remaining gains are
    bounded by ``gain_floor`` and 1. This amplitude-only conditional fit does not
    identify velocity, bow pressure, timbre, or a transferable instrument preset.
    No reference waveform or phase is needed by the resulting synthesizer.
    """
    knots = np.asarray(times, dtype=float)
    if (knots.ndim != 1 or len(knots) < 3 or not np.isfinite(knots).all()
            or knots[0] != 0 or np.any(np.diff(knots) <= 0)
            or knots[-1] > min(reference.time[-1]+reference.onset, candidate.time[-1]+candidate.onset)):
        raise ValueError("expression times must start at zero, increase, and fit both note profiles")
    if not np.isfinite(gain_floor) or not 0 < gain_floor <= 1:
        raise ValueError("expression gain_floor must be from 0 to 1, excluding zero")
    ref = np.interp(knots, reference.time+reference.onset, reference.rms_db)
    source = np.interp(knots, candidate.time+candidate.onset, candidate.rms_db)
    db = ref-source
    db -= np.max(db[2:])
    gains = 10**(np.clip(db, 20*np.log10(gain_floor), 0)/20)
    gains[0] = 0.
    return tuple((float(t), float(g)) for t, g in zip(knots, gains, strict=True))


@dataclass(frozen=True)
class PartialDecay:
    """Local exponential-decay estimate, not a measured physical mode or realism score."""

    harmonic: int
    frequency_hz: float
    decay_db_per_second: float
    t60_seconds: float | None
    residual_rms_db: float
    r_squared: float
    audible_fraction: float
    reliable: bool


def partial_decays(
    profile: NoteProfile,
    *,
    start: float = 0.2,
    end: float = 0.6,
    audibility_db: float = 40.0,
    residual_limit_db: float = 1.0,
    minimum_drop_db: float = 0.25,
    minimum_r_squared: float = 0.8,
) -> tuple[PartialDecay, ...]:
    """Fit each partial's dB slope in one explicit, onset-relative time window.

    A reliable estimate must remain audible throughout the window, decay by at
    least ``minimum_drop_db``, have small deviations from an exponential and a
    trend that explains at least ``minimum_r_squared`` of its dB variance.
    Growing, weak and modulated components remain in the report but have no T60.
    T60 is a local extrapolation: compare disjoint windows to reveal changing
    losses or beating rather than treating one slope as a whole-note lifetime.
    """
    if (not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start
            or end > profile.time[-1]):
        raise ValueError("decay window must lie inside the note profile")
    if any(not np.isfinite(v) or v <= 0 for v in (audibility_db, residual_limit_db, minimum_drop_db)):
        raise ValueError("decay thresholds must be finite and positive")
    if not np.isfinite(minimum_r_squared) or not 0 <= minimum_r_squared <= 1:
        raise ValueError("minimum_r_squared must be between 0 and 1")
    keep = (profile.time >= start) & (profile.time <= end)
    time = profile.time[keep]
    if len(time) < 4:
        raise ValueError("decay window needs at least four profile frames")
    values = profile.partial_db[keep]
    centered = time-time.mean()
    slopes = centered @ values / float(centered @ centered)
    residual = np.sqrt(np.mean((values-values.mean(axis=0)-centered[:, None]*slopes)**2, axis=0))
    variance = np.var(values, axis=0)
    explained = 1-np.divide(residual**2, variance, out=np.ones_like(variance), where=variance > 1e-20)
    explained = np.clip(explained, 0., 1.)
    audible = np.mean(values > float(profile.partial_db.max())-audibility_db, axis=0)
    result = []
    for h, hz in enumerate(profile.partial_hz):
        decay = float(-slopes[h])
        reliable = bool(audible[h] == 1.0 and residual[h] <= residual_limit_db
                        and explained[h] >= minimum_r_squared
                        and decay*(time[-1]-time[0]) >= minimum_drop_db)
        result.append(PartialDecay(h+1, float(hz), decay, 60/decay if reliable else None,
                                   float(residual[h]), float(explained[h]), float(audible[h]), reliable))
    return tuple(result)


def profile_note(audio: np.ndarray, sr: int, f0: float, *, harmonics: int = 20) -> NoteProfile:
    """Measure an isolated note around known f0, retaining its harmonic envelope.

    Mono only, at least 100 ms; silence and invalid pitches are rejected. The onset
    is relative to the original clip; output times are relative to that onset.
    Partial windows cannot overlap neighbouring nominal harmonics. Tuning is
    measured near f0, never inferred from the loudest harmonic.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1 or sr < 1000 or len(x) < sr // 10 or not np.isfinite(x).all():
        raise ValueError("audio must be a finite mono note of at least 100 ms")
    if not np.isfinite(f0) or not 20 <= f0 < sr / 4 or harmonics < 2:
        raise ValueError("invalid f0 or harmonic count")
    size = max(8, round(sr * 0.005))
    # Running-sum cancellation can leave a tiny negative residue after a note
    # ends. Energy is nonnegative, including over an exactly silent tail.
    energy = np.maximum(0, uniform_filter1d(x*x, size=size, mode="constant"))
    peak = float(np.sqrt(energy.max()))
    if peak < 1e-8:
        raise ValueError("silent note")
    # Threshold against this note, not the largest note elsewhere in a scale.
    onset_sample = int(np.flatnonzero(energy >= (peak * 0.08) ** 2)[0])
    start = max(0, onset_sample - size // 2)
    x = x[start:]
    window = max(128, min(round(sr * 0.12), round(6 * sr / f0)))
    hop = max(1, round(sr * 0.01))
    nfft = 1 << (max(4096, window * 4) - 1).bit_length()
    freq, time, z = stft(x, fs=sr, nperseg=min(window, len(x)),
                         noverlap=max(0, min(window, len(x))-hop), nfft=nfft,
                         boundary="zeros", padded=True)
    power = abs(z)**2
    reference = power[:, (time >= 0.03) & (time <= min(0.4, len(x)/sr))].mean(axis=1)
    region = np.flatnonzero(abs(freq - f0) < f0 * 0.055)
    if not len(region):
        raise ValueError("cannot resolve fundamental")
    k = int(region[np.argmax(reference[region])])
    offset = 0.0
    if 0 < k < len(freq)-1:
        a, b, c = np.log(reference[k-1:k+2] + 1e-30)
        den = a - 2*b + c
        if den < 0:
            offset = float(np.clip(0.5*(a-c)/den, -0.5, 0.5))
    measured = float((k+offset)*sr/nfft)
    partials, locations = [], []
    for h in range(1, harmonics+1):
        expected = h * measured
        if expected + measured * 0.3 >= sr / 2:
            break
        region = np.flatnonzero(abs(freq-expected) < measured * 0.3)
        at = int(region[np.argmax(reference[region])])
        locations.append(float(freq[at]))
        band = abs(freq-freq[at]) < max(sr/nfft*2, measured*0.07)
        partials.append(10*np.log10(power[band].sum(axis=0)+1e-20))
    # RMS is measured separately from STFT harmonic energy, so noise is not ignored.
    envelope = np.sqrt(np.maximum(0, uniform_filter1d(x*x, size=max(1, round(sr*0.01)), mode="constant")))
    indices = np.minimum((time*sr).astype(int), len(envelope)-1)
    rms_db = 20*np.log10(envelope[indices]+1e-12)
    first = envelope[:min(len(envelope), round(sr*0.6))]
    top = float(first.max())
    t10 = int(np.flatnonzero(first >= top*0.1)[0])
    t90 = int(np.flatnonzero(first >= top*0.9)[0])
    edges = np.geomspace(20, sr*0.48, 25)
    bands = np.array([10*np.log10(power[(freq >= a) & (freq < b)].sum(axis=0)+1e-20)
                      for a, b in zip(edges[:-1], edges[1:], strict=True)]).T
    return NoteProfile(start/sr, measured, time, rms_db, np.array(partials).T,
                       np.array(locations), max(0, t90-t10)/sr, bands)


def harmonic_shape_distance(
    reference: NoteProfile,
    candidate: NoteProfile,
    *,
    start: float = 0.15,
    end: float | None = None,
    step: float = 0.02,
    audibility_db: float = 45.0,
    floor_db: float = 60.0,
) -> float:
    """Mean partial-ratio difference in dB, independent of each frame's gain.

    Both profiles are sampled on a shared onset-relative clock. Each frame's
    partial energies are normalized to their sum; the union of audible cells
    retains added as well as missing harmonics. Only shared harmonic indices
    are compared. This deliberately removes expression/decay of overall level,
    not changes in relative partial decay. It measures neither pitch, phase,
    nonharmonic noise, nor perceptual realism; check those separately.

    Useful for isolating timbre from expression or comparing renders under
    numerical refinement. A small distance alone does not prove convergence.
    """
    shared_end = min(reference.time[-1], candidate.time[-1])
    end = min(shared_end, 2.0) if end is None else end
    if (not np.isfinite(start) or not np.isfinite(end) or not np.isfinite(step)
            or start < 0 or end <= start or end > shared_end or step <= 0):
        raise ValueError("shape window and step must be finite and lie inside both profiles")
    if (not np.isfinite(audibility_db) or not np.isfinite(floor_db)
            or not 0 < audibility_db <= floor_db):
        raise ValueError("shape thresholds must satisfy 0 < audibility_db <= floor_db")
    times = np.arange(start, end, step)
    count = min(reference.partial_db.shape[1], candidate.partial_db.shape[1])
    if len(times) < 3 or count < 2:
        raise ValueError("shape window needs at least three frames and two shared partials")

    def normalized(profile: NoteProfile) -> np.ndarray:
        values = np.array([np.interp(times, profile.time, profile.partial_db[:, h])
                           for h in range(count)]).T
        # Removing the row maximum first avoids overflow for large common gains.
        values -= values.max(axis=1, keepdims=True)
        return values-10*np.log10(np.sum(10**(values/10), axis=1, keepdims=True))

    ref, est = normalized(reference), normalized(candidate)
    keep = (ref > -audibility_db) | (est > -audibility_db)
    if not keep.any():
        raise ValueError("shape thresholds leave no audible partials")
    error = abs(np.maximum(ref, -floor_db)-np.maximum(est, -floor_db))
    return float(error[keep].mean())


def compare_profiles(reference: NoteProfile, candidate: NoteProfile) -> dict[str, float | int]:
    """Gain-invariant harmonic-envelope errors, pitch and attack; no acceptance label."""
    end = min(reference.time[-1], candidate.time[-1], 2.0)
    times = np.arange(0.02, end, 0.02)
    count = min(reference.partial_db.shape[1], candidate.partial_db.shape[1])
    if len(times) < 3 or count < 2:
        raise ValueError("not enough shared note material")
    ref = np.array([np.interp(times, reference.time, reference.partial_db[:, h]) for h in range(count)]).T
    est = np.array([np.interp(times, candidate.time, candidate.partial_db[:, h]) for h in range(count)]).T
    # Match one overall gain by total harmonic energy, then use the UNION of
    # audible reference and candidate cells. A reference-only mask silently lets
    # an optimizer add strong harmonics where the recording has none.
    ref_peak, est_peak = float(ref.max()), float(est.max())
    ref_energy = float(np.sum(10**((ref-ref_peak)/10)))
    est_energy = float(np.sum(10**((est-est_peak)/10)))
    gain = ref_peak-est_peak + 10*np.log10(ref_energy/est_energy)
    reference_audible = ref > ref_peak-45
    candidate_audible = est+gain > ref_peak-45
    keep = reference_audible | candidate_audible
    floor = ref_peak-60
    error = np.minimum(abs(np.maximum(est+gain, floor)-np.maximum(ref, floor)), 60)
    band_ref = np.array([np.interp(times, reference.time, reference.band_db[:, h])
                         for h in range(reference.band_db.shape[1])]).T
    band_est = np.array([np.interp(times, candidate.time, candidate.band_db[:, h])
                         for h in range(candidate.band_db.shape[1])]).T
    band_gain = 10*np.log10(np.sum(10**(band_ref/10))/np.sum(10**(band_est/10)))
    band_peak = float(band_ref.max())
    band_keep = (band_ref > band_peak-45) | (band_est+band_gain > band_peak-45)
    band_error = abs(np.maximum(band_ref, band_peak-60)-np.maximum(band_est+band_gain, band_peak-60))
    envelope_ref = np.interp(times, reference.time, reference.rms_db)
    envelope_est = np.interp(times, candidate.time, candidate.rms_db)
    envelope_error = envelope_est-envelope_ref
    envelope_error -= np.median(envelope_error)
    attack = keep & (times[:, None] < 0.15)
    tail = keep & (times[:, None] >= 0.15)
    return {
        "profile_metric_version": 2,
        "spurious_partial_cells": int(np.sum(candidate_audible & ~reference_audible)),
        "missing_partial_cells": int(np.sum(reference_audible & ~candidate_audible)),
        "pitch_cents": float(1200*np.log2(candidate.f0/reference.f0)),
        "attack_error_seconds": candidate.attack_seconds-reference.attack_seconds,
        "broadband_mae_db": float(np.minimum(band_error, 60)[band_keep].mean()),
        "harmonic_mae_db": float(error[keep].mean()),
        "attack_harmonic_mae_db": float(error[attack].mean()) if attack.any() else 0.0,
        "tail_harmonic_mae_db": float(error[tail].mean()) if tail.any() else 0.0,
        "envelope_mae_db": float(abs(envelope_error).mean()),
        "gain_db": gain,
        "partial_count": count,
        "seconds": float(end),
    }


def note_profile_loss(report: Mapping[str, float | int]) -> float:
    """Diagnostic calibration objective; not a perceptual-realism score."""
    return float(report["harmonic_mae_db"] + 0.45*report["envelope_mae_db"]
                 + 0.35*report["broadband_mae_db"])


def audit_profile_set(
    references: Mapping[str, NoteProfile],
    baseline: Mapping[str, NoteProfile],
    candidate: Mapping[str, NoteProfile],
    *,
    regression_tolerance: float = 0.0,
) -> dict[str, object]:
    """Report every note regression even when the average candidate improves.

    All three mappings must cover exactly the same nonempty case set. A tolerance
    of .02 allows at most a 2% per-case increase. Passing this diagnostic guard
    does not establish realism; pitch/attack and individual metrics remain visible.
    """
    if not references or set(references) != set(baseline) or set(references) != set(candidate):
        raise ValueError("reference, baseline and candidate must have identical nonempty case IDs")
    if not np.isfinite(regression_tolerance) or regression_tolerance < 0:
        raise ValueError("regression_tolerance must be finite and nonnegative")
    reports = {}
    regressions = []
    before, after = [], []
    for key in sorted(references):
        old = compare_profiles(references[key], baseline[key])
        new = compare_profiles(references[key], candidate[key])
        old_loss, new_loss = note_profile_loss(old), note_profile_loss(new)
        regressed = new_loss > old_loss*(1+regression_tolerance)+1e-6
        if regressed:
            regressions.append(key)
        reports[key] = {"baseline": old, "candidate": new, "baseline_loss": old_loss,
                        "candidate_loss": new_loss, "regressed": regressed}
        before.append(old_loss)
        after.append(new_loss)
    return {"status": "diagnostic-only", "profile_metric_version": 2,
            "baseline_mean_loss": float(np.mean(before)), "candidate_mean_loss": float(np.mean(after)),
            "no_regression": not regressions, "regressions": regressions,
            "regression_tolerance": regression_tolerance, "cases": reports}
