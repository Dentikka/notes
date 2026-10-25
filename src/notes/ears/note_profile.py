"""Known-note timbre measurements for procedural synthesis calibration.

Recordings are inputs to analysis only. These features do not certify perceptual realism.
Pitch is supplied explicitly, preventing octave/harmonic mistakes from being hidden.
"""
from __future__ import annotations

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
    energy = uniform_filter1d(x*x, size=size, mode="constant")
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
    envelope = np.sqrt(uniform_filter1d(x*x, size=max(1, round(sr*0.01)), mode="constant"))
    indices = np.minimum((time*sr).astype(int), len(envelope)-1)
    rms_db = 20*np.log10(envelope[indices]+1e-12)
    first = envelope[:min(len(envelope), round(sr*0.6))]
    top = float(first.max())
    t10 = int(np.flatnonzero(first >= top*0.1)[0])
    t90 = int(np.flatnonzero(first >= top*0.9)[0])
    return NoteProfile(start/sr, measured, time, rms_db, np.array(partials).T,
                       np.array(locations), max(0, t90-t10)/sr)


def compare_profiles(reference: NoteProfile, candidate: NoteProfile) -> dict[str, float | int]:
    """Gain-invariant harmonic-envelope errors, pitch and attack; no acceptance label."""
    end = min(reference.time[-1], candidate.time[-1], 2.0)
    times = np.arange(0.02, end, 0.02)
    count = min(reference.partial_db.shape[1], candidate.partial_db.shape[1])
    if len(times) < 3 or count < 2:
        raise ValueError("not enough shared note material")
    ref = np.array([np.interp(times, reference.time, reference.partial_db[:, h]) for h in range(count)]).T
    est = np.array([np.interp(times, candidate.time, candidate.partial_db[:, h]) for h in range(count)]).T
    keep = ref > ref.max()-45
    gain = float(np.median((ref-est)[keep]))
    error = np.minimum(abs(est+gain-ref), 60)
    envelope_ref = np.interp(times, reference.time, reference.rms_db)
    envelope_est = np.interp(times, candidate.time, candidate.rms_db)
    envelope_error = envelope_est-envelope_ref
    envelope_error -= np.median(envelope_error)
    attack = keep & (times[:, None] < 0.15)
    tail = keep & (times[:, None] >= 0.15)
    return {
        "pitch_cents": float(1200*np.log2(candidate.f0/reference.f0)),
        "attack_error_seconds": candidate.attack_seconds-reference.attack_seconds,
        "harmonic_mae_db": float(error[keep].mean()),
        "attack_harmonic_mae_db": float(error[attack].mean()) if attack.any() else 0.0,
        "tail_harmonic_mae_db": float(error[tail].mean()) if tail.any() else 0.0,
        "envelope_mae_db": float(abs(envelope_error).mean()),
        "gain_db": gain,
        "partial_count": count,
        "seconds": float(end),
    }
