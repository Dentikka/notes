"""Template-based polyphonic transcription for isolated pitched stems.

Templates are recordings/renders of individual notes at the input's tuning and sample
rate. Their harmonic spectra are fitted jointly with a nonnegative sparse activation.
Overtones belong to their template, rather than being labelled as independent notes.
This is a baseline for matching instrument families, not instrument-agnostic AMT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import stft

from notes.ears.transcribe import TrackedNote

if TYPE_CHECKING:
    from notes.engine.base import Instrument

__all__ = ["TemplateTrack", "instrument_templates", "notes_from_templates", "template_track"]


def instrument_templates(
    instrument: Instrument, pitches: Sequence[int], *, sr: int = 44100, a4: float = 440, seconds: float = 0.6,
) -> dict[int, np.ndarray]:
    """Render a reusable mono note dictionary, including the instrument's insert effects."""
    from notes import Note, Song, Track

    if seconds <= 0 or sr <= 0 or a4 <= 0:
        raise ValueError("template duration, sample rate and tuning must be positive")
    return {p: Song(Track("template", instrument)(Note(p, seconds)), bpm=60, sample_rate=sr,
                    a4=a4, seed=17, loudness=None).render().audio.mean(axis=0)[:round(seconds * sr)]
            for p in pitches}


@dataclass(frozen=True)
class TemplateTrack:
    t: np.ndarray
    pitches: np.ndarray
    activation: np.ndarray
    residual: np.ndarray
    duration: float


def template_track(
    x: np.ndarray,
    sr: int,
    templates: Mapping[int, np.ndarray],
    *,
    n_fft: int = 4096,
    hop: int = 512,
    sparsity: float = 0.03,
    iterations: int = 120,
) -> TemplateTrack:
    """Fit fixed note spectra; return nonnegative (pitches, frames) activation and residual.

    Inputs are finite mono arrays. Templates should contain a single sounding note,
    with no long silence. Sample rate and tuning must match; octave courses of a
    twelve-string should be included in each template. Whitening is capped at 40 dB
    below the reference's strongest mean spectral bin to avoid amplifying tiny noise.
    Residual is relative spectral reconstruction error per frame, not confidence in notes.
    """
    x = np.asarray(x, dtype=float)
    if (x.ndim != 1 or not x.size or not np.isfinite(x).all() or sr <= 0 or not templates
            or not 0 < hop <= n_fft or n_fft < 64 or sparsity < 0 or iterations < 1):
        raise ValueError("need finite mono audio, templates and valid analysis/search parameters")

    def spectrum(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
            raise ValueError("templates must be nonempty finite mono audio")
        y = np.pad(y, (0, max(0, n_fft - len(y))))
        f, t, z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
        return f, t, np.abs(z)

    f, t, target = spectrum(x)
    keep = (f >= 45) & (f <= min(10000, sr * 0.48))
    duration = len(x) / sr
    frames = t < duration
    t, target = t[frames], target[keep][:, frames]
    pitches = np.array(sorted(templates), dtype=int)
    if len(pitches) != len(set(pitches.tolist())):
        raise ValueError("template pitches must be distinct integer MIDI notes")
    columns = []
    for pitch in pitches:
        _, _, mag = spectrum(templates[int(pitch)])
        energy = np.linalg.norm(mag, axis=0)
        sounding = energy > energy.max() * 0.1
        if not sounding.any():
            raise ValueError(f"silent template for MIDI {pitch}")
        columns.append(mag[keep][:, sounding].mean(axis=1))
    dictionary = np.column_stack(columns)
    average = target.mean(axis=1)
    weight = 1 / np.sqrt(np.maximum(average, max(float(average.max()) * 1e-4, 1e-12)))
    dictionary, target = dictionary * weight[:, None], target * weight[:, None]
    norms = np.linalg.norm(dictionary, axis=0)
    dictionary /= np.maximum(norms, 1e-20)
    gram = dictionary.T @ dictionary
    projection = dictionary.T @ target
    lipschitz = max(float(np.linalg.norm(gram, 2)), 1e-12)
    penalty = sparsity * np.linalg.norm(target, axis=0)
    activation = np.zeros_like(projection)
    extrapolated = activation.copy()
    momentum = 1.0
    for _ in range(iterations):
        updated = np.maximum(extrapolated + (projection - gram @ extrapolated - penalty) / lipschitz, 0)
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = updated + ((momentum - 1) / next_momentum) * (updated - activation)
        activation, momentum = updated, next_momentum
    residual = np.linalg.norm(target - dictionary @ activation, axis=0) / np.maximum(
        np.linalg.norm(target, axis=0), 1e-12)
    return TemplateTrack(t, pitches, activation, residual, duration)


def notes_from_templates(
    track: TemplateTrack,
    onsets: Sequence[float] = (),
    *,
    relative_db: float = 18,
    floor_db: float = 35,
    min_dur: float = 0.08,
    gap: float = 0.03,
    max_polyphony: int | None = None,
) -> list[TrackedNote]:
    """Independent note lifetimes from activation, plus pitch-specific re-articulations.

    Notes stop when their own activation falls below thresholds, not when some other
    bass note changes. A supplied onset splits an active pitch only if its activation
    rises by at least 25% around that onset. Limits/thresholds are explicit caller
    assumptions; they do not certify correct transcription of a real recording.
    """
    if min(relative_db, floor_db, min_dur, gap) < 0 or (max_polyphony is not None and max_polyphony < 1):
        raise ValueError("thresholds/durations must be nonnegative and max_polyphony positive")
    t, a = track.t, track.activation
    if a.shape != (len(track.pitches), len(t)) or not len(t):
        return []
    hop = float(t[1] - t[0]) if len(t) > 1 else track.duration
    peak = float(a.max())
    if peak <= 0:
        return []
    active = (a >= a.max(axis=0) * 10 ** (-relative_db / 20)) & (a >= peak * 10 ** (-floor_db / 20))
    if max_polyphony is not None and max_polyphony < len(a):
        ranks = np.argsort(np.argsort(-a, axis=0), axis=0)
        active &= ranks < max_polyphony
    notes = []
    for pitch, values, mask in zip(track.pitches, a, active, strict=True):
        # Close short internal gaps only; do not extend into leading/trailing silence.
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for end, start in zip(edges[1:-1:2], edges[2::2], strict=True):
            if (start - end) * hop <= gap:
                mask[end:start] = True
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            cuts = [int(start)]
            for onset in sorted(onsets):
                i = int(np.searchsorted(t, onset))
                radius = max(1, int(round(0.04 / hop)))
                if start + radius <= i < end - radius:
                    before = float(np.median(values[i - radius:i]))
                    after = float(np.max(values[i:i + radius + 1]))
                    if after > max(1.25 * before, peak * 10 ** (-floor_db / 20)):
                        cuts.append(i)
            cuts.append(int(end))
            for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
                stop = min(float(t[hi - 1] + hop), track.duration)
                if stop - t[lo] >= min_dur:
                    contour_t = t[lo:hi]
                    notes.append(TrackedNote(float(t[lo]), stop, float(pitch),
                                             float(20 * np.log10(max(float(values[lo:hi].max()), 1e-20))),
                                             contour_t, np.full(len(contour_t), float(pitch))))
    return sorted(notes, key=lambda n: (n.start, n.pitch))
