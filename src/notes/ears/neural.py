"""Optional Basic Pitch ONNX inference with a small, inspectable note decoder.

Uses the official Spotify model's waveform interface, not the basic-pitch Python
package. Supply a locally downloaded model; no implicit network access. Window size,
overlap and output names follow https://github.com/spotify/basic-pitch . Our decoder
uses pitch-specific onsets and frame probabilities and is not upstream's decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, resample_poly

from notes.ears.transcribe import TrackedNote


@dataclass(frozen=True)
class NeuralTrack:
    t: np.ndarray
    note: np.ndarray
    onset: np.ndarray
    contour: np.ndarray
    duration: float


def basic_pitch_track(x: np.ndarray, sr: int, model: str | Path, *, a4: float = 440) -> NeuralTrack:
    """Run Basic Pitch at 22050 Hz on CPU, returning native pitch probabilities.

    For non-concert tuning, time-stretch by resampling before inference and map the
    timestamps back. MIDI labels then refer to ``a4``. Model windows overlap by 30
    frames; half-overlaps are dropped and actual window timestamps are retained.
    Requires the optional onnxruntime package. Model files are supplied by the caller.
    """
    import onnxruntime as ort

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or sr <= 0 or not np.isfinite(a4) or a4 <= 0:
        raise ValueError("need nonempty finite mono audio, positive sample rate and tuning")
    ratio = Fraction(22050 * a4 / (sr * 440)).limit_denominator(100000)
    audio = resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)
    audio = np.pad(audio, (3840, 0))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    output_names = ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:0"]
    times, notes, onsets, contours = [], [], [], []
    for start in range(0, len(audio), 36164):
        window = audio[start:start + 43844]
        window = np.pad(window, (0, 43844 - len(window)))
        predicted = session.run(output_names, {"serving_default_input_2:0": window[None, :, None]})
        t = (start - 3840 + np.arange(15, 157) * 256) / (22050 * a4 / 440)
        times.append(t)
        for values, destination in zip(predicted, (notes, onsets, contours), strict=True):
            destination.append(values[0, 15:-15])
    t = np.concatenate(times)
    valid = (t >= 0) & (t < len(x) / sr)
    return NeuralTrack(t[valid], np.concatenate(notes)[valid], np.concatenate(onsets)[valid],
                       np.concatenate(contours)[valid], len(x) / sr)


def notes_from_probabilities(
    track: NeuralTrack, *, low: int = 21, high: int = 108,
    onset_threshold: float = 0.5, frame_threshold: float = 0.3, min_dur: float = 0.08,
) -> list[TrackedNote]:
    """Decode per-pitch probabilities, never using another voice's attacks.

    Frame-active runs provide a fallback when an onset is missed. Small internal
    gaps up to 50 ms are joined; within a run only that pitch's predicted onsets
    can create re-picks. Contours are limited to one semitone around each MIDI label.
    These are model estimates, not verified note labels.
    """
    if not 21 <= low <= high <= 108 or not 0 < onset_threshold <= 1 or not 0 < frame_threshold <= 1 or min_dur <= 0:
        raise ValueError("invalid MIDI range, probability thresholds or minimum duration")
    if not len(track.t):
        return []
    t = track.t
    hop = float(np.median(np.diff(t))) if len(t) > 1 else track.duration
    notes = []
    for pitch in range(low, high + 1):
        index = pitch - 21
        energy = track.note[:, index]
        active = energy >= frame_threshold
        edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
        for end, start in zip(edges[1:-1:2], edges[2::2], strict=True):
            if t[start] - t[end] <= 0.05:
                active[end:start] = True
        edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
        peaks, _ = find_peaks(track.onset[:, index], height=onset_threshold,
                              distance=max(1, round(0.06 / hop)))
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            candidates = peaks[(peaks >= max(0, start - round(0.06 / hop))) & (peaks < end)]
            first = min(int(start), int(candidates[0])) if len(candidates) else int(start)
            cuts = [first] + [int(i) for i in candidates if t[i] - t[first] >= 0.06] + [int(end)]
            for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
                stop = min(float(t[hi - 1] + hop), track.duration)
                if stop - t[lo] < min_dur:
                    continue
                bins = np.arange(max(0, index * 3 - 3), min(264, index * 3 + 4))
                weights = np.exp(-0.5 * ((bins - index * 3) / 1.5) ** 2)
                salience = track.contour[lo:hi, bins] * weights
                contour = np.where(salience.max(axis=1) > 1e-9,
                                   21 + bins[np.argmax(salience, axis=1)] / 3, float(pitch))
                amplitude = float(np.mean(energy[max(lo, start):hi]))
                notes.append(TrackedNote(float(t[lo]), stop, float(pitch),
                                         20 * np.log10(max(amplitude, 1e-9)), t[lo:hi], contour))
    return sorted(notes, key=lambda n: (n.start, n.pitch))
