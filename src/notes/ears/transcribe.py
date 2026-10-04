"""Transcription helpers for matching a recording: what notes and hits a stem holds.

- `pitch_track`: the fundamental of one voice (a bass line, a lead) per frame by YIN
  (de Cheveigné and Kawahara, JASA 2002), with its clarity and level;
- `melody_track`: the most salient pitch per frame in a register, by harmonic summation, for
  a lead over chords where YIN, made for a single voice, fails;
- `notes_from_track`: that track cut into notes at onsets and at pitch jumps, each with its
  median pitch and its contour (where bends, slides and vibrato show);
- `drum_hits`: the onsets of a drum stem, each explained as a non-negative mix of a kit's
  own sounds, so the hits come back as names the kit can play;
- `chroma` and `best_chord`: pitch-class profiles on a given tuning and the chord they fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

__all__ = [
    "CHORD_TEMPLATES",
    "PitchTrack",
    "TrackedNote",
    "best_chord",
    "chroma",
    "drum_hits",
    "drum_profile",
    "melody_track",
    "notes_from_track",
    "pitch_track",
]

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
#: Chord qualities as pitch classes above the root.
CHORD_TEMPLATES = {
    "5": (0, 7), "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11), "sus2": (0, 2, 7), "sus4": (0, 5, 7), "add9": (0, 2, 4, 7), "madd9": (0, 2, 3, 7),
}
#: Log-spaced bands (Hz) of the drum profiles.
_DRUM_BANDS = np.geomspace(30.0, 16000.0, 25)


@dataclass(frozen=True)
class PitchTrack:
    """Per frame: centre time (s), fundamental (Hz, nan where unvoiced), clarity (1 minus the
    normalised difference at the chosen lag) and level (dB)."""

    t: np.ndarray
    f0: np.ndarray
    clarity: np.ndarray
    level_db: np.ndarray


@dataclass(frozen=True)
class TrackedNote:
    """A note found in a pitch track: its span (s), median pitch (MIDI, fractional, on the
    tuning given), level (dB) and contour (times and MIDI pitches of its frames)."""

    start: float
    end: float
    pitch: float
    level_db: float
    contour_t: np.ndarray
    contour: np.ndarray

    @property
    def span(self) -> float:
        """How far the pitch moves inside the note, in semitones (5th to 95th percentile)."""
        return float(np.percentile(self.contour, 95) - np.percentile(self.contour, 5))


def pitch_track(
    x: np.ndarray, sr: int, fmin: float = 40.0, fmax: float = 1000.0, *, hop: int = 256,
    threshold: float = 0.15, floor_db: float = 50.0,
) -> PitchTrack:
    """YIN on a mono signal: the first lag whose cumulative-mean-normalised difference falls
    under `threshold` (else the deepest, marked unvoiced), refined by a parabola. Frames more
    than `floor_db` under the loudest are unvoiced too."""
    x = np.asarray(x, dtype=float)
    max_lag, min_lag = int(np.ceil(sr / fmin)), max(2, int(sr / fmax))
    window = max_lag
    frame = window + max_lag + 1
    padded = np.pad(x, (frame // 2, frame))
    starts = np.arange(0, len(x), hop)
    n_fft = 1 << (2 * frame).bit_length()
    taus = np.arange(max_lag + 1)
    f0 = np.full(len(starts), np.nan)
    clarity, level = np.zeros(len(starts)), np.zeros(len(starts))
    for c0 in range(0, len(starts), 256):
        frames = np.stack([padded[s : s + frame] for s in starts[c0 : c0 + 256]])
        head = np.fft.rfft(frames[:, :window], n_fft)
        whole = np.fft.rfft(frames, n_fft)
        r = np.fft.irfft(np.conj(head) * whole, n_fft)[:, : max_lag + 1]
        cs = np.concatenate((np.zeros((len(frames), 1)), np.cumsum(frames**2, axis=1)), axis=1)
        energy = cs[:, taus + window] - cs[:, taus]
        d = energy[:, :1] + energy - 2.0 * r
        d[:, 0] = 0.0
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(np.cumsum(d[:, 1:], axis=1), 1e-12)
        level[c0 : c0 + len(frames)] = 10.0 * np.log10(energy[:, 0] / window + 1e-20)
        for i, row in enumerate(cmnd):
            below = np.flatnonzero(row[min_lag:max_lag] < threshold)
            k = int(below[0]) + min_lag if len(below) else min_lag + int(np.argmin(row[min_lag:max_lag]))
            while k + 1 < max_lag and row[k + 1] < row[k]:
                k += 1
            shift = 0.0
            if 1 <= k < max_lag and (den := row[k - 1] - 2.0 * row[k] + row[k + 1]) > 0:
                shift = 0.5 * (row[k - 1] - row[k + 1]) / den
            clarity[c0 + i] = 1.0 - row[k]
            if row[k] < threshold:
                f0[c0 + i] = sr / (k + shift)
    f0[level < level.max() - floor_db] = np.nan
    return PitchTrack(starts / sr, f0, clarity, level)


def melody_track(
    x: np.ndarray, sr: int, fmin: float = 200.0, fmax: float = 1400.0, *, hop: int = 256, n_fft: int = 4096,
    harmonics: int = 6, smooth: int = 5, min_clarity: float = 0.35,
) -> PitchTrack:
    """The predominant pitch in `fmin`-`fmax` per frame: the candidate (10-cent grid) whose
    first `harmonics` partials (weights 0.8^k) carry the most compressed magnitude, median
    filtered over `smooth` frames against octave flips. Clarity is the winner's share of the
    frame's salience above its median; frames under `min_clarity` are unvoiced."""
    x = np.asarray(x, dtype=float)
    window = np.hanning(n_fft)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    grid = fmin * 2.0 ** (np.arange(int(1200 * np.log2(fmax / fmin)) // 10 + 1) / 120.0)
    weights = 0.8 ** np.arange(harmonics)
    starts = np.arange(0, max(len(x) - n_fft, 1), hop)
    best, clarity, level = np.zeros(len(starts)), np.zeros(len(starts)), np.zeros(len(starts))
    for i, s in enumerate(starts):
        seg = x[s : s + n_fft]
        mag = np.sqrt(np.abs(np.fft.rfft(np.pad(seg, (0, n_fft - len(seg))) * window)))
        sal = sum(w * np.interp(grid * (h + 1), f, mag) for h, w in enumerate(weights))
        k = int(np.argmax(sal))
        best[i] = grid[k]
        floor = float(np.median(sal))
        clarity[i] = (sal[k] - floor) / max(float(sal[k]), 1e-12)
        level[i] = 10.0 * np.log10(np.mean(seg**2) + 1e-20)
    cents = 1200.0 * np.log2(best / fmin)
    pad = smooth // 2
    cents = np.array([np.median(cents[max(0, i - pad) : i + pad + 1]) for i in range(len(cents))])
    f0 = fmin * 2.0 ** (cents / 1200.0)
    f0[clarity < min_clarity] = np.nan
    return PitchTrack((starts + n_fft / 2) / sr, f0, clarity, level)


def notes_from_track(
    track: PitchTrack, onsets: Sequence[float] = (), a4: float = 440.0, *, jump: float = 0.7,
    hold: float = 0.05, min_dur: float = 0.06, gap: float = 0.04,
) -> list[TrackedNote]:
    """Cut a pitch track into notes: a note runs over voiced frames (bridging unvoiced gaps up
    to `gap` s) and ends at the next onset, or where the pitch moves more than `jump` semitones
    from the note's median and stays there for `hold` s. Notes under `min_dur` s are dropped."""
    t = track.t
    hop = float(t[1] - t[0]) if len(t) > 1 else 0.01
    midi = 69.0 + 12.0 * np.log2(track.f0 / a4)
    voiced = np.isfinite(midi)
    bridge = int(round(gap / hop))
    run = voiced.copy()
    i = 0
    while i < len(run):  # close short unvoiced gaps inside a note
        if not run[i]:
            j = i
            while j < len(run) and not run[j]:
                j += 1
            if i > 0 and j < len(run) and j - i <= bridge:
                run[i:j] = True
            i = j
        else:
            i += 1
    cuts = set(np.searchsorted(t, list(onsets)))
    hold_n = max(1, int(round(hold / hop)))
    notes: list[TrackedNote] = []
    k = 0
    while k < len(t):
        if not run[k]:
            k += 1
            continue
        start = k
        k += 1
        while k < len(t) and run[k] and k not in cuts:
            ahead = midi[k : k + hold_n]
            ahead = ahead[np.isfinite(ahead)]
            here = midi[start:k]
            here = here[np.isfinite(here)]
            if len(ahead) == hold_n and len(here) and np.all(np.abs(ahead - np.median(here)) > jump):
                break
            k += 1
        seg = slice(start, k)
        pitches = midi[seg]
        ok = np.isfinite(pitches)
        if ok.sum() and (k - start) * hop >= min_dur:
            notes.append(TrackedNote(float(t[start]), float(t[k - 1] + hop), float(np.median(pitches[ok])),
                                     float(np.max(track.level_db[seg])), t[seg][ok], pitches[ok]))
    return notes


def drum_profile(x: np.ndarray, sr: int, t: float, after: float = 0.08, before: float = 0.08) -> np.ndarray:
    """What a hit at `t` adds: mean power per log band over `after` s from `t`, minus the
    mean power per band over the `before` s just before it (never negative), so a sound still
    ringing from an earlier hit, decaying, adds nothing."""

    def bands(a: int, b: int) -> np.ndarray:
        seg = x[max(a, 0) : max(b, 0)]
        if len(seg) < 64:
            return np.zeros(len(_DRUM_BANDS) - 1)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2 / len(seg) ** 2  # mean power, any length
        f = np.fft.rfftfreq(len(seg), 1.0 / sr)
        return np.array([spec[(f >= lo) & (f < hi)].sum() for lo, hi in zip(_DRUM_BANDS[:-1], _DRUM_BANDS[1:],
                                                                              strict=True)])

    s = int(round(t * sr))
    return np.maximum(bands(s, s + int(after * sr)) - bands(s - int(before * sr), s), 0.0)


def drum_hits(
    x: np.ndarray, sr: int, templates: Mapping[str, np.ndarray], onsets: Sequence[float], *, min_level: float = 0.25,
) -> list[tuple[float, str, float]]:
    """Explain every onset of a drum stem as a non-negative mix of `templates` (a kit's own
    sounds' `drum_profile`s): (time, sound, level) for every sound in the mix whose amplitude
    is at least `min_level` of that sound's typical hit in the stem (its 90th percentile), the
    level being that ratio. Judging each sound against itself keeps a quiet hat on top of a
    ringing kick, and drops the kick's leftovers under the hat."""
    names = list(templates)
    basis = np.stack([templates[n] for n in names], axis=1)
    scale = np.maximum(basis.sum(axis=0), 1e-30)
    amps = []
    for t in onsets:
        profile = drum_profile(x, sr, float(t))
        weights = nnls(basis / scale, profile)[0] if profile.sum() > 0 else np.zeros(len(names))
        amps.append(np.sqrt(weights / scale))
    amps = np.array(amps).reshape(len(onsets), len(names))
    typical = np.array([np.percentile(a[a > 0], 90) if (a > 0).any() else 1.0 for a in amps.T])
    out = []
    for t, row in zip(onsets, amps, strict=True):
        for name, a, ref in zip(names, row, typical, strict=True):
            if a >= min_level * ref:
                out.append((float(t), name, float(a / ref)))
    return out


def chroma(x: np.ndarray, sr: int, a4: float = 440.0, *, n_fft: int = 8192, hop: int = 2048, fmin: float = 60.0,
           fmax: float = 2000.0) -> tuple[np.ndarray, np.ndarray]:
    """Frame centres (s) and a (12, frames) pitch-class profile (C = 0) on the tuning `a4`."""
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = (f >= fmin) & (f <= fmax)
    classes = np.mod(np.round(12.0 * np.log2(f[sel] / a4)).astype(int) + 9, 12)
    window = np.hanning(n_fft)
    starts = np.arange(0, max(len(x) - n_fft, 1), hop)
    out = np.zeros((12, len(starts)))
    for i, s in enumerate(starts):
        seg = x[s : s + n_fft]
        mag = np.abs(np.fft.rfft(np.pad(seg, (0, n_fft - len(seg))) * window))[sel]
        np.add.at(out[:, i], classes, mag)
    return (starts + n_fft / 2) / sr, out


def best_chord(profile: np.ndarray, qualities: Sequence[str] = tuple(CHORD_TEMPLATES)) -> tuple[str, float]:
    """The chord name ('A#m', 'D5', 'Ebmaj7' spelled with sharps) whose template's cosine with
    a 12-bin pitch-class `profile` is the largest, and that cosine."""
    p = np.asarray(profile, dtype=float)
    p = p / max(float(np.linalg.norm(p)), 1e-12)
    best, score = "", -1.0
    for root in range(12):
        for q in qualities:
            tmpl = np.zeros(12)
            tmpl[[(root + k) % 12 for k in CHORD_TEMPLATES[q]]] = 1.0
            tmpl[root] = 1.5  # the root carries the most energy in a guitar chord
            s = float(p @ tmpl / np.linalg.norm(tmpl))
            if s > score:
                best, score = f"{NAMES[root]}{q}", s
    return best, score
