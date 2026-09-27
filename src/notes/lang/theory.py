"""Pitch, scale and chord arithmetic — the number theory underneath the notation.

Pitches are MIDI note numbers (C4 = 60, A4 = 69); fractional values are microtones.
Scales are interval sets modulo the octave, chords are interval stacks over a root, and
a `Key` turns scale degrees — including degrees outside one octave — into pitches.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from notes.lang.music import BeatsLike, Music

__all__ = [
    "CHORD_QUALITIES",
    "SCALES",
    "Key",
    "chord_pitches",
    "midi_to_hz",
    "parse_chord",
    "parse_pitch",
    "pitch_name",
    "voice_lead",
]

_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_PITCH_RE = re.compile(r"^([A-Ga-g])(#{1,2}|b{1,2})?(-?\d+)?$")
_ROOT_RE = re.compile(r"^([A-G])(#|b)?")

SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
    "chromatic": tuple(range(12)),
}
_MODE_ALIASES = {
    "": "major",
    "maj": "major",
    "ionian": "major",
    "m": "minor",
    "min": "minor",
    "aeolian": "minor",
    "pentatonic": "major_pentatonic",
}

CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "maj": (0, 4, 7),
    "M": (0, 4, 7),
    "m": (0, 3, 7),
    "min": (0, 3, 7),
    "-": (0, 3, 7),
    "dim": (0, 3, 6),
    "°": (0, 3, 6),
    "aug": (0, 4, 8),
    "+": (0, 4, 8),
    "5": (0, 7, 12),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "sus": (0, 5, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "min7": (0, 3, 7, 10),
    "mmaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9),
    "°7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "ø": (0, 3, 6, 10),
    "7sus4": (0, 5, 7, 10),
    "add9": (0, 4, 7, 14),
    "madd9": (0, 3, 7, 14),
    "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
    "m9": (0, 3, 7, 10, 14),
    "11": (0, 4, 7, 10, 14, 17),
    "m11": (0, 3, 7, 10, 14, 17),
    "13": (0, 4, 7, 10, 14, 21),
}
_CHORD_RE = re.compile(r"^([A-G])(#|b)?(.*?)(?:/([A-G](?:#|b)?))?$")

_ROMAN_RE = re.compile(
    r"^(#|b)?(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)(°|o|dim|\+|aug|ø)?(maj7|M7|7|6|9|sus2|sus4)?$"
)
_ROMAN_VALUES = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}


def parse_pitch(token: str | float, octave: int = 4) -> float:
    """'C4' -> 60.0, 'F#3' -> 54.0, 'Bb2' -> 46.0; numbers pass through as MIDI pitches.

    A name without an octave ('E', 'Bb') is placed in `octave`.
    """
    if isinstance(token, int | float) and not isinstance(token, bool):
        return float(token)
    m = _PITCH_RE.match(str(token).strip())
    if m is None:
        raise ValueError(f"Not a pitch: {token!r} (expected a name like 'C4', 'F#3', 'Bb2' or a MIDI number)")
    letter, acc, octv = m.groups()
    acc = acc or ""
    pc = _LETTERS[letter.upper()] + acc.count("#") - acc.count("b")
    return float(12 * ((int(octv) if octv is not None else octave) + 1) + pc)


def pitch_name(pitch: float) -> str:
    """60 -> 'C4'; microtonal pitches get a cents suffix: 60.5 -> 'C4+50c'."""
    n = round(pitch)
    cents = round((pitch - n) * 100)
    name = f"{_SHARP_NAMES[n % 12]}{n // 12 - 1}"
    return name if cents == 0 else f"{name}{cents:+d}c"


def midi_to_hz(pitch: float, a4: float = 440.0) -> float:
    """Twelve-tone equal temperament; `a4` sets the reference (440 Hz by default)."""
    return a4 * 2.0 ** ((pitch - 69.0) / 12.0)


def _root_pc(text: str) -> int:
    m = _ROOT_RE.match(text)
    if m is None:
        raise ValueError(f"Not a note name: {text!r}")
    letter, acc = m.groups()
    return (_LETTERS[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)) % 12


def parse_chord(symbol: str) -> tuple[int, tuple[int, ...], int | None]:
    """'Am7' -> (9, (0, 3, 7, 10), None); 'C/E' -> (0, (0, 4, 7), 4)."""
    m = _CHORD_RE.match(symbol.strip())
    if m is None:
        raise ValueError(f"Not a chord symbol: {symbol!r}")
    letter, acc, quality, bass = m.groups()
    if quality not in CHORD_QUALITIES:
        known = " ".join(q or "(major)" for q in CHORD_QUALITIES)
        raise ValueError(f"Unknown chord quality {quality!r} in {symbol!r}. Known: {known}")
    root = _root_pc(letter + (acc or ""))
    return root, CHORD_QUALITIES[quality], None if bass is None else _root_pc(bass)


def chord_pitches(symbol: str, octave: int = 3) -> list[float]:
    """Close-position pitches of a chord symbol with its root in `octave`; a slash bass goes below."""
    root, intervals, bass = parse_chord(symbol)
    base = 12 * (octave + 1) + root
    pitches = [float(base + i) for i in intervals]
    if bass is not None:
        low = 12 * (octave + 1) + bass
        while low >= pitches[0]:
            low -= 12
        pitches.insert(0, float(low))
    return pitches


def _voicings(chord: Sequence[float], around: float) -> list[list[float]]:
    base = sorted(chord)
    found = []
    for k in range(len(base)):
        inverted = base[k:] + [p + 12 for p in base[:k]]
        for shift in range(-3, 4):
            v = [p + 12 * shift for p in inverted]
            if abs(fmean(v) - around) <= 9:
                found.append(v)
    return found or [base]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) == len(b):
        return sum(abs(x - y) for x, y in zip(sorted(a), sorted(b), strict=True))
    return sum(min(abs(x - y) for x in a) for y in b) + sum(min(abs(x - y) for y in b) for x in a)


def voice_lead(chords: Sequence[Sequence[float]]) -> list[list[float]]:
    """Re-voice chords (inversion + octave) so each moves as little as possible from the last.

    Uses the L1 distance between sorted voicings — the optimal voice assignment on a line —
    plus a weak pull toward the register of the first chord to prevent drift.
    """
    if not chords:
        return []
    out = [sorted(chords[0])]
    center = fmean(out[0])
    for chord in chords[1:]:
        prev = out[-1]
        best = min(_voicings(chord, fmean(prev)), key=lambda v: _distance(prev, v) + 0.1 * abs(fmean(v) - center))
        out.append(best)
    return out


@dataclass(frozen=True)
class Key:
    """A tonic and a mode; maps scale degrees and roman numerals to pitches."""

    tonic: int
    mode: str = "major"

    def __post_init__(self) -> None:
        if self.mode not in SCALES:
            raise ValueError(f"Unknown mode {self.mode!r}. Known: {', '.join(SCALES)}")
        object.__setattr__(self, "tonic", self.tonic % 12)

    @classmethod
    def parse(cls, text: str) -> Key:
        """'A minor', 'F# dorian', 'Bb', 'Am', 'C harmonic minor'."""
        m = re.match(r"^\s*([A-G](?:#|b)?)\s*(.*)$", text)
        if m is None:
            raise ValueError(f"Not a key: {text!r} (expected e.g. 'A minor', 'F# dorian')")
        mode = m.group(2).strip().lower().replace(" ", "_")
        return cls(_root_pc(m.group(1)), _MODE_ALIASES.get(mode, mode))

    @property
    def intervals(self) -> tuple[int, ...]:
        return SCALES[self.mode]

    def __str__(self) -> str:
        return f"{_SHARP_NAMES[self.tonic]} {self.mode.replace('_', ' ')}"

    def degree(self, d: int, octave: int = 4, alter: int = 0) -> float:
        """Pitch of scale degree `d`: 1 is the tonic in `octave`, 8 the next octave, 0 the step below 1."""
        octaves, idx = divmod(d - 1, len(self.intervals))
        return float(12 * (octave + 1 + octaves) + self.tonic + self.intervals[idx] + alter)

    def pitches(self, octave: int = 4, count: int | None = None) -> list[float]:
        """Scale pitches upward from the tonic (one octave by default)."""
        n = len(self.intervals) + 1 if count is None else count
        return [self.degree(d, octave) for d in range(1, n + 1)]

    def contains(self, pitch: float) -> bool:
        return (round(pitch) - self.tonic) % 12 in self.intervals

    def roman(self, numeral: str, octave: int = 3) -> list[float]:
        """Chord for a roman numeral: case sets major/minor ('V' vs 'v'), '°'/'+' dim/aug, '7' adds a seventh."""
        m = _ROMAN_RE.match(numeral.strip())
        if m is None:
            raise ValueError(f"Not a roman numeral: {numeral!r} (e.g. 'i', 'IV', 'V7', 'bVII', 'vii°')")
        acc, num, quality, ext = m.groups()
        root = self.degree(_ROMAN_VALUES[num.lower()], octave) + (1 if acc == "#" else -1 if acc == "b" else 0)
        if quality in ("°", "o", "dim", "ø"):
            third, fifth = 3, 6
        elif quality in ("+", "aug"):
            third, fifth = 4, 8
        else:
            third, fifth = (3, 7) if num.islower() else (4, 7)
        intervals = [0, third, fifth]
        if ext == "sus2":
            intervals[1] = 2
        elif ext == "sus4":
            intervals[1] = 5
        elif ext == "6":
            intervals.append(9)
        elif ext in ("maj7", "M7"):
            intervals.append(11)
        elif ext in ("7", "9"):
            intervals.append(9 if quality in ("°", "o", "dim") else 10)
            if ext == "9":
                intervals.append(14)
        elif quality == "ø":
            intervals.append(10)
        return [root + i for i in intervals]

    def melody(self, text: str, step: BeatsLike = "1/2", *, vel: float = 0.8, octave: int = 4) -> Music:
        """Mini-notation over scale degrees: 'b3' flattens, '#4' sharpens, 8 is the octave."""
        from notes.lang.notation import melody

        def resolve(token: str) -> float:
            m = re.match(r"^(#|b)?(-?\d+)$", token)
            if m is None:
                raise ValueError(f"Not a scale degree: {token!r} (use 1..7, 8+ for higher octaves, b3/#4)")
            alter = 1 if m.group(1) == "#" else -1 if m.group(1) == "b" else 0
            return self.degree(int(m.group(2)), octave, alter)

        return melody(text, step, vel=vel, resolve=resolve)

    def chords(
        self, text: str, dur: BeatsLike = 4, *, octave: int = 3, vel: float = 0.7, voicing: str = "close"
    ) -> Music:
        """A progression in roman numerals, one chord per `dur` beats: 'i VI III VII'."""
        from notes.lang.notation import chords

        return chords(text, dur, octave=octave, vel=vel, voicing=voicing, resolve=lambda s: self.roman(s, octave))
