"""The music algebra: exact-time trees built with ``+`` (then) and ``|`` (together).

Leaves are `Note` and `Rest`. `Seq` plays its items one after another, `Par` plays them
at once, `Repeat` loops a body (optionally varying some repetitions), and `Modify`
attaches an operation — transpose, mute, bind to a track, name a section — to a subtree.
Values are immutable, so a phrase can be reused, varied and recombined freely. Durations
are exact rationals in beats (a quarter note is 1), which keeps triplets, polyrhythms and
long forms free of rounding drift. The tree keeps its structure (``chorus * 2`` stays a
`Repeat`) until `notes.lang.perform` flattens it into IR events.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from notes.lang.theory import parse_pitch

if TYPE_CHECKING:
    from notes.lang.song import Track

__all__ = [
    "BeatsLike",
    "E",
    "H",
    "Modify",
    "Music",
    "Note",
    "Par",
    "Q",
    "Repeat",
    "Rest",
    "S",
    "Seq",
    "T",
    "W",
    "bars",
    "beats",
    "cat",
    "dotted",
    "parse_meter",
    "place",
    "stack",
    "triplet",
]

BeatsLike = int | float | Fraction | str
_MAX_DENOMINATOR = 1 << 20


def beats(x: BeatsLike) -> Fraction:
    """Coerce a duration to exact beats: 1 is a quarter note; floats snap to simple rationals."""
    if isinstance(x, bool):
        raise TypeError(f"Not a duration: {x!r}")
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(x).limit_denominator(_MAX_DENOMINATOR)
    if isinstance(x, str):
        return Fraction(x.strip())
    raise TypeError(f"Not a duration: {x!r} (use beats: 1 = quarter note, or W H Q E S T)")


#: Note values in beats: whole, half, quarter, eighth, sixteenth, thirty-second.
W, H, Q, E, S, T = Fraction(4), Fraction(2), Fraction(1), Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)


def dotted(d: BeatsLike, dots: int = 1) -> Fraction:
    return beats(d) * (2 - Fraction(1, 2**dots))


def triplet(d: BeatsLike) -> Fraction:
    return beats(d) * Fraction(2, 3)


def parse_meter(meter: str | tuple[int, int]) -> tuple[int, int]:
    """'6/8' or (6, 8) -> (6, 8)."""
    if isinstance(meter, str):
        num, _, den = meter.partition("/")
        return int(num), int(den)
    return int(meter[0]), int(meter[1])


def bars(n: BeatsLike, meter: str | tuple[int, int] = "4/4") -> Fraction:
    """Length of `n` bars in beats: bars(2) == 8, bars(1, '6/8') == 3."""
    num, den = parse_meter(meter)
    return beats(n) * Fraction(4 * num, den)


def _freeze_params(params: Mapping[str, Any] | Iterable[tuple[str, Any]] | None) -> tuple[tuple[str, Any], ...]:
    if not params:
        return ()
    items = params.items() if isinstance(params, Mapping) else params
    return tuple(sorted((str(k), v) for k, v in items))


class Music:
    """Base of every musical value: an immutable tree with an exact duration `dur` in beats."""

    __slots__ = ()
    dur: Fraction

    # --- composition ------------------------------------------------------------------
    def __add__(self, other: Music) -> Music:
        return cat(self, other) if isinstance(other, Music) else NotImplemented

    def __or__(self, other: Music) -> Music:
        return stack(self, other) if isinstance(other, Music) else NotImplemented

    def __mul__(self, n: int) -> Music:
        if isinstance(n, bool) or not isinstance(n, int):
            return NotImplemented
        return self.repeat(n)

    __rmul__ = __mul__

    def repeat(self, n: int) -> Repeat:
        return Repeat(self, n)

    def every(self, n: int, f: Callable[[Music], Music] | Music, *, offset: int | None = None) -> Music:
        raise TypeError("every() varies repetitions; build them first: (phrase * 8).every(4, fill)")

    def loop(self, length: BeatsLike) -> Music:
        """Repeat to fill exactly `length` beats; the last copy is cut if needed."""
        total = beats(length)
        if self.dur <= 0:
            raise ValueError("Cannot loop a phrase of zero length")
        k = max(1, math.ceil(total / self.dur))
        looped = self.repeat(k)
        return looped if k * self.dur == total else looped.slice(0, total)

    def named(self, name: str) -> Music:
        """Label a section; it becomes a marker in the IR and a path prefix on its notes."""
        return Modify(self, Tag(name))

    def on(self, track: Track) -> Music:
        """Bind to a track; an inner binding wins over an outer one."""
        return Modify(self, OnTrack(track))

    # --- time -------------------------------------------------------------------------
    def stretch(self, factor: BeatsLike) -> Music:
        k = beats(factor)
        if k <= 0:
            raise ValueError("stretch() needs a positive factor")
        return Modify(self, Stretch(k))

    def fast(self, factor: BeatsLike) -> Music:
        return self.stretch(1 / beats(factor))

    def slow(self, factor: BeatsLike) -> Music:
        return self.stretch(factor)

    def shift(self, offset: BeatsLike) -> Music:
        """Delay the start by `offset` beats (prepends a rest)."""
        return cat(Rest(offset), self)

    def reverse(self) -> Music:
        """Retrograde: the phrase played backwards in time."""
        return Modify(self, Reverse())

    def slice(self, start: BeatsLike = 0, end: BeatsLike | None = None) -> Music:
        """Keep notes starting in [start, end) beats; notes crossing `end` are cut."""
        a, b = beats(start), self.dur if end is None else beats(end)
        if not 0 <= a <= b:
            raise ValueError(f"slice() needs 0 <= start <= end, got {a}, {b}")
        return Modify(self, Slice(a, b))

    def mute(
        self,
        bars: int | Iterable[int] | None = None,
        *,
        window: tuple[BeatsLike, BeatsLike] | Iterable[tuple[BeatsLike, BeatsLike]] | None = None,
        meter: str | tuple[int, int] = "4/4",
    ) -> Music:
        """Silence notes that start in the given bars (numbered from 1) or beat windows."""
        spans: list[tuple[Fraction, Fraction]] = []
        if bars is not None:
            num, den = parse_meter(meter)
            size = Fraction(4 * num, den)
            for k in [bars] if isinstance(bars, int) else bars:
                if k < 1:
                    raise ValueError("Bars are numbered from 1")
                spans.append(((k - 1) * size, k * size))
        if window is not None:
            windows = list(window)
            if windows and not isinstance(windows[0], tuple | list):
                windows = [tuple(windows)]
            spans += [(beats(a), beats(b)) for a, b in windows]
        if not spans:
            raise ValueError("mute() needs bars=... or window=(start, end)")
        return Modify(self, Mute(tuple(spans)))

    def swing(self, ratio: BeatsLike = Fraction(2, 3), grid: BeatsLike = Fraction(1, 2)) -> Music:
        """Delay every second `grid` step so each pair splits ratio : 1 - ratio (2/3 = triplet swing)."""
        r = beats(ratio)
        if not Fraction(1, 2) <= r < 1:
            raise ValueError("swing ratio must be in [1/2, 1)")
        return Modify(self, Swing(r, beats(grid)))

    def humanize(self, time: float = 0.01, vel: float = 0.08, seed: int = 0) -> Music:
        """Seeded jitter: onsets by up to `time` beats, velocities by up to ±`vel` (relative)."""
        return Modify(self, Humanize(float(time), float(vel), int(seed)))

    # --- pitch ------------------------------------------------------------------------
    def transpose(self, semitones: float) -> Music:
        return Modify(self, Transpose(float(semitones)))

    def octave(self, n: int) -> Music:
        return self.transpose(12 * n)

    def invert(self, axis: str | float) -> Music:
        """Mirror pitches around `axis` (melodic inversion)."""
        return Modify(self, Invert(parse_pitch(axis)))

    # --- dynamics and articulation ----------------------------------------------------
    def velocity(self, scale: float) -> Music:
        return Modify(self, Velocity(float(scale)))

    def legato(self, factor: BeatsLike) -> Music:
        """Scale note lengths, keeping onsets: < 1 is staccato, > 1 overlaps."""
        return Modify(self, Legato(beats(factor)))

    def param(self, **params: Any) -> Music:
        """Attach per-note parameters for the instrument (e.g. palm_mute=True); inner values win."""
        return Modify(self, Params(_freeze_params(params)))

    # --- re-articulation --------------------------------------------------------------
    def rhythm(self, pattern: str, step: BeatsLike = Fraction(1, 2)) -> Music:
        """Re-strike whatever sounds at each hit of a step pattern — held chords become a riff.

        The pattern uses the `steps` grammar and repeats over the phrase. Only pitched notes are
        taken and track bindings are dropped: apply it to raw harmony, then bind a track.
        """
        from notes.lang.notation import parse_hits
        from notes.lang.perform import perform

        st = beats(step)
        hits, cells = parse_hits(pattern)
        if cells == 0:
            raise ValueError("Empty rhythm pattern")
        sounding = [e for e in perform(self).events if e.pitched]
        placed: list[tuple[Fraction, Music]] = []
        cycle_start = Fraction(0)
        while cycle_start < self.dur:
            for start, length, vel in hits:
                t = cycle_start + start * st
                if t >= self.dur:
                    break
                d = min(length * st, self.dur - t)
                for e in sounding:
                    if e.time <= t < e.end:
                        placed.append((t, Note(e.pitch, d, min(1.0, e.vel * vel / 0.8), e.params)))
            cycle_start += cells * st
        return place(placed, self.dur)

    def arp(self, mode: str = "up", step: BeatsLike = Fraction(1, 4), octaves: int = 1) -> Music:
        """Arpeggiate each chord ('up', 'down', 'updown', 'downup') over its own duration."""
        from notes.lang.perform import perform

        st = beats(step)
        groups: dict[Fraction, list] = {}
        for e in perform(self).events:
            if e.pitched:
                groups.setdefault(e.time, []).append(e)
        placed: list[tuple[Fraction, Music]] = []
        for start in sorted(groups):
            chord = groups[start]
            base = sorted({e.pitch for e in chord})
            tones = [p + 12 * k for k in range(octaves) for p in base]
            orders = {
                "up": tones,
                "down": tones[::-1],
                "updown": tones + tones[-2:0:-1],
                "downup": tones[::-1] + tones[1:-1],
            }
            if mode not in orders:
                raise ValueError(f"arp mode must be one of {', '.join(orders)}")
            order, vel, end = orders[mode], max(e.vel for e in chord), max(e.end for e in chord)
            t, i = start, 0
            while t < end:
                placed.append((t, Note(order[i % len(order)], min(st, end - t), vel)))
                t, i = t + st, i + 1
        return place(placed, self.dur)


@dataclass(frozen=True)
class Note(Music):
    """A single pitched (or drum) note; `pitch` accepts names ('A4') or MIDI numbers."""

    pitch: float
    dur: Fraction
    vel: float = 0.8
    params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pitch", parse_pitch(self.pitch))
        object.__setattr__(self, "dur", beats(self.dur))
        object.__setattr__(self, "params", _freeze_params(self.params))
        if self.dur <= 0:
            raise ValueError(f"A note needs a positive duration, got {self.dur}")
        if not 0.0 <= self.vel <= 1.0:
            raise ValueError(f"Velocity must be in [0, 1], got {self.vel}")


@dataclass(frozen=True)
class Rest(Music):
    dur: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "dur", beats(self.dur))
        if self.dur < 0:
            raise ValueError("A rest cannot be negative")


@dataclass(frozen=True)
class Seq(Music):
    items: tuple[Music, ...]
    dur: Fraction = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dur", sum((m.dur for m in self.items), Fraction(0)))


@dataclass(frozen=True)
class Par(Music):
    items: tuple[Music, ...]
    dur: Fraction = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dur", max((m.dur for m in self.items), default=Fraction(0)))


@dataclass(frozen=True)
class Repeat(Music):
    """`body` played `times` times; `variants` replace chosen repetitions (0-based)."""

    body: Music
    times: int
    variants: tuple[tuple[int, Music], ...] = ()
    dur: Fraction = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.times < 1:
            raise ValueError("repeat() needs at least one repetition")
        variants = dict(self.variants)
        total = sum((variants.get(i, self.body).dur for i in range(self.times)), Fraction(0))
        object.__setattr__(self, "dur", total)

    def at(self, i: int) -> Music:
        """The music played on repetition `i` (0-based)."""
        return dict(self.variants).get(i, self.body)

    def every(self, n: int, f: Callable[[Music], Music] | Music, *, offset: int | None = None) -> Repeat:
        """Transform (or replace) every n-th repetition; by default the last of each group of n."""
        if n < 1:
            raise ValueError("every() needs n >= 1")
        g = f if callable(f) else (lambda _: f)
        target = n - 1 if offset is None else offset % n
        variants = dict(self.variants)
        for i in range(self.times):
            if i % n == target:
                variants[i] = g(variants.get(i, self.body))
        return Repeat(self.body, self.times, tuple(sorted(variants.items())))


@dataclass(frozen=True)
class Modify(Music):
    body: Music
    op: Op
    dur: Fraction = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dur", self.op.map_dur(self.body.dur))


def _collect(parts: tuple[Music | Iterable[Music], ...]) -> list[Music]:
    out: list[Music] = []
    for p in parts:
        if isinstance(p, Music):
            out.append(p)
        else:
            out.extend(_collect(tuple(p)))
    return out


def cat(*parts: Music | Iterable[Music]) -> Music:
    """Sequential composition: parts one after another (same as ``a + b + c``)."""
    flat: list[Music] = []
    for m in _collect(parts):
        flat.extend(m.items if type(m) is Seq else (m,))
    if not flat:
        return Rest(0)
    return flat[0] if len(flat) == 1 else Seq(tuple(flat))


def stack(*parts: Music | Iterable[Music]) -> Music:
    """Parallel composition: parts at once (same as ``a | b | c``)."""
    flat: list[Music] = []
    for m in _collect(parts):
        flat.extend(m.items if type(m) is Par else (m,))
    if not flat:
        return Rest(0)
    return flat[0] if len(flat) == 1 else Par(tuple(flat))


def place(items: Iterable[tuple[BeatsLike, Music]], length: BeatsLike | None = None) -> Music:
    """Put phrases at absolute offsets: place([(0, a), (6, b)], length=8)."""
    layers = [m if beats(t) == 0 else cat(Rest(t), m) for t, m in items]
    if length is not None:
        layers.append(Rest(length))
    return stack(*layers)


# --- operations: data attached to Modify nodes; `perform` gives them meaning ------------
class Op:
    """A transformation attached to a subtree."""

    def map_dur(self, d: Fraction) -> Fraction:
        return d


@dataclass(frozen=True)
class Transpose(Op):
    semitones: float


@dataclass(frozen=True)
class Invert(Op):
    axis: float


@dataclass(frozen=True)
class Stretch(Op):
    factor: Fraction

    def map_dur(self, d: Fraction) -> Fraction:
        return d * self.factor


@dataclass(frozen=True)
class Velocity(Op):
    scale: float


@dataclass(frozen=True)
class Legato(Op):
    factor: Fraction


@dataclass(frozen=True)
class Params(Op):
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class OnTrack(Op):
    track: Track


@dataclass(frozen=True)
class Tag(Op):
    name: str


@dataclass(frozen=True)
class Mute(Op):
    spans: tuple[tuple[Fraction, Fraction], ...]


@dataclass(frozen=True)
class Slice(Op):
    start: Fraction
    end: Fraction

    def map_dur(self, d: Fraction) -> Fraction:
        return self.end - self.start


@dataclass(frozen=True)
class Reverse(Op):
    pass


@dataclass(frozen=True)
class Swing(Op):
    ratio: Fraction
    grid: Fraction


@dataclass(frozen=True)
class Humanize(Op):
    time: float
    vel: float
    seed: int
