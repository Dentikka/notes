"""Interpretation of the music algebra: flatten a tree into timed events (Hudak's `perform`).

Each node is interpreted in its own local time starting at 0; parents place, filter or
transform the events of their children. Structure leaves two traces in the result:
section markers from `named()` and a path on every event, so each note can be traced back
to the phrase it came from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from notes.lang.music import (
    Humanize,
    Invert,
    Legato,
    Modify,
    Music,
    Mute,
    Note,
    OnTrack,
    Op,
    Par,
    Params,
    Repeat,
    Rest,
    Reverse,
    Seq,
    Slice,
    Stretch,
    Swing,
    Tag,
    Transpose,
    Velocity,
)

if TYPE_CHECKING:
    from notes.lang.song import Track

__all__ = ["Performance", "Performed", "perform"]

#: Humanised onsets snap to 1/1920 beat (about 0.26 ms at 120 bpm) to stay exact rationals.
_HUMANIZE_GRID = 1920

_Marker = tuple[str, Fraction, Fraction]


@dataclass(slots=True)
class Performed:
    """A mutable event while the tree is being interpreted."""

    time: Fraction
    dur: Fraction
    pitch: float
    vel: float
    track: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    path: tuple[str, ...] = ()

    @property
    def end(self) -> Fraction:
        return self.time + self.dur

    @property
    def pitched(self) -> bool:
        return "sound" not in self.params

    def copy(self) -> Performed:
        return Performed(self.time, self.dur, self.pitch, self.vel, self.track, dict(self.params), self.path)


@dataclass(slots=True)
class Performance:
    events: list[Performed]
    markers: list[_Marker]
    tracks: dict[str, Track]


def perform(music: Music) -> Performance:
    """Flatten `music` into events sorted by time, plus section markers and the tracks met."""
    tracks: dict[str, Track] = {}
    events, markers = _run(music, tracks)
    events.sort(key=lambda e: (e.time, e.track or "", e.pitch, -e.vel))
    markers.sort(key=lambda m: (m[1], -m[2], m[0]))
    return Performance(events, markers, tracks)


def _shift(events: list[Performed], markers: list[_Marker], t: Fraction) -> None:
    if t:
        for e in events:
            e.time += t
        markers[:] = [(n, a + t, b + t) for n, a, b in markers]


def _run(m: Music, tracks: dict[str, Track]) -> tuple[list[Performed], list[_Marker]]:
    if isinstance(m, Note):
        return [Performed(Fraction(0), m.dur, m.pitch, m.vel, None, dict(m.params))], []
    if isinstance(m, Rest):
        return [], []
    if isinstance(m, Seq | Par):
        events: list[Performed] = []
        markers: list[_Marker] = []
        t = Fraction(0)
        for item in m.items:
            e, k = _run(item, tracks)
            if isinstance(m, Seq):
                _shift(e, k, t)
                t += item.dur
            events += e
            markers += k
        return events, markers
    if isinstance(m, Repeat):
        cache: dict[int, tuple[list[Performed], list[_Marker]]] = {}
        events, markers, t = [], [], Fraction(0)
        for i in range(m.times):
            body = m.at(i)
            if id(body) not in cache:
                cache[id(body)] = _run(body, tracks)
            e0, k0 = cache[id(body)]
            e, k = [x.copy() for x in e0], list(k0)
            _shift(e, k, t)
            events += e
            markers += k
            t += body.dur
        return events, markers
    if isinstance(m, Modify):
        e, k = _run(m.body, tracks)
        return _apply(m.op, e, k, m.body.dur, tracks)
    raise TypeError(f"Not a music value: {type(m).__name__}")


def _apply(
    op: Op, events: list[Performed], markers: list[_Marker], span: Fraction, tracks: dict[str, Track]
) -> tuple[list[Performed], list[_Marker]]:
    """Give meaning to one operation; `span` is the duration of the subtree it wraps."""
    if isinstance(op, Transpose):
        for e in events:
            if e.pitched:
                e.pitch += op.semitones
    elif isinstance(op, Invert):
        for e in events:
            if e.pitched:
                e.pitch = 2 * op.axis - e.pitch
    elif isinstance(op, Stretch):
        for e in events:
            e.time *= op.factor
            e.dur *= op.factor
        markers = [(n, a * op.factor, b * op.factor) for n, a, b in markers]
    elif isinstance(op, Velocity):
        for e in events:
            e.vel = min(1.0, max(0.0, e.vel * op.scale))
    elif isinstance(op, Legato):
        for e in events:
            e.dur *= op.factor
    elif isinstance(op, Params):
        for e in events:
            for key, value in op.items:
                e.params.setdefault(key, value)
    elif isinstance(op, OnTrack):
        known = tracks.get(op.track.name)
        if known is not None and known != op.track:
            raise ValueError(f"Two different tracks are named {op.track.name!r}; give them distinct names")
        tracks[op.track.name] = op.track
        for e in events:
            if e.track is None:
                e.track = op.track.name
    elif isinstance(op, Tag):
        for e in events:
            e.path = (op.name, *e.path)
        markers = [(op.name, Fraction(0), span), *markers]
    elif isinstance(op, Mute):
        events = [e for e in events if not any(a <= e.time < b for a, b in op.spans)]
    elif isinstance(op, Slice):
        kept = []
        for e in events:
            if op.start <= e.time < op.end:
                e.dur = min(e.dur, op.end - e.time)
                e.time -= op.start
                kept.append(e)
        events = kept
        markers = [
            (n, max(a, op.start) - op.start, min(b, op.end) - op.start)
            for n, a, b in markers
            if a < op.end and b > op.start
        ]
    elif isinstance(op, Reverse):
        for e in events:
            e.time = span - e.end
        markers = [(n, span - b, span - a) for n, a, b in markers]
    elif isinstance(op, Swing):
        delay = op.grid * (2 * op.ratio - 1)
        for e in events:
            q = e.time / op.grid
            if q.denominator == 1 and q.numerator % 2 == 1:
                e.time += delay
    elif isinstance(op, Humanize):
        rng = random.Random(op.seed)
        for e in sorted(events, key=lambda x: (x.time, x.pitch)):
            jitter = Fraction(round(rng.uniform(-op.time, op.time) * _HUMANIZE_GRID), _HUMANIZE_GRID)
            e.time = max(Fraction(0), e.time + jitter)
            e.vel = min(1.0, max(0.0, e.vel * (1.0 + rng.uniform(-op.vel, op.vel))))
    else:
        raise TypeError(f"Unknown operation {type(op).__name__}")
    return events, markers
