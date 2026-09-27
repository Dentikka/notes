"""The intermediate representation: a flat, exact, serialisable score.

A `Score` is what the language compiles to and what every back-end consumes. Time is kept
in beats as exact rationals; seconds appear only at render time, through the tempo map.
The IR knows nothing about how sound is made: instruments and effects are plain specs
(``{"type": ..., **params}``) that a back-end resolves through its own registry. In JSON,
integral rationals are numbers and the rest are ``"p/q"`` strings, so triplets and
polyrhythms survive a round trip exactly.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

__all__ = ["IR_VERSION", "Marker", "NoteEvent", "Score", "TempoMap", "TrackSpec"]

IR_VERSION = 1

Spec = dict[str, Any]


def _dump_rational(x: Fraction) -> int | str:
    return int(x) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _load_rational(x: int | float | str) -> Fraction:
    return Fraction(x).limit_denominator(1 << 20) if isinstance(x, float) else Fraction(x)


def _dump_pitch(p: float) -> int | float:
    return int(p) if float(p).is_integer() else p


@dataclass(frozen=True)
class NoteEvent:
    """One sounding note: onset and length in beats, pitch as a (fractional) MIDI number."""

    time: Fraction
    dur: Fraction
    pitch: float
    vel: float
    track: str
    params: tuple[tuple[str, Any], ...] = ()
    path: str = ""

    @property
    def end(self) -> Fraction:
        return self.time + self.dur

    @property
    def pitched(self) -> bool:
        """Drum hits carry a ``sound`` parameter and have no musical pitch."""
        return all(key != "sound" for key, _ in self.params)

    def param(self, key: str, default: Any = None) -> Any:
        return dict(self.params).get(key, default)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "t": _dump_rational(self.time),
            "dur": _dump_rational(self.dur),
            "pitch": _dump_pitch(self.pitch),
            "vel": self.vel,
            "track": self.track,
        }
        if self.params:
            d["params"] = dict(self.params)
        if self.path:
            d["path"] = self.path
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> NoteEvent:
        return cls(
            time=_load_rational(d["t"]),
            dur=_load_rational(d["dur"]),
            pitch=float(d["pitch"]),
            vel=float(d["vel"]),
            track=str(d["track"]),
            params=tuple(sorted(dict(d.get("params", {})).items())),
            path=str(d.get("path", "")),
        )


@dataclass(frozen=True)
class TrackSpec:
    """A mixer channel: which instrument plays the track's notes and what follows it."""

    name: str
    instrument: Spec
    fx: tuple[Spec, ...] = ()
    gain_db: float = 0.0
    pan: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instrument": dict(self.instrument),
            "fx": [dict(f) for f in self.fx],
            "gain_db": self.gain_db,
            "pan": self.pan,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TrackSpec:
        return cls(
            name=str(d["name"]),
            instrument=dict(d["instrument"]),
            fx=tuple(dict(f) for f in d.get("fx", ())),
            gain_db=float(d.get("gain_db", 0.0)),
            pan=float(d.get("pan", 0.0)),
        )


@dataclass(frozen=True)
class Marker:
    """A named span of the timeline — a section such as 'verse' or 'chorus'."""

    name: str
    start: Fraction
    end: Fraction

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "start": _dump_rational(self.start), "end": _dump_rational(self.end)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Marker:
        return cls(str(d["name"]), _load_rational(d["start"]), _load_rational(d["end"]))


class TempoMap:
    """Piecewise-constant tempo: converts beats to seconds."""

    def __init__(self, points: Iterable[tuple[Fraction | int, float]]) -> None:
        pts = sorted((Fraction(b), float(bpm)) for b, bpm in points)
        if not pts or pts[0][0] != 0:
            raise ValueError("A tempo map must define the tempo at beat 0")
        if any(bpm <= 0 for _, bpm in pts):
            raise ValueError("Tempo must be positive")
        self._beats = [float(b) for b, _ in pts]
        self._bpm = [bpm for _, bpm in pts]
        self._secs = [0.0]
        for i in range(1, len(pts)):
            span = self._beats[i] - self._beats[i - 1]
            self._secs.append(self._secs[-1] + span * 60.0 / self._bpm[i - 1])

    def seconds(self, beat: Fraction | float) -> float:
        b = float(beat)
        i = max(bisect_right(self._beats, b) - 1, 0)
        return self._secs[i] + (b - self._beats[i]) * 60.0 / self._bpm[i]

    def bpm_at(self, beat: Fraction | float) -> float:
        return self._bpm[max(bisect_right(self._beats, float(beat)) - 1, 0)]


@dataclass(frozen=True)
class Score:
    """The whole piece as data: events, tracks, sections, tempo and render settings."""

    events: tuple[NoteEvent, ...]
    tracks: tuple[TrackSpec, ...]
    length: Fraction
    tempo: tuple[tuple[Fraction, float], ...] = ((Fraction(0), 120.0),)
    meter: tuple[int, int] = (4, 4)
    markers: tuple[Marker, ...] = ()
    key: str | None = None
    title: str | None = None
    sample_rate: int = 44100
    seed: int = 0
    a4: float = 440.0
    master: Spec = field(default_factory=lambda: {"loudness": -14.0, "ceiling": -1.0})
    version: int = IR_VERSION

    @property
    def beats_per_bar(self) -> Fraction:
        return Fraction(4 * self.meter[0], self.meter[1])

    def tempo_map(self) -> TempoMap:
        return TempoMap(self.tempo)

    def duration_seconds(self) -> float:
        return self.tempo_map().seconds(self.length)

    def track(self, name: str) -> TrackSpec:
        for spec in self.tracks:
            if spec.name == name:
                return spec
        raise KeyError(f"No track {name!r}; tracks: {', '.join(t.name for t in self.tracks)}")

    def events_of(self, track: str) -> list[NoteEvent]:
        return [e for e in self.events if e.track == track]

    # --- serialisation ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "title": self.title,
            "key": self.key,
            "meter": list(self.meter),
            "tempo": [[_dump_rational(b), bpm] for b, bpm in self.tempo],
            "length": _dump_rational(self.length),
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "a4": self.a4,
            "master": dict(self.master),
            "tracks": [t.to_dict() for t in self.tracks],
            "markers": [m.to_dict() for m in self.markers],
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Score:
        if d.get("version", IR_VERSION) != IR_VERSION:
            raise ValueError(f"Unsupported IR version {d.get('version')}; expected {IR_VERSION}")
        return cls(
            events=tuple(NoteEvent.from_dict(e) for e in d["events"]),
            tracks=tuple(TrackSpec.from_dict(t) for t in d["tracks"]),
            length=_load_rational(d["length"]),
            tempo=tuple((_load_rational(b), float(bpm)) for b, bpm in d["tempo"]),
            meter=(int(d["meter"][0]), int(d["meter"][1])),
            markers=tuple(Marker.from_dict(m) for m in d.get("markers", ())),
            key=d.get("key"),
            title=d.get("title"),
            sample_rate=int(d.get("sample_rate", 44100)),
            seed=int(d.get("seed", 0)),
            a4=float(d.get("a4", 440.0)),
            master=dict(d.get("master", {})),
        )

    def to_json(self, path: str | Path | None = None) -> str:
        """Pretty JSON with one event per line; also written to `path` when given."""
        d = self.to_dict()
        events = d.pop("events")
        head = json.dumps(d, indent=2, ensure_ascii=False)
        body = ",\n".join("    " + json.dumps(e, ensure_ascii=False) for e in events)
        text = head[:-2] + ',\n  "events": [\n' + body + ("\n" if body else "") + "  ]\n}\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> Score:
        """Load from a JSON string or from a path to a JSON file."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    # --- digest -----------------------------------------------------------------------
    def summary(self) -> str:
        """A compact human- and agent-readable description of the piece."""
        bpb = self.beats_per_bar

        def bar(beat: Fraction) -> str:
            pos = beat / bpb + 1
            return str(int(pos)) if pos.denominator == 1 else f"{float(pos):.2f}"

        bpm = ", ".join(f"{bpm:g}" for _, bpm in self.tempo)
        head = (
            f"{self.title or 'Untitled'}: {float(self.length / bpb):g} bars of "
            f"{self.meter[0]}/{self.meter[1]} at {bpm} bpm = {self.duration_seconds():.1f} s"
        )
        lines = [head + (f", key {self.key}" if self.key else "")]
        if self.markers:
            spans = [f"{m.name} {bar(m.start)}-{bar(m.end)}" for m in self.markers]
            lines.append("sections (bars, end exclusive): " + ", ".join(spans))
        for spec in self.tracks:
            evs = self.events_of(spec.name)
            fx = ", ".join(f["type"] for f in spec.fx) or "none"
            line = f"track {spec.name}: {spec.instrument.get('type')}, {len(evs)} notes"
            pitched = [e.pitch for e in evs if e.pitched]
            if pitched:
                line += f", pitch {min(pitched):g}-{max(pitched):g}"
            sounds = sorted({str(e.param("sound")) for e in evs if not e.pitched})
            if sounds:
                line += f", sounds {'/'.join(sounds)}"
            lines.append(line + f", gain {spec.gain_db:+g} dB, pan {spec.pan:+g}, fx {fx}")
        return "\n".join(lines)
