"""Tracks (an instrument with its mixer strip) and the Song (arrangement plus global settings)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from notes.engine.base import Effect, Instrument
from notes.ir.score import Marker, NoteEvent, Score, TrackSpec
from notes.lang.music import BeatsLike, Music, beats, cat, parse_meter, stack
from notes.lang.perform import perform

if TYPE_CHECKING:
    from notes.engine.mixer import RenderResult

logger = logging.getLogger(__name__)

__all__ = ["Song", "Track"]


@dataclass(frozen=True)
class Track:
    """An instrument plus its mixer strip; calling the track on music binds the music to it.

        guitar = Track("guitar", ElectricGuitar(drive=0.7), fx=[Reverb(mix=0.2)], gain=-6, pan=0.3)
        riff = guitar(melody("E2 G2 A2 _"))
    """

    name: str
    instrument: Instrument
    fx: tuple[Effect, ...] = ()
    gain: float = 0.0
    pan: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "fx", tuple(self.fx))
        if not self.name:
            raise ValueError("A track needs a name")
        if not -1.0 <= self.pan <= 1.0:
            raise ValueError(f"pan must be in [-1, 1], got {self.pan}")

    def __call__(self, *music: Music | Iterable[Music]) -> Music:
        return cat(*music).on(self)

    def doubled(self, *music: Music | Iterable[Music], spread: float = 0.8, drift: float = 0.01) -> Music:
        """Double-track a part: two takes on copies of this track panned to -/+`spread`.

        The takes are tracks `<name>_L` and `<name>_R`, so their string noise differs (seeds
        hash the track name), and each gets its own timing jitter of up to `drift` beats.
        """
        part = cat(*music)
        takes = [
            replace(self, name=f"{self.name}_{side}", pan=sign * spread)(part.humanize(time=drift, vel=0.05, seed=seed))
            for side, sign, seed in (("L", -1.0, 1), ("R", 1.0, 2))
        ]
        return stack(*takes)

    def spec(self) -> TrackSpec:
        return TrackSpec(
            name=self.name,
            instrument=self.instrument.to_spec(),
            fx=tuple(f.to_spec() for f in self.fx),
            gain_db=float(self.gain),
            pan=float(self.pan),
        )


def _default_track() -> Track:
    from notes.engine.instruments.fm import EPiano

    return Track("main", EPiano())


class Song:
    """A piece: its parts played in order, plus tempo, meter, key and render settings.

        song = Song(intro, verse, chorus * 2, outro, bpm=112, key="A minor")
        song.render("song.wav")

    `loudness` is the integrated-loudness target of the master in LUFS (None keeps the raw
    level); `ceiling` is the peak limit in dBFS. Notes bound to no track play on `default`.
    """

    def __init__(
        self,
        *parts: Music | Iterable[Music],
        bpm: float = 120.0,
        meter: str | tuple[int, int] = "4/4",
        key: str | None = None,
        title: str | None = None,
        tempo: Sequence[tuple[BeatsLike, float]] | None = None,
        seed: int = 0,
        sample_rate: int = 44100,
        a4: float = 440.0,
        loudness: float | None = -14.0,
        ceiling: float = -1.0,
        default: Track | None = None,
    ) -> None:
        self.music = cat(*parts)
        self.meter = parse_meter(meter)
        self.tempo = tuple((beats(b), float(v)) for b, v in (tempo or [(0, bpm)]))
        self.key = key
        self.title = title
        self.seed = seed
        self.sample_rate = sample_rate
        self.a4 = a4
        self.loudness = loudness
        self.ceiling = ceiling
        self.default = default

    def __repr__(self) -> str:
        return f"Song(title={self.title!r}, beats={self.music.dur}, tempo={self.tempo[0][1]:g})"

    def compile(self) -> Score:
        """Interpret the arrangement into the intermediate representation."""
        if self.music.dur == 0:
            raise ValueError("The song is empty: pass its parts, e.g. Song(intro, verse, chorus * 2)")
        perf = perform(self.music)
        tracks = dict(perf.tracks)
        loose = [e for e in perf.events if e.track is None]
        if loose:
            fallback = self.default or _default_track()
            if tracks.get(fallback.name, fallback) != fallback:
                raise ValueError(f"The default track name {fallback.name!r} is taken by another track")
            logger.info("%d notes are bound to no track; they play on %r", len(loose), fallback.name)
            tracks[fallback.name] = fallback
            for e in loose:
                e.track = fallback.name
        events = tuple(
            NoteEvent(
                time=e.time,
                dur=e.dur,
                pitch=float(e.pitch),
                vel=float(e.vel),
                track=str(e.track),
                params=tuple(sorted(e.params.items())),
                path="/".join(e.path),
            )
            for e in perf.events
        )
        return Score(
            events=events,
            tracks=tuple(t.spec() for t in tracks.values()),
            length=self.music.dur,
            tempo=self.tempo,
            meter=self.meter,
            markers=tuple(Marker(n, a, b) for n, a, b in perf.markers),
            key=self.key,
            title=self.title,
            sample_rate=self.sample_rate,
            seed=self.seed,
            a4=self.a4,
            master={"loudness": self.loudness, "ceiling": self.ceiling},
        )

    def render(self, path: str | Path | None = None, *, bits: int = 16) -> RenderResult:
        """Render to audio; also writes a WAV file when `path` is given."""
        from notes.engine.mixer import render_score

        result = render_score(self.compile())
        if path is not None:
            result.write(path, bits=bits)
        return result

    def export_midi(self, path: str | Path) -> Path:
        from notes.export.midi import write_midi

        return write_midi(self.compile(), path)

    def pianoroll(self, path: str | Path) -> Path:
        from notes.viz import pianoroll

        return pianoroll(self.compile(), path)

    def describe(self) -> str:
        return self.compile().summary()
