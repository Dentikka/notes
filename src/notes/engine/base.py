"""Contracts between the renderer and the sound sources it drives.

Instruments and effects are frozen dataclasses: their fields are the parameters the DSL
exposes, `to_spec()` turns them into IR specs, and the registry turns specs back into
objects. An instrument renders one mono voice per note; an effect processes a
``(channels, samples)`` buffer in place of a track or of the instrument's own inserts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np

__all__ = ["Effect", "Instrument", "RenderContext", "Voice"]


@dataclass(frozen=True)
class RenderContext:
    """Render settings; `offset` is where the buffer being processed starts in the song
    (samples), so time-varying effects stay locked to song time whatever span they get."""

    sr: int
    bpm: float
    a4: float = 440.0
    seed: int = 0
    offset: int = 0

    def samples(self, seconds: float) -> int:
        return max(0, int(round(seconds * self.sr)))


@dataclass(frozen=True)
class Voice:
    """One note as the instrument sees it: frequency, gate length in seconds, velocity."""

    freq: float
    pitch: float
    dur: float
    vel: float
    params: Mapping[str, Any]


class _Spec:
    spec_type: ClassVar[str] = ""  # the registry name; distinct from any parameter name

    def to_spec(self) -> dict[str, Any]:
        params = {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]
        return {"type": type(self).spec_type, **params}


@dataclass(frozen=True, kw_only=True)
class Instrument(_Spec):
    """A sound source. `program` is the General MIDI program used by the MIDI exporter."""

    program: int | None = None

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        """Render one note (gate `v.dur` plus release) as a mono float array."""
        raise NotImplementedError

    def release_time(self) -> float:
        """Seconds a voice may ring after its gate closes."""
        return 0.05

    def inserts(self) -> tuple[Effect, ...]:
        """Effects wired into the instrument itself (e.g. a guitar amp), applied to the voice sum."""
        return ()

    def choke_groups(self) -> Mapping[int, str]:
        """Pitch -> group; a new note in a group cuts the ringing ones (open vs closed hat)."""
        return {}


@dataclass(frozen=True, kw_only=True)
class Effect(_Spec):
    """A signal processor for ``(channels, samples)`` buffers."""

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        raise NotImplementedError

    def tail(self, ctx: RenderContext) -> float:
        """Seconds of output the effect adds after its input falls silent."""
        return 0.0
