"""GuitarSet JAMS annotations -> an exact-time Music performance at BPM 60.

Only note/control data enter the score; no recorded audio or impulse responses.
See https://github.com/marl/GuitarSet for the dataset and known annotation errors.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from notes.lang.music import Music, Note, Rest, place

_KNOWN_ERRORS = {"04_BN3-154-E_comp", "04_Jazz1-200-B_comp", "02_Funk2-119-G_comp"}
_OPEN_PITCHES = (40, 45, 50, 55, 59, 64)
logger = logging.getLogger(__name__)


def _rows(data: object) -> list[dict[str, Any]]:
    """JAMS serializes observations as rows or as equally long columns."""
    if isinstance(data, list) and all(isinstance(row, dict) for row in data):
        return data
    if isinstance(data, dict):
        if not data:
            return []
        if "time" not in data or "value" not in data:
            raise ValueError("JAMS observations require time and value")
        if not all(isinstance(column, list) for column in data.values()):
            raise ValueError("JAMS columns must be arrays")
        lengths = {len(column) for column in data.values()}
        if len(lengths) != 1:
            raise ValueError("JAMS columns have different lengths")
        return [dict(zip(data, values, strict=True)) for values in zip(*data.values(), strict=True)]
    raise ValueError("JAMS observations must be rows or columns")


def guitarset_music(
    source: str | Path | Mapping[str, Any], *, contours: bool = True, velocity: float = .8,
    string: int | None = None,
) -> Music:
    """Read a whole annotated performance; bind it to a Track and play at BPM 60.

    String indices 0..5 (low E to high e) become DSL strings 6..1. Fractional
    tuning and voiced contours become pitch_points around an integer fret pitch.
    Annotation confidence never becomes velocity. Known timing/duplicate errors
    are rejected. A whole performance is imported so crop boundaries cannot
    accidentally retrigger a note already sounding before the crop.
    ``string=1..6`` imports one string while retaining the full timeline.
    """
    if not np.isfinite(velocity) or not 0 <= velocity <= 1:
        raise ValueError("velocity must be finite and between 0 and 1")
    if string is not None and (not isinstance(string, int) or not 1 <= string <= 6):
        raise ValueError("string must be an integer from 1 to 6")
    selected_string = string
    data = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, str | Path) else source
    metadata = data.get("file_metadata", {})
    title = str(metadata.get("title", ""))
    if title.removesuffix(".jams") in _KNOWN_ERRORS:
        raise ValueError(f"known GuitarSet annotation error: {title}")
    duration = float(metadata.get("duration", 0))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("GuitarSet duration must be finite and positive")
    notes: dict[int, list[dict[str, Any]]] = {}
    tracks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for annotation in data.get("annotations", []):
        namespace = annotation.get("namespace")
        if namespace not in ("note_midi", "pitch_contour"):
            continue
        value = annotation.get("annotation_metadata", {}).get("data_source")
        if str(value) not in {str(i) for i in range(6)}:
            raise ValueError("GuitarSet data_source must identify a string index 0..5")
        string = int(value)
        observations = _rows(annotation.get("data", []))
        if namespace == "note_midi":
            if string in notes:
                raise ValueError("duplicate note annotation for a guitar string")
            notes[string] = observations
        elif contours:
            if string in tracks:
                raise ValueError("duplicate pitch contour for a guitar string")
            times, pitches, indices = [], [], []
            missing = 0
            for row in observations:
                point = row["value"]
                if not point.get("voiced", False):
                    continue
                time, freq = float(row["time"]), float(point["frequency"])
                if not np.isfinite(time) or time < 0 or not np.isfinite(freq) or freq < 0:
                    raise ValueError("invalid voiced pitch observation")
                if freq == 0:
                    missing += 1
                    continue  # Some official files mark missing frequencies as voiced zero.
                index = point.get("index", -1)
                if not isinstance(index, int) or index < -1:
                    raise ValueError("pitch observation index must be a nonnegative note index")
                times.append(time)
                pitches.append(69 + 12*np.log2(freq/440))
                indices.append(index)
            t, p, ids = np.asarray(times), np.asarray(pitches), np.asarray(indices, dtype=int)
            if np.any(ids >= 0) and np.any(ids < 0):
                raise ValueError("pitch observations must consistently identify their notes")
            if any(np.any(np.diff(t[ids == index]) <= 0) for index in set(indices)):
                raise ValueError("pitch observations must have increasing times within each note")
            tracks[string] = t, p, ids
            if missing:
                logger.warning("%s: string %d has %d missing voiced frequencies", title, 6-string, missing)
    placed = []
    for string, observations in sorted(notes.items()):
        if selected_string is not None and selected_string != 6-string:
            continue
        seen = set()
        for note_index, row in enumerate(observations):
            start, length, pitch = float(row["time"]), float(row["duration"]), float(row["value"])
            if (not np.isfinite([start, length, pitch]).all() or start < 0 or length <= 0
                    or start+length > duration+.001):
                raise ValueError("note is invalid or outside the performance")
            nominal = round(pitch)
            if not _OPEN_PITCHES[string] <= nominal <= _OPEN_PITCHES[string]+24:
                raise ValueError("note cannot be played on its annotated string")
            if (start, length, pitch) in seen:
                raise ValueError("duplicate note observation")
            seen.add((start, length, pitch))
            params: dict[str, Any] = {"string": 6-string}
            points = [(0., pitch-nominal)]
            if contours and string in tracks:
                times, pitches, indices = tracks[string]
                within = (times >= start) & (times < start+length)
                if np.any(indices >= 0):
                    within &= indices == note_index
                if within.any():
                    t, p = times[within]-start, pitches[within]-nominal
                    points = [(0., float(p[0]))]
                    points.extend((float(time), float(value)) for time, value in zip(t, p, strict=True) if time > 0)
                    points.append((length, float(p[-1])))
            params["pitch_points"] = tuple(points)
            note = Note(nominal, Fraction(str(length)), velocity, tuple(sorted(params.items())))
            placed.append((Fraction(str(start)), note))
    if not placed:
        if selected_string is not None and notes:
            return Rest(Fraction(str(duration)))
        raise ValueError("GuitarSet annotation contains no playable notes")
    return place(placed, Fraction(str(duration)))
