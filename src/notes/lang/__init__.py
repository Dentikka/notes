"""The language: a Python-embedded DSL whose values are music.

`music` defines the algebra, `notation` the mini-notation, `theory` pitches, scales and
chords, `perform` the interpretation into IR events, and `song` tracks and whole pieces.
"""

from notes.lang.music import (
    E,
    H,
    Music,
    Note,
    Q,
    Rest,
    S,
    T,
    W,
    bars,
    beats,
    cat,
    dotted,
    place,
    stack,
    triplet,
)
from notes.lang.notation import chord, chords, euclid, hit, melody, steps
from notes.lang.song import Song, Track
from notes.lang.theory import Key, chord_pitches, midi_to_hz, parse_pitch, pitch_name, voice_lead

__all__ = [
    "E",
    "H",
    "Key",
    "Music",
    "Note",
    "Q",
    "Rest",
    "S",
    "Song",
    "T",
    "Track",
    "W",
    "bars",
    "beats",
    "cat",
    "chord",
    "chord_pitches",
    "chords",
    "dotted",
    "euclid",
    "hit",
    "melody",
    "midi_to_hz",
    "parse_pitch",
    "pitch_name",
    "place",
    "stack",
    "steps",
    "triplet",
    "voice_lead",
]
