"""Separate piano key release from held-note timbre in an upper-register phrase.

Both variants use the accepted hammer_damped model without pedal or effects.
Shortening release changes overlap after note-off, not the hammer attack.
This is an articulation example, not a new piano timbre or a realism claim.
"""
from notes import Note, Piano, Rest, Song, Track, cat


def build_song(*, short_release: bool = False) -> Song:
    piano = Piano(model="hammer_damped", release=.045 if short_release else .18)
    held = cat(Note(76, dur=.9, vel=.7), Rest(.6))
    pitches = (76, 79, 83, 79, 74, 77, 81, 77, 72, 76, 79, 76, 71, 74, 78, 83)
    phrase = cat(*(cat(Note(pitch, dur=.16, vel=.7), Rest(.09)) for pitch in pitches))
    return Song(Track("piano", piano)(cat(held, phrase)), bpm=60, seed=17,
                sample_rate=44100, loudness=-22, title="Piano key release: " +
                ("45 ms" if short_release else "180 ms"))


song = build_song()
