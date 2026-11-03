"""Procedural modal acoustic candidate, calibrated on four Seagull notes.

Two GuitarSet performances improved on the physical objective in all six
regions and on MERT in five. One region and several isolated notes regressed;
this is a separate optional model, not a universal replacement.
"""
from notes import AcousticGuitar, Note, Rest, Song, Track, cat


def build_song(*, baseline: bool = False) -> Song:
    settings = {} if baseline else dict(model="modal", pick_position=.08, brightness=1.,
                                       treble_decay=1.5, prompt_decay=.7, pick_noise=.3)
    guitar = AcousticGuitar(fretting=True, **settings)
    phrase = cat(*(cat(Note(pitch, dur=1., vel=.7).param(string=string), Rest(.3))
                   for pitch, string in ((40, 6), (50, 4), (55, 3), (59, 2), (64, 1), (76, 1))))
    return Song(Track("guitar", guitar)(phrase), bpm=60, seed=17,
                sample_rate=44100, loudness=-22,
                title="Acoustic guitar: "+("baseline" if baseline else "modal candidate"))


song = build_song()
