"""Procedural bridge DI candidate fitted on three FSBS notes.

Eight other note/dynamic cases passed the profile check with two seeds at
22.05 kHz. D3 regressed under the auxiliary MERT diagnostic. This example does
not certify realism and does not load any recordings.
"""
from notes import ElectricGuitar, Note, Rest, Song, Track, cat


def build_song(*, baseline: bool = False) -> Song:
    settings = {} if baseline else dict(pickup_position=.06, pickup_width=.04, pickup_q=.7)
    guitar = Track("guitar", ElectricGuitar(amplified=False, fretting=True, **settings))
    phrase = cat(*(cat(Note(pitch, dur=1.4, vel=.7).param(string=string), Rest(.4))
                   for pitch, string in ((40, 6), (52, 4), (59, 2), (64, 1), (71, 1))))
    return Song(guitar(phrase), bpm=60, seed=17, sample_rate=44100, loudness=-22,
                title="Bridge guitar: " + ("baseline" if baseline else "spatial pickup candidate"))


song = build_song()
