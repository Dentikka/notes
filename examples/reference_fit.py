"""Known-part bass matching. Supply an isolated recording of this exact phrase:

    notes fit reference.wav examples/reference_fit.py -o ../renders/<run>/

Keep pitches, durations, tuning, seed and recording offset correct before searching.
Replace the phrase with the transcribed part of the reference for real recordings.
"""

from notes import BassGuitar, Song, Track, melody

initial = {"tone": 0.1, "brightness": 0.2}
grid = {"tone": [0.1, 0.5, 0.9], "brightness": [0.2, 0.6, 1.0]}


def build_song(settings):
    bass = Track("bass", BassGuitar(**settings))
    return Song(bass(melody("A1 D2 E2 G1", step=1)), bpm=120, loudness=None, seed=17)


song = build_song(initial)
