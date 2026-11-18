"""A short source demonstration; experimental piano, no recordings or fitted phases."""
from notes import CoupledPiano, Note, Song, Track


def build_song() -> Song:
    piano = CoupledPiano(contact_oversample=2, ring_time=1.4, unison_weights=(.115, .77, .115))
    phrase = (
        Note('C4', .5, .55)+Note('D4', .5, .6)+Note('E4', 1, .72)
        +Note('G4', 1, .64)+Note('E4', .5, .56)+Note('D4', .5, .48)
        +Note('C4', 2, .58).param(pedal=True)
    )
    return Song(Track('piano', piano)(phrase), bpm=96, loudness=-20.,
                title='Shared hammer: weighted string output')


song = build_song()
