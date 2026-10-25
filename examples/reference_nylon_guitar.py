"""Experimental procedural nylon guitar, fitted to three Iowa classical notes.

No recordings are loaded. Four other pitches test spectral/envelope transfer.
Perceptual realism remains unverified; this does not replace the default guitar.
"""
from notes import NylonGuitar, Song, Track, melody

settings = {'pick_position': 0.12, 'body': 0, 'treble_decay': 4.0}
guitar = Track("nylon", NylonGuitar(**settings))
song = Song(guitar(melody("E4 F4 G4 A4 B4 A4 G4 E4", step=0.6)),
            bpm=100, seed=17, loudness=-20, title="Experimental reference-calibrated nylon")
