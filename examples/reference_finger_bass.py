"""Procedural finger bass, calibrated against FreePats Yamaha RBX recordings.

Developed on four notes; all eight unseen E1-D#2 notes passed the diagnostic
no-regression check with two synthesis seeds. Other registers and playing
techniques remain unverified. No recordings are loaded by this example.
"""
from notes import BassGuitar, Song, Track, melody

settings = {'treble_decay': 0.1, 'drive': 0.3, 'sustain': 14.0, 'polarization': 0.5}
bass = Track("finger-bass", BassGuitar(**settings))
song = Song(bass(melody("E1 G1 A1 D2 G1 A1 E1@2", step=0.5)),
            bpm=100, seed=17, loudness=-20, title="Reference-calibrated finger bass")
