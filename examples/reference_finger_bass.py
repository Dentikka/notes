"""Procedural finger-bass calibration against FreePats Yamaha RBX notes.

No recordings are loaded. Fitted on E1/A1 and checked on G1/D2; a picked E2
is a separate articulation. D2 regresses in the combined feature score, so this
is a diagnostic candidate, not a replacement for the default bass preset.
Numbers improve reference features, not a certificate of full realism.
"""
from notes import BassGuitar, Song, Track, melody

settings = {'treble_decay': 0.3, 'tone': 0.1, 'sustain': 14.0, 'drive': 0.3, 'polarization': 0.5, 'brightness': 0.9}
bass = Track("finger-bass", BassGuitar(**settings))
song = Song(bass(melody("E1 G1 A1 D2 G1 A1 E1@2", step=0.5)),
            bpm=100, seed=17, loudness=-20, title="Reference-calibrated finger bass")
