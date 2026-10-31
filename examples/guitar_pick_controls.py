"""Procedural electric-guitar geometry and pick articulation probe.

First E4 on open high e and on B-string fret 5; then A3 with two pick positions.
This demonstrates controls, not a claim of perceptual reference matching.
"""
from notes import ElectricGuitar, Note, Rest, Song, Track, cat

guitar = Track("guitar", ElectricGuitar(drive=0.1, pickup="neck", tone=0.35,
                                      fretting=True, pitch_attack=0, pick_noise=0.05))
notes = cat(Note("E4", dur=1, vel=0.7).param(string=1), Rest(0.2),
            Note("E4", dur=1, vel=0.7).param(string=2), Rest(0.5),
            Note("A3", dur=1, vel=0.7).param(string=3, pick_position=0.08, pick_hardness=0.8), Rest(0.2),
            Note("A3", dur=1, vel=0.7).param(string=3, pick_position=0.3, pick_hardness=0.3))
song = Song(guitar(notes), bpm=60, seed=17, loudness=-22, title="String and pick geometry")
