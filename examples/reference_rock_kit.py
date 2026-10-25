"""Procedural kick calibrated against MuldjordKit; baseline snare/hat retained.

The room response is computed from geometry and generated noise. This composition
loads no recordings. Kick checks cover distinct soft/hard hits, not every kit.
"""
from notes import RockKit, Room, Song, Track, stack, steps

kick_settings = {'tune': -5, 'decay': 0.7, 'click': 0.6, 'kick_damping': 0.7, 'kick_body': 0.3}
kick = Track("kick", RockKit(**kick_settings),
             fx=[Room(size=0.35, decay=0.4, damp=0.65, mix=0.3)])
snare = Track("snare", RockKit())
hat = Track("hat", RockKit(), gain=-2)
beat = stack(kick(steps(kick="X...X.X.", step=0.5)),
             snare(steps(snare="..X...X.", step=0.5)),
             hat(steps(hat="XxXxXxXx", step=0.5)))
song = Song(beat * 2, bpm=96, seed=17, loudness=-20, title="Procedural reference kick groove")
