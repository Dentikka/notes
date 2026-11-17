"""Experimental quiet-response controls, with no recordings or reference phases.

The fitted lower-velocity mapping improves level transfer but does not establish
realistic timbre. This example leaves the accepted palette and Piano defaults alone.
"""
import numpy as np

from notes import Note, Piano, Song, Track, parse_pitch, place
from notes.engine import RenderContext, Voice

BPM = 96
SR = 44100
KNEE = 44/127
FLOOR = .3757533224445332
POWER = 3.9824125044077547


def soft_speed(velocity: float) -> float | None:
    """Original source above the calibration knee, monotone quiet taper below it."""
    if velocity >= KNEE:
        return None
    r = velocity/KNEE
    smooth = r*r*(3-2*r)
    return FLOOR+(.6+3.4*velocity-FLOOR)*smooth**POWER


def build_song(mode: str = 'balanced') -> Song:
    """baseline, changed contact plus scaled impact, or energy-matched output gain."""
    if mode not in ('baseline', 'balanced', 'gain_control'):
        raise ValueError('mode must be baseline, balanced or gain_control')
    piano = Piano(model='hammer_damped', contact_oversample=8, radiation_compensation=1.,
        impact_gain=10**(-56.240024717070135/20), impact_velocity_power=.24945152753129549,
        impact_decay=.436297100064779, impact_color=-.1822236456438433, impact_delay=.014172333516327367)
    preview = Piano(model='hammer_damped', contact_oversample=8, radiation_compensation=1.)
    ctx = RenderContext(SR, BPM)
    events = []
    phrase = [('D4', 8), ('F4', 16), ('A3', 24), ('C4', 40),
              ('E4', 55), ('F4', 72), ('G4', 96), ('D4', 116)]
    for index, (name, midi_velocity) in enumerate(phrase):
        pitch, velocity = parse_pitch(name), midi_velocity/127
        params = {'felt_stiffness_scale': float(min(16., 2**max(0., (pitch-45)/12)))}
        speed = soft_speed(velocity)
        if mode != 'baseline' and speed is not None:
            voice = Voice(440*2**((pitch-69)/12), float(pitch), .8*60/BPM, velocity, params)
            old = preview.voice(voice, ctx, np.random.default_rng(0))
            adjusted = preview.voice(Voice(voice.freq, voice.pitch, voice.dur, voice.vel,
                params|{'hammer_speed': speed}), ctx, np.random.default_rng(0))
            gain = float(np.sqrt(np.mean(adjusted*adjusted)/max(float(np.mean(old*old)), 1e-30)))
            if mode == 'balanced':
                params.update(hammer_speed=speed, impact_gain=piano.impact_gain*gain)
            else:
                params['source_gain_db'] = float(20*np.log10(max(gain, 1e-6)))
        events.append((index, Note(pitch, .8, vel=velocity).param(**params)))
    return Song(Track('piano', piano)(place(events, length=8)), bpm=BPM, sample_rate=SR,
                seed=117, loudness=-22., title='Quiet piano response: '+mode)


song = build_song()
