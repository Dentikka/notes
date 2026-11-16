"""Four-bar contact-piano impact demonstration, using only procedural sources.

The scalar profile is an experimental close-microphone MAPS calibration, not a
replacement for the accepted preset. build_song(with_impact=False) keeps the same
contact/readout profile and removes only the nonharmonic component.
"""
from notes import Music, Note, Piano, Song, Track, parse_pitch, place


def _note(name: str, duration: float, velocity: float) -> Music:
    pitch = parse_pitch(name)
    felt = float(min(16., 2**max(0., (pitch-45)/12)))
    return Note(pitch, duration, vel=velocity).param(felt_stiffness_scale=felt)


def build_song(*, with_impact: bool = True) -> Song:
    piano = Piano(model='hammer_damped', contact_oversample=8, radiation_compensation=1.,
        impact_gain=10**(-56.240024717070135/20) if with_impact else 0.,
        impact_velocity_power=.24945152753129549, impact_decay=.436297100064779,
        impact_color=-.1822236456438433, impact_delay=.014172333516327367)
    events = []
    bars = ['E4 G4 A4 C5 B4 A4 G4 E4', 'G4 C5 E5 D5 C5 B4 A4 G4',
            'A4 C5 F5 E5 D5 G5 E5 C5', 'G4 B4 D5 G5 C6 G5 E5 C5']
    for bar, melody in enumerate(bars):
        for index, name in enumerate(melody.split()):
            velocity = [.48, .55, .62, .58][bar]+(.10 if index in (0, 4) else .03 if index in (2, 6) else 0)
            if bar == 3 and index == 4:
                velocity = .82
            duration = .5 if bar == 3 and index == 7 else .62
            events.append((4*bar+.5*index, _note(name, duration, velocity)))
    for bar, low, fifth in [(0, 'C3', 'G3'), (1, 'A2', 'E3'), (2, 'F3', 'C4'), (3, 'G2', 'D3')]:
        for offset, name, duration, velocity in [(0, low, 2.6, .54), (2, fifth, 1.6, .43)]:
            events.append((4*bar+offset, _note(name, duration, velocity)))
    return Song(Track('piano', piano)(place(events, length=16)), bpm=96, sample_rate=44100,
                seed=17, loudness=-22., title='Contact piano: '+('with impact' if with_impact else 'strings only'))


song = build_song()
