"""Iron Current: a 30-second melodic hard-rock miniature, entirely procedural.

Sixteen bars at 128 BPM. Clean introduction, syncopated double-tracked riff,
short lift, half-time piano/bow interlude, melodic solo and a final band hit.
Velocities and articulations are written into the part rather than randomized.
"""
from fractions import Fraction

from notes import (
    EQ,
    AcousticGuitar,
    BassGuitar,
    Cello,
    Chorus,
    Compressor,
    Delay,
    ElectricGuitar,
    Filter,
    Note,
    Piano,
    Rest,
    Reverb,
    RockKit,
    Room,
    Song,
    Track,
    Violin,
    cat,
    stack,
)
from notes.lang.music import place

BPM = 128
LENGTH = 64


def build_song() -> Song:
    tracks = {
        'kick': Track('kick', RockKit(tune=-5, decay=.7, click=.6, kick_damping=.7, kick_body=.3),
                      gain=.5),
        'snare': Track('snare', RockKit(decay=.8, wires=.85, rimshot=.42),
                       fx=[Compressor(threshold=-17, ratio=2., attack=.012, release=.11),
                           Room(size=.25, decay=.32, damp=.65, mix=.07)], gain=-1),
        'hats': Track('hats', RockKit(decay=.7), gain=-4, pan=.22),
        'toms': Track('toms', RockKit(decay=.75), fx=[Room(size=.3, decay=.35, mix=.06)],
                      gain=-2, pan=-.18),
        'cymbals': Track('cymbals', RockKit(decay=.65), gain=-7, pan=-.25),
        'bass': Track('bass', BassGuitar(treble_decay=.1, drive=.3, sustain=14., polarization=.5,
                                        clank=.12, release=.035),
                      fx=[Compressor(threshold=-20, ratio=2.5, attack=.018, release=.1),
                          Filter(kind='lowpass', cutoff=3200)], gain=0),
        'rhythm-L': Track('rhythm-L', ElectricGuitar(drive=.68, tone=.46, fretting=True,
            pickup_position=.06, pickup_width=.04, pickup_q=.7, treble_decay=2., pick_noise=.12),
            fx=[Filter(kind='highpass', cutoff=100), EQ(low=-1.5, low_freq=160, high=-1., high_freq=6000),
                Reverb(size=.24, damp=.65, mix=.045)], gain=-4.3, pan=-.78),
        'rhythm-R': Track('rhythm-R', ElectricGuitar(drive=.64, tone=.5, fretting=True,
            pickup_position=.065, pickup_width=.035, pickup_q=.75, pick_position=.17,
            treble_decay=1.8, pick_noise=.1),
            fx=[Filter(kind='highpass', cutoff=110), Reverb(size=.27, damp=.65, mix=.045)],
            gain=-4.8, pan=.78),
        'clean': Track('clean', ElectricGuitar(drive=.1, tone=.48, pickup='neck', pick_noise=.08),
                       fx=[Filter(kind='highpass', cutoff=140), Chorus(mix=.12, depth=.0015),
                           Reverb(size=.4, damp=.6, mix=.14)], gain=-3.5, pan=-.34),
        'acoustic': Track('acoustic', AcousticGuitar(model='modal', fretting=True, pick_position=.08,
            brightness=1., treble_decay=1.5, prompt_decay=.7, pick_noise=.3),
            fx=[Filter(kind='highpass', cutoff=160), Reverb(size=.3, mix=.10)], gain=-5, pan=.4),
        'lead': Track('lead', ElectricGuitar(drive=.62, tone=.55, pickup='neck', sustain=12.,
            treble_decay=3., polarization=.22, pick_noise=.09, fretting=True),
            fx=[Filter(kind='highpass', cutoff=220), Delay(time=.75, feedback=.23, mix=.16, tone=3300),
                Reverb(size=.38, damp=.55, mix=.12)], gain=.5, pan=.06),
        'piano': Track('piano', Piano(model='hammer_damped', release=.12),
                       fx=[Filter(kind='highpass', cutoff=150), Reverb(size=.35, mix=.10)], gain=-1.5, pan=.18),
        'cello': Track('cello', Cello(pressure=.45, vibrato=.065, bow_noise=.018, release=.25),
                       fx=[Filter(kind='highpass', cutoff=100), Reverb(size=.4, mix=.12)], gain=-3, pan=-.18),
        'violin': Track('violin', Violin(pressure=.4, vibrato=.08, attack=.10, release=.22),
                        fx=[Filter(kind='lowpass', cutoff=5000), Reverb(size=.43, mix=.16)], gain=-6, pan=.25),
    }
    events = {name: [] for name in tracks}

    def add(name, time, pitch, dur, vel, **params):
        at = time if isinstance(time, Fraction) else Fraction(str(time))
        length = dur if isinstance(dur, Fraction) else Fraction(str(dur))
        events[name].append((at, Note(pitch, length, vel, params)))

    def drum(name, time, pitch, vel):
        add(name, time, pitch, .1, vel)

    grips = {
        'Em': ((40, 6), (47, 5), (52, 4)),
        'C': ((48, 5), (55, 4), (60, 3)),
        'G': ((43, 6), (50, 5), (55, 4)),
        'D': ((50, 4), (57, 3), (62, 2)),
    }

    def power(time, chord, dur, vel, muted=False):
        for side, name in enumerate(('rhythm-L', 'rhythm-R')):
            offset = Fraction(side, 64)
            for voice, (pitch, string) in enumerate(grips[chord]):
                strum = Fraction(voice, 100)
                add(name, Fraction(str(time))+offset+strum, pitch, Fraction(str(dur))-strum,
                    vel*(1., .84, .72)[voice]*(1. if side == 0 else .96),
                    string=string, palm_mute=muted, pitch_interpolation='sinc')

    def chug(time, chord, vel):
        pitch, string = grips[chord][0]
        for side, name in enumerate(('rhythm-L', 'rhythm-R')):
            add(name, Fraction(str(time))+Fraction(side, 64), pitch, .16, vel*(1. if side == 0 else .95),
                string=string, palm_mute=True, pitch_interpolation='sinc')

    # The opening colour is exposed; the same picking returns under the quiet break.
    arps = {
        0: ((52, 4), (59, 2), (66, 1), (67, 1), (59, 2), (64, 1), (66, 1), (59, 2)),
        1: ((48, 5), (55, 3), (59, 2), (64, 1), (55, 3), (62, 2), (64, 1), (59, 2)),
        8: ((45, 5), (52, 4), (60, 2), (64, 1), (55, 3), (59, 2), (64, 1), (60, 2)),
        9: ((47, 5), (54, 4), (57, 3), (64, 1), (63, 2), (57, 3), (66, 1), (63, 2)),
    }
    for bar, tones in arps.items():
        for i, (pitch, string) in enumerate(tones):
            t = 4*bar+i*.5
            accent = (.72, .48, .58, .50, .66, .45, .61, .48)[i]
            add('clean', t, pitch, .72, accent*(.85 if bar >= 8 else 1.))
            add('acoustic', t+(.025 if i % 2 else .015), pitch, .65, accent*.75, string=string)
    add('cello', 0, 52, 3.7, .56)
    add('cello', 4, 48, 3.35, .52, legato=True)
    for t, p, v in ((6.5, 51, .38), (7., 47, .52), (7.5, 45, .64), (7.75, 38, .68)):
        drum('toms' if p != 38 else 'snare', t, p, v)
    for t in (0., 2., 4., 6.):
        drum('hats', t, 44, .32)

    # Tight low-string riff: kick, bass and muted guitar share intentional accents.
    riff_chords = ('Em', 'C', 'G', 'D')
    bass_roots = {'Em': 28, 'C': 36, 'G': 31, 'D': 38}
    patterns = (
        (0., .5, .75, 1.5, 2., 2.75, 3.25, 3.75),
        (0., .75, 1.5, 2., 2.5, 3.25, 3.5),
        (0., .5, 1.25, 1.5, 2.5, 3., 3.5),
        (0., .75, 1.5, 2.25, 2.5, 3.25, 3.75),
    )
    for local, bar in enumerate(range(2, 6)):
        chord = riff_chords[local]
        root = bass_roots[chord]
        for i, beat in enumerate(patterns[local]):
            t = bar*4+beat
            v = .82 if beat in (0., 1.5, 3.25) else .56+.035*(i % 3)
            if beat in (1.5, 3.25):
                power(t, chord, .43, v)
            else:
                chug(t, chord, v)
            add('bass', t, root, .32 if i % 3 else .44, .82 if beat in (0., 1.5) else .66)
        for beat, v in ((0., .92), (.75, .67), (1.5, .82), (2.5, .76), (3.25, .84)):
            drum('kick', bar*4+beat, 36, v)
        for beat in (1., 3.):
            drum('snare', bar*4+beat, 38, .86 if beat == 1 else .91)
        for beat, v in ((1.75, .24), (2.75, .18)):
            drum('snare', bar*4+beat, 38, v)
        for i in range(8):
            drum('hats', bar*4+i*.5, 46 if i == 7 and bar % 2 else 42,
                 (.64 if i % 2 == 0 else .37)+(.04 if i == 0 else 0.))
    for bar in (2, 4):
        drum('cymbals', bar*4, 49, .78)

    # A rising two-bar answer, then a coordinated half-beat stop before the break.
    for bar, chord in ((6, 'C'), (7, 'D')):
        for beat, dur, v in ((0., .8, .83), (1.5, .45, .71), (2.5, .45, .78), (3.25, .18, .91)):
            power(bar*4+beat, chord, dur, v)
            add('bass', bar*4+beat, bass_roots[chord], min(dur, .6), v)
            drum('kick', bar*4+beat, 36, v)
        for beat in (1., 3.):
            drum('snare', bar*4+beat, 38, .88)
        for i in range(8 if bar == 6 else 7):
            drum('hats', bar*4+i*.5, 46 if i in (3, 7) else 42, .53 if i % 2 == 0 else .38)
    for t, pitch, dur, vel, params in (
        (24., 67, .65, .63, {}), (25., 69, .65, .68, {'legato': True}),
        (26., 71, 1.5, .74, {'vibrato': .08}), (28., 69, .6, .68, {}),
        (29., 66, .6, .65, {'legato': True}), (30., 64, .65, .73, {}),
        (31., 66, .3, .62, {}),
    ):
        add('lead', t, pitch, dur, vel, string=1, **params)

    # Half-time interlude: Am9 -> B7, a quieter breath with a clear dominant lift.
    piano = ((32., 57, .48, .62), (32.5, 60, .48, .49), (33., 64, .7, .56),
             (34., 71, .65, .64), (35., 67, .7, .52), (35.5, 64, .35, .44),
             (36., 59, .5, .61), (36.5, 66, .45, .5), (37., 69, .6, .56),
             (38., 64, .45, .62), (38.5, 63, .4, .67), (39., 66, .45, .59))
    for t, pitch, dur, vel in piano:
        add('piano', t, pitch, dur, vel)
    for bar, root in ((8, 33), (9, 35)):
        add('bass', 4*bar, root, 2.7, .61)
        add('bass', 4*bar+3., root+12, .5, .49, legato=True)
        drum('kick', 4*bar, 36, .76)
        drum('kick', 4*bar+1.5, 36, .48)
        drum('snare', 4*bar+2., 38, .70)
        for beat in (0., 1., 2., 3.):
            drum('hats', 4*bar+beat, 42, .34 if beat % 2 else .45)
    add('cello', 32, 45, 3.7, .58)
    add('cello', 36, 47, 3.3, .61, legato=True)
    add('violin', 34, 71, 1.6, .48)
    add('violin', 37, 75, 2.1, .52, legato=True)
    for i, pitch in enumerate((50, 47, 45)):
        drum('toms', Fraction(39)+Fraction(i, 3), pitch, .56+.12*i)
    drum('snare', Fraction(119, 3), 38, .78)

    # Full-band hook; the tune has room to sing between quick legato responses.
    for local, bar in enumerate(range(10, 14)):
        chord = riff_chords[local]
        root = bass_roots[chord]
        for beat, dur, vel in ((0., 1.2, .87), (1.5, .40, .65), (2., .7, .78), (3.25, .5, .88)):
            power(bar*4+beat, chord, dur, vel)
        for i, beat in enumerate((0., .75, 1.5, 2., 2.75, 3.25, 3.75)):
            pitch = root+12 if i == 4 else root
            add('bass', bar*4+beat, pitch, .32 if i else .57, .81 if i in (0, 3, 5) else .66,
                **({'legato': True} if i == 4 else {}))
        for beat, vel in ((0., .94), (.75, .68), (1.5, .85), (2., .79), (3.25, .83)):
            drum('kick', bar*4+beat, 36, vel)
        for beat in (1., 3.):
            drum('snare', bar*4+beat, 38, .93 if beat == 3 else .88)
        drum('snare', bar*4+2.75, 38, .23)
        for i in range(8):
            drum('cymbals', bar*4+i*.5, 51, .52 if i % 2 == 0 else .31)
        if local in (0, 2):
            drum('cymbals', bar*4, 49, .81)
        add('violin', bar*4+.08, (71, 76, 74, 69)[local], 3.5, .54, legato=local > 0)
        add('cello', bar*4, (52, 48, 43, 50)[local], 3.5, .5, legato=True)
    solo = (
        (40., 64, .65, .83, {}), (40.75, 67, .2, .65, {'legato': True}),
        (41., 69, 1.45, .88, {'bend': 2., 'bend_start': .06, 'bend_time': .20, 'vibrato': .055}),
        (42.75, 71, .35, .78, {}), (43.25, 67, .55, .74, {'legato': True}),
        (44., 67, .65, .82, {}), (44.75, 64, .45, .69, {'legato': True}),
        (45.5, 67, .45, .73, {}), (46., 71, .65, .83, {}), (47., 74, .72, .87, {'vibrato': .07}),
        (48., 71, .45, .8, {}), (48.5, 74, .65, .88, {}),
        (49.25, 71, .20, .68, {'legato': True}), (49.5, 69, .35, .74, {'legato': True}),
        (50., 67, .45, .79, {}), (50.75, 64, .4, .71, {'legato': True}),
        (51.5, 62, .3, .67, {}), (52., 66, .65, .80, {}),
        (52.75, 69, .45, .76, {'legato': True}),
        (53.5, 74, 1.15, .91, {'bend': 2., 'bend_time': .22, 'vibrato': .06}),
        (55., 71, .45, .78, {'legato': True}),
    )
    for t, pitch, dur, vel, params in solo:
        add('lead', t, pitch, dur, vel, string=1 if pitch >= 64 else 2,
            pitch_interpolation='sinc', **params)
    for i, pitch in enumerate((50, 47, 45)):
        drum('toms', Fraction(55)+Fraction(i, 3), pitch, .58+.12*i)

    # A final riff answer, then a single wide E-minor hit with time for the tail.
    for beat in (0., .5, .75, 1.5, 2., 2.75):
        chug(56+beat, 'Em', .80 if beat in (0., 1.5, 2.75) else .57)
        add('bass', 56+beat, 28, .32, .79 if beat in (0., 1.5) else .64)
        drum('kick', 56+beat, 36, .87 if beat in (0., 1.5) else .68)
    power(59.25, 'D', .35, .85)
    add('bass', 59.25, 35, .35, .78)
    for beat in (57., 59.):
        drum('snare', beat, 38, .90)
    for i in range(8):
        drum('hats', 56+i*.5, 42, .62 if i % 2 == 0 else .37)
    power(60., 'Em', 1.25, .94)
    add('bass', 60., 28, 1.25, .88)
    add('lead', 60., 76, 1.4, .85, string=1, vibrato=.08, vibrato_delay=.12)
    add('piano', 60.04, 64, 1.2, .61)
    add('piano', 60.08, 71, 1.15, .5)
    add('cello', 60., 52, 1.4, .58, legato=True)
    add('violin', 60., 67, 1.5, .56, legato=True)
    drum('kick', 60., 36, .96)
    drum('snare', 60., 38, .77)
    drum('cymbals', 60., 49, .88)

    # A fretting finger damps the preceding bass note before the next attack.
    bass_notes = sorted(events['bass'], key=lambda item: item[0])
    for i, (at, note) in enumerate(bass_notes[:-1]):
        available = bass_notes[i+1][0]-at-Fraction(3, 40)
        if note.dur > available:
            bass_notes[i] = (at, Note(note.pitch, max(Fraction(1, 20), available), note.vel, note.params))
    events['bass'] = bass_notes
    sections = cat(Rest(8).named('Clean opening'), Rest(16).named('Syncopated riff'),
                   Rest(8).named('Lift and stop'), Rest(8).named('Half-time interlude'),
                   Rest(16).named('Melodic solo'), Rest(4).named('Riff answer'), Rest(4).named('Final hit'))
    band = stack(*(tracks[name](place(notes, length=LENGTH)) for name, notes in events.items()))
    return Song(stack(band, sections), bpm=BPM, seed=47, sample_rate=44100, loudness=-15.5, ceiling=-1.,
                key='E minor', title='Iron Current / Искра')


song = build_song()
