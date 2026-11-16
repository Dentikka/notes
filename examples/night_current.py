"""A connected thirty-second melodic rock miniature, entirely procedural."""
from fractions import Fraction

from notes import (
    BassGuitar,
    Compressor,
    Delay,
    ElectricGuitar,
    Filter,
    Note,
    Rest,
    Reverb,
    RockKit,
    Room,
    Song,
    Track,
    cat,
    place,
    stack,
)


def build_song() -> Song:
    tracks = {
        'kick': Track('kick', RockKit(tune=-5, decay=.7, click=.6, kick_damping=.7, kick_body=.3), gain=2.5),
        'snare': Track('snare', RockKit(decay=.8, wires=.85, rimshot=.35), gain=2.,
                       fx=[Room(size=.25, decay=.32, damp=.65, mix=.07)]),
        'hats': Track('hats', RockKit(decay=.65), gain=3.5, pan=.22),
        'toms': Track('toms', RockKit(decay=.75), gain=-1., pan=-.18),
        'cymbals': Track('cymbals', RockKit(decay=.65), gain=-3., pan=-.25),
        'bass': Track('bass', BassGuitar(treble_decay=.1, drive=.3, sustain=14., polarization=.5,
                                       clank=.1, release=.035), gain=-6.,
                      fx=[Compressor(threshold=-20, ratio=2.5, attack=.018, release=.1),
                          Filter(kind='lowpass', cutoff=3000)]),
        'rhythm-L': Track('rhythm-L', ElectricGuitar(drive=.56, tone=.4, fretting=True,
            pickup_position=.06, pickup_width=.04, pickup_q=.7, treble_decay=2., pick_noise=.08),
            gain=2., pan=-.72,
            fx=[Filter(kind='highpass', cutoff=110), Filter(kind='lowpass', cutoff=5100),
                Reverb(size=.24, damp=.65, mix=.045)]),
        'rhythm-R': Track('rhythm-R', ElectricGuitar(drive=.53, tone=.43, fretting=True,
            pickup_position=.065, pickup_width=.035, pickup_q=.75, pick_position=.17,
            treble_decay=1.8, pick_noise=.07), gain=1.5, pan=.72,
            fx=[Filter(kind='highpass', cutoff=120), Filter(kind='lowpass', cutoff=4900),
                Reverb(size=.27, damp=.65, mix=.045)]),
        'lead': Track('lead', ElectricGuitar(drive=.64, tone=.46, sustain=12., fretting=True,
            pickup_position=.06, pickup_width=.03, pickup_q=.7, treble_decay=3.,
            polarization=.22, pick_noise=.09), gain=4., pan=.04,
            fx=[Filter(kind='highpass', cutoff=200), Filter(kind='lowpass', cutoff=5800),
                Delay(time=.75, feedback=.22, mix=.14, tone=3100),
                Reverb(size=.38, damp=.6, mix=.10)]),
    }
    events: dict[str, list] = {name: [] for name in tracks}

    def add(name: str, at: float, pitch: int, duration: float, velocity: float, **params: object) -> None:
        events[name].append((Fraction(str(at)), Note(pitch, Fraction(str(duration)), velocity, params)))

    def drum(name: str, at: float, pitch: int, velocity: float) -> None:
        add(name, at, pitch, .1, velocity)

    grips = {'Em': ((40, 6), (47, 5), (52, 4)), 'C': ((48, 5), (55, 4), (60, 3)),
             'G': ((43, 6), (50, 5), (55, 4)), 'D': ((50, 4), (57, 3), (62, 2))}
    roots = {'Em': 28, 'C': 36, 'G': 31, 'D': 38}

    def guitar(at: float, chord: str, duration: float, velocity: float, muted: bool = False) -> None:
        voices = grips[chord][:1] if muted else grips[chord]
        for side, name in enumerate(('rhythm-L', 'rhythm-R')):
            for voice, (pitch, string) in enumerate(voices):
                offset = side/64+voice/100
                add(name, at+offset, pitch, duration-voice/100,
                    velocity*(1., .82, .68)[voice]*(1.-side*.04), string=string,
                    palm_mute=muted, pitch_interpolation='sinc')

    # One progression and one rhythmic signature connect the entire piece.
    chords = ('Em', 'C', 'G', 'D')*3+('Em', 'C', 'D', 'Em')
    hits = (0., .75, 1.5, 2.5, 3.25)
    for bar, chord in enumerate(chords):
        if bar == 15:
            break
        quiet = bar in (6, 7)
        intro = bar < 2
        solo = 8 <= bar < 12
        strength = .64 if intro else .48 if quiet else .80 if solo else .86
        pattern = (0., 2.5) if quiet else hits
        for beat in pattern:
            at = 4*bar+beat
            open_hit = beat in (0., 1.5) or (bar == 14 and beat == 3.25)
            duration = 1.3 if quiet else .65 if open_hit else .19
            velocity = strength if open_hit else strength*.76
            guitar(at, chord, duration, velocity, muted=not open_hit)
            add('bass', at, roots[chord], 1.1 if quiet else .47 if open_hit else .24, velocity)
            drum('kick', at, 36, .88 if open_hit and not quiet else .56 if quiet else .68)
        backbeats = (2.,) if quiet else (1., 3.)
        for beat in backbeats:
            drum('snare', 4*bar+beat, 38, .55 if quiet else .75 if intro else .87+(.04 if beat == 3 else 0))
        if not quiet:
            drum('snare', 4*bar+2.75, 38, .18)
        for i in range(4 if quiet else 8):
            beat = float(i) if quiet else i/2
            drum('hats', 4*bar+beat, 46 if i == 7 and bar in (3, 11, 13) else 42,
                 .28 if quiet else (.53 if i % 2 == 0 else .32)*(.85 if intro else 1.))
        if bar in (2, 8, 12, 14):
            drum('cymbals', 4*bar, 49, .73 if bar < 12 else .82)

    # The hook has a different final answer over each chord, rather than stacked competing tunes.
    hook = (
        ((0., 64, .8), (1.5, 67, .7), (2.5, 71, 1.1)),
        ((0., 72, 1.), (1.5, 71, .7), (2.5, 67, 1.1)),
        ((0., 71, .8), (1.5, 69, .7), (2.5, 67, 1.1)),
        ((0., 66, .8), (1.5, 69, .7), (2.5, 74, 1.1)),
    )
    for bar in (0, 1, 2, 3, 4, 5, 12, 13, 14):
        for i, (beat, pitch, duration) in enumerate(hook[3 if bar == 14 else bar % 4]):
            add('lead', 4*bar+beat, pitch, duration,
                (.59 if bar < 2 else .76 if bar < 12 else .84)-i*.025,
                string=1 if pitch >= 64 else 2, legato=i == 1,
                vibrato=.045 if duration > 1 else 0., vibrato_delay=.18,
                pitch_interpolation='sinc')
    for at, pitch, duration, velocity in ((24., 71, 1.4, .47), (26., 67, 1.5, .42),
                                           (28., 69, 1.1, .49), (29.5, 66, .7, .53), (30.5, 62, .7, .57)):
        add('lead', at, pitch, duration, velocity, string=1 if pitch >= 64 else 2, legato=True, vibrato=.04)
    solo = ((0., 64, .65), (.75, 67, .35), (1.25, 69, 1.4), (3., 71, .65),
            (4., 72, .7), (4.75, 71, .4), (5.5, 67, 1.1), (7., 64, .65),
            (8., 67, .4), (8.5, 69, .4), (9., 71, .65), (9.75, 74, 1.2),
            (11.25, 71, .4), (12., 69, .65), (12.75, 66, .6), (13.5, 74, 1.6))
    for i, (offset, pitch, duration) in enumerate(solo):
        bend = 2. if offset in (1.25, 13.5) else 0.
        add('lead', 32+offset, pitch, duration, .88 if bend else .76+(.06 if i % 3 == 0 else 0.),
            string=1, legato=i % 3 == 1, bend=bend, bend_time=.22, bend_start=.06,
            vibrato=.06 if duration > 1 else .025, vibrato_delay=.2, pitch_interpolation='sinc')
    for bar in (7, 11, 14):
        for i, pitch in enumerate((50, 47, 45)):
            drum('toms', 4*bar+3+i*.25, pitch, .48+.12*i)
    # Resolve the D harmony to E minor, then leave space for the shared decay.
    guitar(60., 'Em', 1.9, .94)
    add('bass', 60., 28, 1.8, .88)
    add('lead', 60., 76, 1.9, .86, string=1, vibrato=.07, vibrato_delay=.15)
    drum('kick', 60., 36, .96)
    drum('snare', 60., 38, .76)
    drum('cymbals', 60., 49, .87)

    sections = cat(Rest(8).named('Hook pickup'), Rest(16).named('Main riff'),
                   Rest(8).named('Half-time breath'), Rest(16).named('Solo'),
                   Rest(12).named('Hook return'), Rest(4).named('Final hit'))
    band = stack(*(tracks[name](place(part, length=64)) for name, part in events.items()))
    return Song(stack(band, sections), bpm=128, seed=116, sample_rate=44100,
                loudness=-15.5, ceiling=-1., key='E minor', title='Night Current / Ночной ток')


song = build_song()
