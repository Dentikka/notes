"""Three alternate arrangements; the accepted original suite stays available.

Clean picking quotes Iron Current's actual opening. Only one part leads at a
time. Steel Pulse has an independent solo, and Warm Light uses continuous bow
phrases and written dynamic arcs. Every sound is procedurally synthesized.
"""
from __future__ import annotations

import runpy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from notes import EQ, Filter, Song

_OLD = runpy.run_path(str(Path(__file__).with_name('iron_current_suite.py')))
Band = _OLD['Band']
MAIN = _OLD['MAIN']
CHORUS = _OLD['CHORUS']
SOFT = _OLD['SOFT']
VOICINGS = _OLD['VOICINGS']
SOFT_THEME = _OLD['SOFT_THEME']
ROCK_THEME = _OLD['ROCK_THEME']
string_for = _OLD['string_for']

# The first two rows, accents, gates and acoustic offsets quote the miniature.
PICKING = {
    'Em9': (52, 59, 66, 67, 59, 64, 66, 59),
    'Cmaj7': (48, 55, 59, 64, 55, 62, 64, 59),
    'G6': (43, 55, 62, 67, 59, 62, 67, 62),
    'Dadd9': (50, 57, 62, 66, 57, 64, 66, 62),
    'Am9': (45, 52, 60, 64, 55, 59, 64, 60),
    'B7': (47, 54, 57, 63, 54, 59, 66, 63),
}
ACCENTS = (.72, .48, .58, .50, .66, .45, .61, .48)
CLEAN_HOOK = (
    ((64, .75), (66, .25), (67, .5), (71, 1), (69, .5), (67, .5), (None, .5)),
    ((67, .5), (64, .5), (67, 1), (71, .75), (67, .75), (None, .5)),
    ((71, 1), (69, .5), (67, .5), (74, .75), (71, .75), (None, .5)),
    ((69, .75), (66, .25), (64, .5), (62, 1), (66, 1), (None, .5)),
    ((71, .75), (67, .25), (64, 1), (67, .5), (66, .5), (64, .5), (None, .5)),
    ((64, .5), (67, .5), (71, 1), (67, .5), (64, 1), (None, .5)),
    ((69, .75), (72, .25), (71, .5), (69, 1), (64, 1), (None, .5)),
    ((66, .75), (63, .25), (66, .5), (69, .5), (71, 1), (66, .5), (None, .5)),
)
# Eight distinct bars on CHORUS, with bends landing on chord tones. Rests let
# the rhythm answer instead of every instrument filling every subdivision.
SOLO = (
    ((71, .5), (74, .25), (76, .75), (None, .5), (74, .25), (71, .25), (67, .5), (64, 1)),
    ((67, .5), (71, .5), (74, .5), (76, 1.5), (74, .5), (71, .5)),
    ((74, .5), (79, 1), (78, .25), (76, .25), (74, .5), (71, .5), (None, 1)),
    ((78, 1.5), (74, .5), (69, .5), (66, .5), (69, 1)),
    ((67, .25), (71, .25), (72, .5), (76, .75), (74, .25), (72, .5), (71, .5), (67, 1)),
    ((69, .75), (72, .25), (76, 1.5), (74, .5), (72, .5), (69, .5)),
    ((75, .75), (78, .25), (81, .5), (78, .5), (75, .5), (71, 1), (None, .5)),
    ((76, 1.5), (74, .5), (71, .5), (67, .5), (64, 1)),
)


def melody(b: Band, bar: int, row: tuple, track: str, energy: float, *, bend: bool = False) -> None:
    """A phrase with connected gates, accents and breaths; durations include rests."""
    cursor = Fraction(bar*4)
    previous = None
    for i, (pitch, duration) in enumerate(row):
        d = Fraction(str(duration))
        if pitch is not None:
            params = {}
            if track in ('lead', 'clean-melody'):
                params.update(string=string_for(pitch), pitch_interpolation='sinc')
                if previous is not None and d <= Fraction(1, 2):
                    params['legato'] = True
                if d >= 1:
                    params.update(vibrato=.06 if track == 'lead' else .02, vibrato_delay=.25)
                if bend and pitch == 76 and d >= 1:
                    # D -> E, never the earlier A -> B over a D-major chord.
                    pitch -= 2
                    params.update(bend=2., bend_start=.02, bend_time=.18)
            accent = 1. if i == 0 or d >= 1 else (.82, .91, .86)[i % 3]
            b.add(track, cursor, pitch, d*Fraction(98, 100), energy*accent, **params)
        previous = pitch
        cursor += d
    if cursor != (bar+1)*4:
        raise ValueError('A melody row must occupy exactly one bar')


def bow_line(b: Band, track: str, bar: int, pitches: tuple[int, ...], energy: float) -> None:
    """One excitation/envelope, continuous phase and short finger transitions."""
    seconds = 60/b.bpm
    points = [(0., 0.)]
    for i, pitch in enumerate(pitches[1:], 1):
        at = i*4*seconds
        points.extend(((at-.045, float(pitches[i-1]-pitches[0])), (at, float(pitch-pitches[0]))))
    length = len(pitches)*4*seconds
    b.add(track, bar*4, pitches[0], len(pitches)*4-.04, energy, pitch_points=points,
          expression_points=((0., .52), (length*.4, .88), (length*.65, 1.), (length, .48)))


def picking(b: Band, bar: int, chord: str, scale: float, *, sparse: bool = False) -> None:
    for i, pitch in enumerate(PICKING[chord]):
        if sparse and i not in (0, 3, 4, 6):
            continue
        b.add('clean', bar*4+i*.5, pitch, .72, ACCENTS[i]*scale)
        if not sparse or i in (0, 4):
            b.add('acoustic', bar*4+i*.5+(.025 if i % 2 else .015), pitch, .65,
                  ACCENTS[i]*.75*scale, string=string_for(pitch))


def clean_song() -> Song:
    b = Band(48, 128, 'Свет на воде — новая тема / Light on Water II', 'clean')
    b.tracks['cello'] = _OLD['_TRACKS']['cello']
    b.tracks['clean-melody'] = replace(b.tracks['clean-melody'], gain=2.5, pan=.1)
    form = ((0, 4, 'Opening quotation', 'intro'), (4, 8, 'Picking theme', 'verse'),
            (12, 8, 'Singing refrain', 'chorus'), (20, 8, 'Theme variation', 'verse2'),
            (28, 4, 'Cello bridge', 'bridge'), (32, 12, 'Refrain and release', 'final'),
            (44, 4, 'Opening remembered', 'outro'))
    for start, length, name, role in form:
        b.mark(name, start, length)
        for j in range(length):
            bar = start+j
            chord = MAIN[j % 8]
            if role == 'bridge':
                chord = ('Em9', 'Cmaj7', 'Am9', 'B7')[j]
            elif role == 'outro':
                chord = ('Cmaj7', 'Dadd9', 'Em9', 'Em9')[j]
            foreground = role in ('chorus', 'final')
            scale = .62 if foreground else (.72 if role == 'bridge' else 1.)
            if role == 'verse2':
                scale = .78+.025*j
            if role == 'outro':
                scale = .85-.17*j
            picking(b, bar, chord, scale, sparse=foreground or role == 'bridge')
            if foreground:
                melody(b, bar, CLEAN_HOOK[j % 8], 'clean-melody',
                       (.64 if role == 'chorus' else .68)*(.9+.1*(j % 4)/3))
            if role not in ('intro', 'bridge', 'outro'):
                b.bass(bar, chord, .48 if not foreground else .56, variation=0)
                b.drums(bar, .31 if not foreground else .46, kind='light', variation=j,
                        fill=j == length-1)
            elif role == 'intro':
                for beat in (0, 2):
                    b.hit('hats', bar*4+beat, 44, .32)
            if role == 'intro':
                b.add('cello', bar*4, (52, 48, 43, 50)[j], 3.7 if j == 0 else 3.35,
                      .56 if j == 0 else .52, **({'legato': True} if j else {}))
            elif foreground and j % 4 == 0:
                bow_line(b, 'cello', bar, (52, 52, 50, 50) if j % 8 == 0 else (52, 52, 52, 51), .40)
    # A single exposed cello phrase; the guitars give it room.
    bow_line(b, 'cello', 28, (52, 52, 52, 51), .60)
    b.add('cello', 188, 52, 3.2, .40)
    result = b.finish()
    result.seed = 47
    result.loudness = -18.
    return result


def rock_song() -> Song:
    b = Band(64, 128, 'Стальной пульс — развитие / Steel Pulse II', 'rock')
    for name in ('rhythm-L', 'rhythm-R'):
        old = b.tracks[name]
        b.tracks[name] = replace(old,
            instrument=replace(old.instrument, drive=.54, tone=.35, pick_noise=.055, pickup_q=.55),
            fx=(*old.fx, EQ(mid=-2.5, mid_freq=2800, mid_q=.8, high=-2., high_freq=4300),
                Filter(kind='lowpass', cutoff=5700)), gain=old.gain+.8)
    b.tracks['lead'] = replace(b.tracks['lead'], gain=5.)
    form = ((0, 4, 'Riff opening', 'intro'), (4, 8, 'Verse', 'verse'),
            (12, 4, 'Lift', 'lift'), (16, 8, 'Chorus', 'chorus'), (24, 8, 'Second verse', 'verse2'),
            (32, 8, 'Half-time and silence', 'bridge'), (40, 8, 'Independent guitar solo', 'solo'),
            (48, 12, 'Final chorus', 'final'), (60, 4, 'Ending', 'outro'))
    for start, length, name, role in form:
        b.mark(name, start, length)
        for j in range(length):
            bar = start+j
            if role == 'outro' and j >= 2:
                continue
            chord = (CHORUS if role in ('chorus', 'solo', 'final') else MAIN)[j % 8]
            if role == 'lift':
                chord = ('Cmaj7', 'Am9', 'Dadd9', 'B7')[j]
            elif role == 'bridge':
                chord = ('Am9', 'Cmaj7', 'Em9', 'B7')[j % 4]
            energy = {'intro': .7, 'verse': .68, 'lift': .72+.035*j, 'chorus': .83,
                      'verse2': .72, 'bridge': .60, 'solo': .73, 'final': .86, 'outro': .77}[role]
            if role == 'bridge':
                # Held accents, then a bar of space before the solo pickup.
                if j != 7:
                    b.power(bar*4, chord, 1.45, energy)
                    b.power(bar*4+2.5, chord, .5, energy*.78)
                    b.bass(bar, chord, .62)
                    b.drums(bar, .63, kind='half', variation=j)
                else:
                    b.hit('kick', bar*4, 36, .64)
                    b.power(bar*4, 'B7', .5, .62)
                    melody(b, bar, ((None, 2), (66, .5), (69, .5), (71, .5), (75, .5)), 'lead', .72)
                continue
            b.riff(bar, chord, energy, open_chords=role in ('chorus', 'solo', 'final', 'lift'), variation=j)
            if role == 'verse2' and j < 4:
                # A bass-led response to verse one; no continuous double guitar chug.
                for name_ in ('rhythm-L', 'rhythm-R'):
                    b.events[name_] = [(t, n) for t, n in b.events[name_] if t < bar*4 or t % 4 < 1.8]
            b.bass(bar, chord, .73 if role != 'solo' else .67, busy=role not in ('intro', 'lift'), variation=j)
            b.drums(bar, .87 if role in ('chorus', 'final') else .75, variation=j, fill=j % 4 == 3)
            if role == 'solo':
                melody(b, bar, SOLO[j], 'lead', .87, bend=True)
            elif role in ('chorus', 'final'):
                # Chorus hook is separate from the later solo's register and rhythm.
                melody(b, bar, ROCK_THEME[j % 8], 'lead', .78 if role == 'chorus' else .84)
            elif role in ('verse', 'verse2') and j % 4 == 3:
                row = ((67, 1), (64, .5), (62, .5), (66, 1), (None, 1)) if chord == 'Dadd9' else (
                    (66, .75), (63, .25), (66, 1), (71, 1), (None, 1))
                melody(b, bar, row, 'lead', .62)
    b.power(248, 'Em9', 2.3, .9)
    b.add('bass', 248, 28, 2.3, .83)
    b.add('lead', 248, 76, 2.5, .87, string=1, vibrato=.075)
    b.hit('kick', 248, 36, .96)
    b.hit('snare', 248, 38, .8)
    b.hit('cymbals', 248, 49, .85)
    result = b.finish()
    result.loudness = -17.
    return result


def soft_song() -> Song:
    b = Band(48, 96, 'Тёплый свет — дыхание / Warm Light II', 'soft')
    form = ((0, 4, 'Piano opening', 'intro'), (4, 8, 'First melody', 'verse'),
            (12, 8, 'Refrain', 'chorus'), (20, 8, 'Second melody', 'verse2'),
            (28, 4, 'Dominant bridge', 'bridge'), (32, 12, 'Final refrain', 'final'),
            (44, 4, 'Quiet coda', 'outro'))
    for start, length, name, role in form:
        b.mark(name, start, length)
        for j in range(length):
            bar = start+j
            harmony = SOFT[4:]+SOFT[:4] if role == 'verse2' else SOFT
            chord = harmony[j % 8]
            if role == 'bridge':
                chord = ('Dm9', 'Fmaj7', 'Bhalf', 'E7')[j]
            elif role == 'outro':
                chord = ('Fmaj7', 'Dm9', 'E7', 'Am9')[j]
            level = {'intro': .44, 'verse': .49, 'chorus': .61, 'verse2': .51,
                     'bridge': .43, 'final': .63, 'outro': .44}[role]
            # Four-bar breathing, an eight-bar rise and a final release.
            wave = (.87, .95, 1.03, .92)[j % 4]
            swell = .90+.16*j/max(1, length-1)
            if role == 'outro':
                swell = 1.-.19*j
            energy = level*wave*swell
            if role not in ('bridge', 'outro'):
                melody(b, bar, SOFT_THEME[(j+(4 if role == 'verse2' else 0)) % 8], 'piano', energy)
            elif role == 'bridge':
                rows = (((65, 1), (64, .5), (62, .5), (69, 2)),
                        ((65, 1), (69, .5), (67, .5), (64, 2)),
                        ((65, 1), (62, .5), (59, .5), (57, 2)),
                        ((64, 1), (68, .5), (71, .5), (68, 2)))
                melody(b, bar, rows[j], 'piano', energy)
            else:
                melody(b, bar, (((69, 1), (65, 1), (64, 2)), ((65, 1), (62, 1), (57, 2)),
                               ((64, 1), (68, 1), (71, 2)), ((72, 1), (71, 1), (69, 2)))[j],
                        'piano', energy)
            tones = VOICINGS[chord]
            # Remove the competing high piano ostinato and duplicate guitar runs.
            b.add('piano-low', bar*4, tones[0], 1.75, energy*.64)
            if role != 'intro':
                for beat, index, amount in ((0, 1, .7), (1.5, 2, .48), (3, 1, .56)):
                    b.add('clean', bar*4+beat, tones[index], .8, energy*amount)
                if role in ('chorus', 'final'):
                    b.add('acoustic', bar*4+.02, tones[0], 1.2, energy*.65, string=string_for(tones[0]))
                    b.add('acoustic', bar*4+2.02, tones[2], .9, energy*.45, string=string_for(tones[2]))
            if bar >= 8 and role != 'outro':
                b.bass(bar, chord, .43*wave*swell, variation=0)
                b.drums(bar, (.40 if role in ('chorus', 'final') else .29)*wave*swell,
                        kind='half', variation=j, fill=j == length-1 and role != 'bridge')
            if j % 4 == 0:
                cello = (52, 53, 52, 50) if (j+(4 if role == 'verse2' else 0)) % 8 == 0 else (52, 53, 50, 52)
                if role == 'bridge':
                    cello = (50, 53, 53, 52)
                elif role == 'outro':
                    cello = (53, 50, 52, 45)
                bow_line(b, 'cello', bar, cello, .46 if role in ('chorus', 'final') else .36)
                # Only selected responses receive upper strings; never another busy tune.
                if role in ('chorus', 'final') and j % 8 == 4:
                    bow_line(b, 'violin', bar, (72, 72, 69, 68), .39)
    result = b.finish('A minor')
    result.loudness = -18.
    return result


BUILDERS = {'clean': clean_song, 'rock': rock_song, 'soft': soft_song}


def build_song(version: str = 'clean') -> Song:
    return BUILDERS[version]()


song = build_song()
