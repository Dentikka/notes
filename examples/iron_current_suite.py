"""Four developed instrumental songs from the palette of Iron Current.

Each standalone song has its own theme, verse/refrain development and ending.
The suite keeps clean opening -> heavy riff -> piano interlude -> rock climax,
with shared melodic cells, pivot harmony and overlapping instrumental handoffs.
All sources are synthesized; no recordings are loaded.
"""
import runpy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from notes import Note, Rest, Song, Track, stack
from notes.engine import build_effect, build_instrument
from notes.lang.music import place

_BASE = runpy.run_path(str(Path(__file__).with_name('iron_current.py')))['build_song']().compile()
_TRACKS = {t.name: Track(t.name, build_instrument(t.instrument),
                       fx=tuple(build_effect(f) for f in t.fx), gain=t.gain_db, pan=t.pan) for t in _BASE.tracks}

VOICINGS = {
    'Em9': (52, 55, 59, 64, 66, 67), 'Cmaj7': (48, 52, 55, 59, 62, 64),
    'G6': (55, 59, 62, 64, 67, 69), 'Dadd9': (50, 54, 57, 62, 64, 66),
    'Am9': (45, 52, 55, 59, 60, 64), 'B7': (47, 54, 57, 59, 63, 66),
    'E7': (52, 56, 59, 62, 64, 68), 'Fmaj7': (53, 57, 60, 64, 67, 69),
    'Dm9': (50, 53, 57, 60, 64, 65), 'Bhalf': (47, 50, 53, 57, 59, 62),
}
ROOTS = {'Em9': 28, 'Cmaj7': 36, 'G6': 31, 'Dadd9': 38, 'Am9': 33, 'B7': 35,
         'E7': 28, 'Fmaj7': 29, 'Dm9': 38, 'Bhalf': 35}
GRIPS = {
    'Em9': ((40, 6), (47, 5), (52, 4)), 'Cmaj7': ((48, 5), (55, 4), (60, 3)),
    'G6': ((43, 6), (50, 5), (55, 4)), 'Dadd9': ((50, 4), (57, 3), (62, 2)),
    'Am9': ((45, 5), (52, 4), (57, 3)), 'B7': ((47, 5), (54, 4), (59, 3)),
    'E7': ((40, 6), (47, 5), (52, 4)),
}
MAIN = ('Em9', 'Cmaj7', 'G6', 'Dadd9', 'Em9', 'Cmaj7', 'Am9', 'B7')
CHORUS = ('Em9', 'Cmaj7', 'G6', 'Dadd9', 'Cmaj7', 'Am9', 'B7', 'Em9')
SOFT = ('Am9', 'Fmaj7', 'Cmaj7', 'G6', 'Am9', 'Fmaj7', 'Dm9', 'E7')

# Eight-bar themes; the short F#-G-F#-E cell also connects the suite's sections.
CLEAN_THEME = (
    ((64, .75), (66, .25), (67, .5), (71, .75), (69, .25), (67, .5), (66, .5), (64, .5)),
    ((67, .5), (64, .5), (62, .5), (64, .5), (67, .75), (71, .75), (67, .5)),
    ((71, 1.), (69, .5), (67, .5), (74, .75), (71, .25), (69, .5), (67, .5)),
    ((69, .75), (66, .25), (64, .5), (62, .5), (66, .75), (69, .75), (66, .5)),
    ((64, .5), (67, .5), (71, .75), (76, .75), (74, .5), (71, .5), (67, .5)),
    ((67, .75), (64, .75), (62, .5), (64, .5), (67, .75), (66, .25), (64, .5)),
    ((69, .75), (72, .25), (71, .5), (69, .5), (64, .5), (67, .5), (69, .5), (71, .5)),
    ((66, .75), (63, .25), (66, .5), (69, .5), (71, 1.), (75, .5), (66, .5)),
)
ROCK_THEME = (
    ((64, .75), (67, .25), (69, 1.5), (71, .5), (67, .5), (64, .5)),
    ((67, .75), (64, .75), (67, .5), (71, .75), (74, .75), (71, .5)),
    ((71, .5), (74, .75), (71, .25), (69, .5), (67, .75), (64, .75), (62, .5)),
    ((66, .75), (69, .75), (74, 1.5), (71, .5), (69, .5)),
    ((76, .75), (74, .25), (71, .5), (67, .5), (64, .75), (67, .75), (71, .5)),
    ((72, .75), (71, .25), (69, .5), (64, .75), (67, .75), (69, 1.)),
    ((66, .5), (69, .5), (71, .75), (75, .75), (71, .5), (69, .5), (66, .5)),
    ((71, .75), (67, .25), (66, .5), (67, .5), (64, 2.)),
)
SOFT_THEME = (
    ((64, .75), (69, .75), (72, .5), (71, .5), (69, .75), (64, .75)),
    ((69, 1.), (67, .5), (64, .5), (65, .75), (64, .75), (60, .5)),
    ((67, .75), (71, .75), (72, .5), (71, .5), (67, .5), (64, 1.)),
    ((62, .75), (67, .75), (71, .5), (69, .5), (67, 1.5)),
    ((72, .75), (71, .25), (69, .5), (76, 1.), (72, .5), (71, .5), (69, .5)),
    ((69, .75), (67, .75), (65, .5), (64, .5), (60, .5), (64, 1.)),
    ((65, .75), (69, .75), (72, .5), (69, .5), (65, 1.5)),
    ((64, .75), (68, .75), (71, .5), (74, .5), (71, .5), (68, .5), (64, .5)),
)


def string_for(pitch: int) -> int:
    return 6-max(i for i, root in enumerate((40, 45, 50, 55, 59, 64)) if pitch >= root)


class Band:
    def __init__(self, bars: int, bpm: int, title: str, style: str):
        self.bars, self.bpm, self.title, self.style = bars, bpm, title, style
        self.tracks = dict(_TRACKS)
        self.tracks['clean-melody'] = replace(_TRACKS['clean'], name='clean-melody', gain=5., pan=.06)
        if style == 'clean':
            self.tracks['bass'] = replace(_TRACKS['bass'], gain=-10.)
            self.tracks['cello'] = replace(_TRACKS['cello'], gain=-5.)
        if style == 'soft':
            self.tracks['clean'] = replace(_TRACKS['clean'], gain=-.5)
            self.tracks['acoustic'] = replace(_TRACKS['acoustic'], gain=4.)
            self.tracks['bass'] = replace(_TRACKS['bass'], gain=-10.)
            self.tracks['piano'] = replace(_TRACKS['piano'], gain=21.)
        self.events = {name: [] for name in self.tracks}
        self.sections = []

    def add(self, name, beat, pitch, dur, vel, **params):
        at, duration = Fraction(str(beat)), Fraction(str(dur))
        if at < 0 or at >= self.bars*4:
            return
        duration = min(duration, Fraction(self.bars*4)-at)
        if duration > 0:
            self.events[name].append((at, Note(pitch, duration, min(.98, max(.05, vel)), params)))

    def hit(self, name, beat, pitch, vel):
        self.add(name, beat, pitch, .1, vel)

    def mark(self, name, start, bars):
        self.sections.append((start*4, Rest(bars*4).named(name)))

    def texture(self, bar, chord, energy=.65, variation=0, acoustic=True):
        tones = VOICINGS[chord]
        orders = ((0, 2, 4, 5, 3, 2, 4, 1), (0, 3, 2, 5, 4, 2, 1, 3),
                  (0, 2, 3, 4, 5, 3, 2, 1), (0, 4, 2, 5, 3, 1, 4, 2))
        for i, index in enumerate(orders[variation % 4]):
            t = bar*4+i*.5
            pitch = tones[index]
            accent = (1., .64, .8, .69, .9, .62, .84, .68)[i]
            self.add('clean', t, pitch, .7 if variation % 2 == 0 else .6, energy*accent)
            if acoustic:
                self.add('acoustic', t+.018, pitch, .62, energy*accent*.72, string=string_for(pitch))

    def bass(self, bar, chord, energy=.65, busy=False, variation=0):
        root = ROOTS[chord]
        if busy:
            offsets = ((0., .5, .75, 1.5, 2., 2.75, 3.25, 3.75),
                       (0., .75, 1.5, 2., 2.5, 3.25, 3.5))[variation % 2]
        else:
            offsets = (0., 2., 3.5) if variation % 2 else (0., 1.5, 3.)
        for i, beat in enumerate(offsets):
            octave = busy and i == 5 and variation % 3 == 1
            pitch = root+12 if octave else root
            self.add('bass', bar*4+beat, pitch, .34 if busy else (1.25 if i == 0 else .6),
                     energy*(1. if i == 0 else .82), legato=bool(octave))

    def power(self, beat, chord, dur, vel, muted=False):
        for side, name in enumerate(('rhythm-L', 'rhythm-R')):
            for voice, (pitch, string) in enumerate(GRIPS[chord]):
                offset = Fraction(side, 64)+Fraction(voice, 100)
                self.add(name, Fraction(str(beat))+offset, pitch, Fraction(str(dur))-Fraction(voice, 100),
                         vel*(1., .84, .72)[voice]*(1. if side == 0 else .96),
                         string=string, palm_mute=muted, pitch_interpolation='sinc')

    def riff(self, bar, chord, energy=.8, open_chords=False, variation=0):
        if open_chords:
            pattern = ((0., 1.2, 1.), (1.5, .42, .78), (2., .7, .9), (3.25, .5, .98))
            for beat, dur, level in pattern:
                self.power(bar*4+beat, chord, dur, energy*level)
        else:
            patterns = ((0., .5, .75, 1.5, 2., 2.75, 3.25, 3.75),
                        (0., .75, 1.5, 2., 2.5, 3.25, 3.5),
                        (0., .5, 1.25, 1.5, 2.5, 3., 3.5),
                        (0., .75, 1.5, 2.25, 2.5, 3.25, 3.75))
            for i, beat in enumerate(patterns[variation % 4]):
                if beat in (1.5, 3.25):
                    self.power(bar*4+beat, chord, .43, energy)
                else:
                    pitch, string = GRIPS[chord][0]
                    for side, name in enumerate(('rhythm-L', 'rhythm-R')):
                        self.add(name, Fraction(bar*4)+Fraction(str(beat))+Fraction(side, 64), pitch, .16,
                                 energy*(1. if beat == 0 else .74+.025*(i % 3)),
                                 string=string, palm_mute=True, pitch_interpolation='sinc')

    def drums(self, bar, energy=.8, kind='rock', variation=0, fill=False):
        t = bar*4
        if kind == 'rock':
            for beat, amount in ((0., 1.), (.75, .73), (1.5, .9), (2.5, .8), (3.25, .91)):
                self.hit('kick', t+beat, 36, energy*amount)
            for beat in (1., 3.):
                self.hit('snare', t+beat, 38, energy*(.94 if beat == 1 else 1.))
            self.hit('snare', t+2.75, 38, energy*.25)
            for i in range(8):
                if fill and i >= 6:
                    continue
                self.hit('hats', t+i*.5, 46 if i == 7 and variation % 2 else 42,
                         energy*(.70 if i % 2 == 0 else .42))
            if variation % 4 == 0:
                self.hit('cymbals', t, 49, energy*.87)
        else:
            for beat, amount in ((0., .83), (1.5, .53)):
                self.hit('kick', t+beat, 36, energy*amount)
            self.hit('snare', t+2., 37 if kind == 'light' else 38, energy*.85)
            for i in range(8):
                self.hit('hats', t+i*.5, 42, energy*(.56 if i % 2 == 0 else .31))
        if fill:
            for i, pitch in enumerate((50, 47, 45)):
                self.hit('toms', Fraction(t+3)+Fraction(i, 3), pitch, energy*(.60+.16*i))

    def theme(self, bar, row, instrument='clean-melody', energy=.75, variant=0):
        cursor = Fraction(bar*4)
        for i, (pitch, duration) in enumerate(row):
            params = {}
            d = Fraction(str(duration))
            if instrument in ('lead', 'clean-melody'):
                params['string'] = string_for(pitch)
                if i > 0 and d <= Fraction(1, 2):
                    params['legato'] = True
                if instrument == 'lead' and pitch == 69 and d >= 1:
                    params.update(bend=2., bend_time=.2, bend_start=.06, vibrato=.055)
                elif d >= 1:
                    params['vibrato'] = .065 if instrument == 'lead' else .025
                params['pitch_interpolation'] = 'sinc'
            accent = 1. if i == 0 or d >= 1 else .83+.035*((i+variant) % 3)
            self.add(instrument, cursor, pitch, d*(Fraction(94, 100) if d >= 1 else Fraction(85, 100)),
                     energy*accent, **params)
            cursor += d

    def piano_support(self, bar, chord, energy=.48, rich=False):
        tones = VOICINGS[chord]
        for offset, index in ((0., 1), (.5, 2), (2., 3), (2.5, 2)):
            self.add('piano', bar*4+offset, tones[index], .62, energy*(1. if offset in (0., 2.) else .72))
        if rich:
            self.add('piano', bar*4+.025, tones[0], 1.3, energy*.68)

    def strings(self, bar, chord, energy=.5, upper=False):
        tones = VOICINGS[chord]
        cello_pitch = tones[0] if tones[0] <= 52 else tones[0]-12
        self.add('cello', bar*4+.04, cello_pitch, 3.6, energy, legato=bar > 0)
        if upper:
            pitch = min(76, tones[3]+12)
            self.add('violin', bar*4+.09, pitch, 3.5, energy*.9, legato=True)

    def finish(self, key='E minor'):
        bass = sorted(self.events['bass'], key=lambda item: item[0])
        for i, (at, note) in enumerate(bass[:-1]):
            limit = max(Fraction(1, 20), bass[i+1][0]-at-Fraction(3, 40))
            if note.dur > limit:
                bass[i] = (at, Note(note.pitch, limit, note.vel, note.params))
        self.events['bass'] = bass
        duration = self.bars*4
        layers = [self.tracks[name](place(events, length=duration)) for name, events in self.events.items() if events]
        markers = place(self.sections, length=duration)
        return Song(stack(*layers, markers), bpm=self.bpm, seed=83, sample_rate=44100,
                    loudness=-16., ceiling=-1., key=key, title=self.title)


def clean_song() -> Song:
    b = Band(48, 112, 'Свет на воде / Light on Water', 'clean')
    form = ((0, 4, 'Opening', 'opening'), (4, 8, 'First theme', 'verse'),
            (12, 8, 'Refrain', 'chorus'), (20, 8, 'Second theme', 'verse2'),
            (28, 8, 'Bridge', 'bridge'), (36, 8, 'Final refrain', 'final'), (44, 4, 'Coda', 'outro'))
    for start, length, name, role in form:
        b.mark(name, start, length)
        for j in range(length):
            bar = start+j
            harmony = ('Am9', 'Cmaj7', 'Dadd9', 'B7') if role == 'bridge' else MAIN
            if role == 'verse2':
                harmony = MAIN[4:]+MAIN[:4]
            chord = harmony[j % len(harmony)]
            if role == 'outro':
                chord = ('Cmaj7', 'Dadd9', 'Em9', 'Em9')[j]
            energy = .64 if role in ('opening', 'bridge') else (.76 if role in ('chorus', 'final') else .68)
            if role == 'outro':
                energy *= 1-j*.16
            if bar < 47:
                b.texture(bar, chord, energy, variation=j+(2 if role == 'verse2' else 0))
            if role != 'opening' and bar < 46:
                b.bass(bar, chord, .56 if role not in ('chorus', 'final') else .66, variation=j)
                b.drums(bar, .42 if role in ('verse', 'bridge') else .56, kind='light', variation=j,
                        fill=j == length-1 and role != 'bridge')
            if role in ('verse', 'chorus', 'verse2', 'final'):
                row = CLEAN_THEME[(j+(4 if role == 'verse2' else 0)) % 8]
                b.theme(bar, row, energy=.67 if role in ('verse', 'verse2') else .76, variant=j)
            elif role == 'bridge':
                # The cello takes the tune; the guitar answers in the second half.
                bridge_row = CLEAN_THEME[(6, 1, 3, 7)[j % 4]]
                if j % 2 == 0:
                    b.theme(bar, tuple((p-12, d) for p, d in bridge_row),
                            instrument='cello', energy=.53)
                else:
                    b.theme(bar, bridge_row, energy=.57)
            if role != 'bridge' and bar < 46:
                b.strings(bar, chord, .43 if role in ('opening', 'verse') else .50,
                          upper=role in ('chorus', 'final') and j % 2 == 0)
    b.theme(46, ((71, 1.), (67, 1.), (66, .75), (64, 1.25)), energy=.63)
    for i, pitch in enumerate((52, 59, 64, 67)):
        b.add('clean', 188+i*.05, pitch, 2.8-i*.05, .6-i*.05)
        b.add('acoustic', 188+i*.05, pitch, 2.5-i*.05, .43, string=string_for(pitch))
    b.add('cello', 188, 52, 2.8, .46, legato=True)
    return b.finish()


def rock_song() -> Song:
    b = Band(64, 128, 'Стальной пульс / Steel Pulse', 'rock')
    form = ((0, 4, 'Riff opening', 'intro'), (4, 8, 'Verse one', 'verse'),
            (12, 4, 'Lift', 'lift'), (16, 8, 'Chorus', 'chorus'), (24, 8, 'Verse two', 'verse2'),
            (32, 8, 'Heavy half-time', 'bridge'), (40, 8, 'Solo', 'solo'),
            (48, 12, 'Final chorus', 'final'), (60, 4, 'Ending', 'outro'))
    for start, length, name, role in form:
        b.mark(name, start, length)
        for j in range(length):
            bar = start+j
            harmony = CHORUS if role in ('chorus', 'solo', 'final') else MAIN
            if role == 'lift':
                harmony = ('Cmaj7', 'Am9', 'Dadd9', 'B7')
            if role == 'bridge':
                harmony = ('Am9', 'Cmaj7', 'Em9', 'B7')
            chord = harmony[j % len(harmony)]
            energy = .84 if role in ('chorus', 'final', 'solo') else .76
            if role == 'intro' and j == 0:
                b.power(bar*4, 'Em9', 1.1, .8)
                b.power(bar*4+2.5, 'Em9', .5, .68)
                b.bass(bar, 'Em9', .7)
                b.hit('kick', bar*4, 36, .9)
                continue
            if role == 'outro' and j >= 2:
                continue
            b.riff(bar, chord, energy, open_chords=role in ('chorus', 'solo', 'final', 'lift'), variation=j)
            b.bass(bar, chord, .78, busy=role != 'bridge', variation=j)
            b.drums(bar, .90 if role in ('chorus', 'final') else .8,
                    kind='half' if role == 'bridge' else 'rock', variation=j, fill=j % 4 == 3)
            if role in ('chorus', 'solo', 'final'):
                b.theme(bar, ROCK_THEME[j % 8], instrument='lead', energy=.83 if role != 'solo' else .87,
                        variant=j)
            elif role in ('verse', 'verse2') and j % 4 == 3:
                b.theme(bar, ((67, 1.), (66, .5), (64, 1.5), (62, 1.)), instrument='lead', energy=.62)
            elif role == 'bridge' and j % 2 == 1:
                b.theme(bar, ((64, 1.), (67, .5), (69, .5), (71, 2.)), instrument='lead', energy=.72)
    b.power(248, 'Em9', 2.2, .92)
    b.add('bass', 248, 28, 2.2, .86, slide=7., slide_time=.065)
    b.add('lead', 248, 76, 2.4, .85, string=1, vibrato=.08)
    b.hit('kick', 248, 36, .96)
    b.hit('snare', 248, 38, .80)
    b.hit('cymbals', 248, 49, .85)
    return b.finish()


def soft_song() -> Song:
    b = Band(48, 96, 'Тёплый свет / Warm Light', 'soft')
    form = ((0, 4, 'Piano opening', 'intro'), (4, 8, 'First melody', 'verse'),
            (12, 8, 'Refrain', 'chorus'), (20, 8, 'Second melody', 'verse2'),
            (28, 4, 'Dominant bridge', 'bridge'), (32, 12, 'Final refrain', 'final'),
            (44, 4, 'Quiet coda', 'outro'))
    for start, length, name, role in form:
        b.mark(name, start, length)
        for j in range(length):
            bar = start+j
            harmony = ('Dm9', 'Fmaj7', 'Bhalf', 'E7') if role == 'bridge' else SOFT
            if role == 'verse2':
                harmony = SOFT[4:]+SOFT[:4]
            chord = harmony[j % len(harmony)]
            if role == 'outro':
                chord = ('Fmaj7', 'Dm9', 'E7', 'Am9')[j]
            energy = .50 if role in ('intro', 'bridge', 'outro') else (.64 if role in ('chorus', 'final') else .56)
            b.piano_support(bar, chord, energy*.73, rich=role in ('chorus', 'final'))
            if role != 'intro':
                b.texture(bar, chord, .40 if role in ('verse', 'outro') else .48, variation=j,
                          acoustic=role in ('chorus', 'final'))
            row = SOFT_THEME[(j+(4 if role == 'verse2' else 0)) % 8]
            if role not in ('bridge', 'outro'):
                b.theme(bar, row, instrument='piano', energy=energy, variant=j)
            elif role == 'bridge':
                b.theme(bar, ((65, 1.), (64, .5), (62, .5), (68 if j == 3 else 64, 2.)),
                        instrument='piano', energy=.55)
            if bar >= 8 and role != 'outro':
                b.bass(bar, chord, .53, variation=j)
                b.drums(bar, .40 if role not in ('chorus', 'final') else .53, kind='half', variation=j,
                        fill=j == length-1 and role != 'bridge')
            if bar < 47:
                b.strings(bar, chord, .43 if role not in ('chorus', 'final') else .52,
                          upper=role in ('chorus', 'final') and j % 2 == 0)
    for i, pitch in enumerate((45, 52, 60, 64, 69)):
        b.add('piano', 188+i*.06, pitch, 2.8-i*.06, .58-.04*i)
    b.add('cello', 188, 45, 2.7, .43, legato=True)
    return b.finish('A minor')


def extended_song() -> Song:
    b = Band(80, 128, 'Искра — полная история / Iron Current Extended', 'suite')
    b.mark('Clean opening and theme', 0, 12)
    for bar in range(12):
        chord = MAIN[bar % 8]
        b.texture(bar, chord, .66, variation=bar)
        b.strings(bar, chord, .45, upper=False)
        if 4 <= bar < 10:
            b.theme(bar, CLEAN_THEME[(bar-4) % 8], energy=.63)
        if bar >= 8:
            b.bass(bar, chord, .48+.04*(bar-8), variation=bar)
            b.drums(bar, .35+.05*(bar-8), kind='light', variation=bar, fill=bar == 11)
        if bar >= 10:
            b.riff(bar, chord, .40+.06*(bar-10), variation=bar)
    b.mark('Riff development', 12, 16)
    for j in range(16):
        bar = 12+j
        chord = MAIN[j % 8]
        b.riff(bar, chord, .80 if j < 8 else .84, variation=j)
        b.bass(bar, chord, .78, busy=True, variation=j)
        b.drums(bar, .83, variation=j, fill=j % 4 == 3)
        if j % 4 == 3:
            b.theme(bar, ((67, 1.), (66, .5), (64, 1.5), (62, 1.)), instrument='lead', energy=.65)
    b.mark('Pivot into the interlude', 28, 4)
    for j, chord in enumerate(('Cmaj7', 'Dadd9', 'Em9', 'E7')):
        bar = 28+j
        b.riff(bar, chord, .83-.11*j, open_chords=True, variation=j)
        b.bass(bar, chord, .73-.06*j, variation=j)
        b.drums(bar, .80-.10*j, kind='rock' if j < 2 else 'half', variation=j)
        if j >= 2:
            b.texture(bar, chord, .42+.06*(j-2), variation=j)
        pivot_row = CLEAN_THEME[j] if j < 3 else ((64, 1.), (68, .75), (71, .75),
                                                (74, .5), (71, .5), (68, .5))
        b.theme(bar, pivot_row, instrument='lead', energy=.68-.07*j)
    b.add('piano', 127.5, 64, .4, .43)
    b.mark('Piano interlude and rebuild', 32, 16)
    interlude = ('Am9', 'B7', 'Em9', 'Cmaj7', 'Am9', 'B7', 'Cmaj7', 'B7')
    for j in range(16):
        bar = 32+j
        chord = interlude[j % 8]
        b.texture(bar, chord, .45 if j < 12 else .52, variation=j)
        b.piano_support(bar, chord, .43, rich=j >= 8)
        b.theme(bar, CLEAN_THEME[(6, 7, 0, 1, 6, 7, 1, 7)[j % 8]], instrument='piano', energy=.56)
        b.strings(bar, chord, .46, upper=j >= 8)
        b.bass(bar, chord, .57 if j < 12 else .67, variation=j)
        b.drums(bar, .47 if j < 12 else .55+.055*(j-12), kind='half' if j < 14 else 'rock',
                variation=j, fill=j in (7, 15))
        if j >= 14:
            b.riff(bar, chord, .45+.09*(j-14), open_chords=True, variation=j)
        if j == 15:
            b.theme(bar, ((66, 1.), (69, .5), (71, .5), (75, 1.), (66, 1.)),
                    instrument='lead', energy=.63)
    b.mark('Rock return and melodic solo', 48, 16)
    b.mark('Final hook', 64, 8)
    for j in range(24):
        bar = 48+j
        chord = CHORUS[j % 8]
        b.riff(bar, chord, .85 if j < 16 else .91, open_chords=True, variation=j)
        b.bass(bar, chord, .81, busy=True, variation=j)
        b.drums(bar, .89 if j < 16 else .94, variation=j, fill=j % 4 == 3)
        b.theme(bar, ROCK_THEME[j % 8], instrument='lead', energy=.84 if j < 16 else .89, variant=j)
        if j >= 8:
            b.strings(bar, chord, .50, upper=True)
        if j >= 16 and j % 2 == 0:
            b.texture(bar, chord, .40, variation=j)
    b.mark('Riff answer and circular ending', 72, 8)
    for j in range(6):
        bar = 72+j
        chord = ('Em9', 'Cmaj7', 'G6', 'Dadd9', 'Cmaj7', 'B7')[j]
        b.riff(bar, chord, .84-.035*j, variation=j)
        b.bass(bar, chord, .76, busy=True, variation=j)
        b.drums(bar, .84-.035*j, variation=j, fill=j == 5)
        if j >= 2:
            b.texture(bar, chord, .50, variation=j)
            b.theme(bar, CLEAN_THEME[(j-2) % 8], energy=.62)
    b.power(312, 'Em9', 2.2, .92)
    b.add('bass', 312, 28, 2.2, .86, slide=7., slide_time=.065)
    b.add('lead', 312, 76, 2.4, .84, string=1, vibrato=.08)
    b.hit('kick', 312, 36, .96)
    b.hit('snare', 312, 38, .77)
    b.hit('cymbals', 312, 49, .86)
    b.texture(78, 'Em9', .53, variation=0)
    b.add('piano', 312.06, 64, 2.6, .52)
    b.add('violin', 312, 67, 2.7, .52, legato=True)
    b.add('cello', 312, 52, 2.7, .52, legato=True)
    return b.finish()


BUILDERS = {'clean': clean_song, 'rock': rock_song, 'soft': soft_song, 'extended': extended_song}


def build_song(version: str = 'extended') -> Song:
    return BUILDERS[version]()


song = build_song()
