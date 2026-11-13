"""Short source study: experimental cello and violin geometry, no recordings.

Bare mode pitch is not a calibrated sounding pitch. This demonstrates contact,
string selection and expression controls; it is not a realism reference preset.
"""
from notes import ModalBowedString, Song, Track, cat, melody

cello = ModalBowedString()
violin = ModalBowedString(open_pitches=(55, 62, 69, 76), length=.328,
                         tension=55., body_size=1., program=40)
phrase = melody('G2 A2 Bb2 D3', step=.75).velocity(.8)
phrase = phrase.param(expression_points=((0., .25), (.12, 1.), (.4, .55)))
song = Song(cat(Track('cello', cello)(phrase).named('cello'),
                Track('violin', violin)(phrase.transpose(12)).named('violin')),
            bpm=90, sample_rate=22050, seed=17, loudness=-22,
            title='Modal bowed source study')
