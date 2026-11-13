"""String-conditioned medium-dynamic violin colouring, without recordings.

The two cuts were fitted on three isolated notes. Applying them to every string
regressed D-string notes; this recipe uses them only on G/A. It is a reference-
specific candidate, not a replacement of the general Violin preset. A separate
four-note test improved both corrected notes and preserved the two other strings.
"""
from typing import Literal

from notes import Music, Song, Track, Violin, cat, melody

BODY_EQ = (
    (201.80698982080165, 4.167141439100775, -35.99983684228459),
    (495.06552545529604, 11.999999999993051, -8.956455446225995),
)


def on_string(music: Music, string: Literal['G', 'D', 'A', 'E']) -> Music:
    """The played string is explicit; pitch alone cannot distinguish A4 on D/A."""
    if string not in ('G', 'D', 'A', 'E'):
        raise ValueError('string must be G, D, A or E')
    return music.param(body_eq=BODY_EQ if string in ('G', 'A') else ())


phrase = cat(
    on_string(melody('G3@2 A3 B3 A3 G3@2', step=.5), 'G').named('G string'),
    on_string(melody('D4@2 E4 F4 E4 D4@2', step=.5), 'D').named('D string'),
    on_string(melody('A4@2 B4 C5 B4 A4@2', step=.5), 'A').named('A string'),
    on_string(melody('E5@2 F5 G5 A5@2 G5 E5@2', step=.5), 'E').named('E string'),
).velocity(.8125)

song = Song(Track('violin', Violin(vibrato=.1))(phrase), bpm=84, sample_rate=22050,
            seed=17, loudness=-22, title='Violin register study')
