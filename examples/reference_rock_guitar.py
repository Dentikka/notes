"""Procedural bridge guitar with a reference-fitted cabinet correction.

Six mathematical peak filters were estimated from three FreePats FSBS DI/amp
note pairs. They improve spectral profiles beyond those notes, with exceptions
under the auxiliary MERT diagnostic. No recordings or measured IRs are loaded.
The baseline keeps the same source, amplifier, notes and excitation gain.
"""
from notes import ElectricGuitar, Filter, Note, Rest, Song, Track, cat

CABINET_BANDS = (
    (125., .8, 6.1286038253905115),
    (350., .8, -2.708125410023477),
    (1000., .8, -6.569733522175824),
    (2800., .8, 2.4927786736061686),
    (6500., .8, -2.3913015328234892),
    (1250., 3., -17.415297003686614),
)


def cabinet_correction() -> tuple[Filter, ...]:
    """Optional scalar peak filters after the existing amplifier and cabinet."""
    return tuple(Filter(kind="peak", cutoff=center, q=q, gain_db=gain) for center, q, gain in CABINET_BANDS)


def build_song(*, baseline: bool = False) -> Song:
    instrument = ElectricGuitar(drive=.65, tone=.5, level=.5, fretting=True,
                                pickup_position=.06, pickup_width=.04, pickup_q=.7)
    track = Track("guitar", instrument, fx=() if baseline else cabinet_correction())
    phrase = cat(*(cat(Note(pitch, dur=.9, vel=.7).param(string=string), Rest(.3))
                   for pitch, string in ((41, 6), (45, 5), (50, 4), (55, 3), (61, 2), (74, 1))))
    return Song(track(phrase), bpm=60, sample_rate=44100, seed=17, loudness=-22,
                title="Rock guitar: " + ("baseline chain" if baseline else "reference cabinet correction"))


song = build_song()
