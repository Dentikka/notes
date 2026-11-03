"""Limited A4/D5/E5 contact study, entirely procedural.

Development anchors were A4 and E5. D5 and E5 soft/firm improved reference MERT
and harmonic profiles; G5 regressed, so this is not a universal treble preset.
The accepted hammer_damped default is retained.
"""
from notes import Note, Piano, Rest, Song, Track, cat


def build_song(*, candidate: bool = True) -> Song:
    parts = []
    for pitch, mass in ((69, 1.7), (74, 1.7+5/7*1.5), (76, 3.2)):
        note = Note(pitch, dur=1.1, vel=.65)
        if candidate:
            note = note.param(hammer_mass_scale=mass, stiffness=.0004)
        parts.append(cat(note, Rest(.25)))
    return Song(Track("piano", Piano(model="hammer_damped"))(cat(*parts)),
                bpm=60, seed=17, sample_rate=44100, loudness=-22,
                title="Piano contact study: " + ("candidate" if candidate else "baseline"))


song = build_song()
