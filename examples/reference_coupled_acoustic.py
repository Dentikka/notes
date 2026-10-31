"""Optional coupled string/body study, with mathematically designed resonances.

Fixed settings passed nine unseen reference notes at two excitation seeds.
Low E2 still regresses on development data; this is not a universal realism preset.
"""
from notes import AcousticGuitar, Note, Rest, Song, Track, cat

guitar = Track("acoustic", AcousticGuitar(model="coupled", fretting=True, sustain=6.,
                                         body_size=1.1, diffuse=.4, coupling=.4, radiation=.25,
                                         pick_position=.15, polarization=0., pick_noise=0.))
phrase = cat(*(cat(Note(pitch, dur=.9, vel=velocity), Rest(.25))
               for pitch, velocity in ((40, .65), (44, .65), (50, .65), (59, .85),
                                       (64, .85), (67, .85), (73, .65), (76, .85), (80, .85))))
song = Song(guitar(phrase), bpm=60, seed=17, loudness=-22,
            title="Coupled acoustic guitar: register study")
