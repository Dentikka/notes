"""Two-bar procedural contact-string phrase, using an experimental scalar profile."""
from notes import ModalString, Note, Song, Track, place

scale = 2**(-5/12)
guitar = ModalString(modes=128, stiffness=4e-5*scale*scale, bridge_mobility=0,
    sustain=30., treble_decay=2.5, fretting=True, pick_position=.2986627951218785*scale,
    pickup_position=.1807003796283706*scale, pickup_width=.04, pickup_q=.7,
    pick_cutoff=1000., pluck_rounding='spatial', palm_resistance=.03598244019612079,
    palm_position=.11059963939498499*scale, palm_width=.08*scale,
    pick_noise=0., pluck_force=1., level=.15)

# All notes use the fifth fret; velocity and contact pressure articulate the phrase.
strokes = [
    (0., 50, 5, .65, .84, .018), (.75, 55, 4, .55, .62, .036),
    (1.5, 60, 3, .55, .70, .018), (2.25, 64, 2, .55, .64, .012),
    (3., 69, 1, .70, .80, .008), (4., 64, 2, .55, .71, .018),
    (4.75, 60, 3, .55, .62, .036), (5.5, 55, 4, .55, .72, .060),
    (6.25, 50, 5, .55, .81, .100), (7., 50, 5, .85, .68, .018),
]
part = place([(at, Note(pitch, duration, vel=velocity).param(string=string, palm_mute=True,
               palm_resistance=resistance)) for at, pitch, string, duration, velocity, resistance in strokes], length=8)
song = Song(Track('guitar', guitar)(part), bpm=100, sample_rate=44100, seed=117,
            loudness=-22., title='Distributed palm contact / direct string')
