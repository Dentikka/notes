"""First Groove — a one-minute demo of the notes DSL.

It shows the moves from the project brief, each one a function: a steady beat, muting it
for a bar, a melody on an electric guitar with a chosen tone, and a chorus played twice.

    notes render examples/first_groove.py -o <output dir>
"""

from notes import (
    Bass,
    Bell,
    Chorus,
    Delay,
    DrumKit,
    E,
    ElectricGuitar,
    EPiano,
    Key,
    Pad,
    Reverb,
    S,
    Song,
    Track,
    W,
    chord,
    hit,
    melody,
    stack,
    steps,
)

key = Key.parse("A minor")

# --- tracks: an instrument plus its mixer strip ----------------------------------------
drums = Track("drums", DrumKit(punch=0.6), gain=-2)
bass = Track("bass", Bass())
keys = Track("keys", EPiano(), fx=[Chorus(mix=0.4), Reverb(size=0.5, mix=0.3)], gain=-2.5, pan=-0.2)
pad = Track("pad", Pad(), fx=[Reverb(size=0.85, mix=0.6)], gain=-9.5)
rhythm = Track("rhythm", ElectricGuitar(drive=0.85, tone=0.45, pickup="bridge"), gain=1, pan=-0.45)
lead = Track(
    "lead",
    ElectricGuitar(drive=0.65, tone=0.6, pickup="neck"),
    fx=[Delay(time=0.75, feedback=0.3, mix=0.22), Reverb(size=0.6, mix=0.3)],
    gain=1.5,
    pan=0.3,
)
bell = Track("bell", Bell(), fx=[Reverb(size=0.9, mix=0.6)], gain=-9.5, pan=0.15)

# --- material ----------------------------------------------------------------------------
harmony = key.chords("i VI III VII", voicing="smooth")  # Am F C G, a bar each
roots = melody("A2 F2 C3 G2", step=W)

groove = steps(kick="x.....x...x.....", snare="....x.......x..o", hat="x.x.x.x.x.x.x.x.")
fill = steps(kick="x.....x.x.......", snare="....x...x.xxXxXX", hat="x.x.x.x.........")
drive = steps(kick="x.....x.x.x.....", snare="....x.......x...", oh="..x...x...x...x.")

hook = melody(
    """
    A4 C5 E5 _  D5 C5 D5 E5 | F5 _ E5 D5 C5 _ A4 _  |
    G4 C5 E5 _  D5 C5 D5 E5 | D5 _ _  _  B4 _ G4 _  |
    A4 C5 E5 _  D5 C5 D5 E5 | F5 _ E5 D5 C5 _ A4 C5 |
    E5 _  D5 C5 G4 _  C5 D5 | E5 _ _  _  _  _ .  .
    """,
    step=E,
)
power_chords = melody("[A2,E3,A3]@8 [F2,C3,F3]@8 [C3,G3,C4]@8 [G2,D3,G3]@8", step=E)

# --- sections ----------------------------------------------------------------------------
# A steady beat is a function; muting its last bar is another.
intro = pad(harmony) | drums((groove * 4).mute(bars=4))

verse = stack(
    drums((groove * 8).every(4, fill)),
    bass((roots * 2).rhythm("x..x..x.x..x..x.", step=S)),
    keys((harmony * 2).rhythm("x_____x_x_______", step=S)),
    pad((harmony * 2).velocity(0.6)),
)

# The melody on a driven neck-pickup guitar; the rhythm guitar chugs palm-muted power chords.
chorus = stack(
    drums((drive * 8).every(4, fill) | hit("crash", W)),
    bass((roots * 2).rhythm("x...x...x...x...", step=S) | (roots * 2).octave(1).rhythm("..x...x...x...x.", step=S)),
    rhythm((power_chords * 2).rhythm("XxXxXxXx", step=E).param(palm_mute=True)),
    lead(hook),
    pad(harmony * 2),
)

ending = key.chords("i VI III VII i@2", voicing="smooth")  # the progression, resolved home
outro = stack(
    pad(ending),
    keys(ending.velocity(0.7)),
    bell(harmony.octave(1).arp("updown", step=E) + chord("Am", 2 * W, octave=5).arp("up", step=E)),
    drums(hit("crash", W)),
)

# Playing the chorus twice is just `* 2`.
song = Song(
    intro.named("intro"),
    verse.named("verse"),
    chorus.named("chorus") * 2,
    outro.named("outro"),
    bpm=112,
    key="A minor",
    title="First Groove",
)

if __name__ == "__main__":
    song.render("first_groove.wav")
