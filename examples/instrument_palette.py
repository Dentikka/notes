"""Solo instrument palette. Each named section has the same deliberate timing."""
from notes import (
    AcousticGuitar,
    BassGuitar,
    Cello,
    ElectricGuitar,
    Piano,
    Reverb,
    RockKit,
    Song,
    Track,
    Violin,
    cat,
    hit,
    melody,
    stack,
)

instruments = {
    "piano": Piano(),
    "violin": Violin(),
    "cello": Cello(),
    "clean-guitar": ElectricGuitar(drive=0.06, pickup="neck", tone=0.3),
    "rock-guitar": ElectricGuitar(drive=0.65),
    "acoustic-guitar": AcousticGuitar(),
    "bass": BassGuitar(),
    "drums": RockKit(),
}


def build_song(name):
    instrument = instruments[name]
    track = Track(name, instrument, fx=[Reverb(size=0.35, mix=0.12)])
    if name == "drums":
        music = stack(hit("kick").rhythm("X...X.X.", step=0.5),
                      hit("snare").rhythm("..X...X.", step=0.5),
                      hit("hat").rhythm("XxXxXxXx", step=0.5)) * 2
    elif name == "piano":
        music = melody("[A2,E3,A3,C4]@2 E4 C4 [F2,C3,A3,C4]@2 D4 E4", step=0.5)
    elif name in ("violin", "cello"):
        music = melody("A3@2 C4 D4 E4@2 D4 C4 A3@2", step=0.5)
        if name == "cello":
            music = music.transpose(-12)
    else:
        music = melody("A3 E4 C4 E4 G3 D4 B3 D4", step=0.5) * 2
        if name == "bass":
            music = music.transpose(-24)
    return Song(track(music), bpm=100, seed=17, loudness=-19, title=name)


song = Song(cat(*(build_song(name).music.named(name) for name in instruments)),
            bpm=100, seed=17, loudness=-19, title="Instrument palette")
