"""The audio back-end: renders an IR `Score` with a built-in numpy/scipy DSP engine."""

from notes.engine.audio import read_wav, write_wav
from notes.engine.base import Effect, Instrument, RenderContext, Voice
from notes.engine.effects import EQ, Cabinet, Chorus, Compressor, Delay, Drive, Filter, GuitarAmp, Reverb, Room
from notes.engine.instruments import (
    FM,
    Bass,
    BassGuitar,
    Bell,
    DrumKit,
    ElectricGuitar,
    EPiano,
    Lead,
    Noise,
    Pad,
    Pluck,
    RockKit,
    Synth,
)
from notes.engine.mixer import RenderResult, render_score
from notes.engine.registry import (
    EFFECTS,
    INSTRUMENTS,
    build_effect,
    build_instrument,
    register_effect,
    register_instrument,
)

__all__ = [
    "EFFECTS",
    "EQ",
    "FM",
    "INSTRUMENTS",
    "Bass",
    "BassGuitar",
    "Bell",
    "Cabinet",
    "Chorus",
    "Compressor",
    "Delay",
    "Drive",
    "DrumKit",
    "EPiano",
    "Effect",
    "ElectricGuitar",
    "Filter",
    "GuitarAmp",
    "Instrument",
    "Lead",
    "Noise",
    "Pad",
    "Pluck",
    "RockKit",
    "Reverb",
    "Room",
    "RenderContext",
    "RenderResult",
    "Synth",
    "Voice",
    "build_effect",
    "build_instrument",
    "read_wav",
    "register_effect",
    "register_instrument",
    "render_score",
    "write_wav",
]
