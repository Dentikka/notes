"""Built-in instruments; importing this package registers them."""

from notes.engine.instruments.acoustic import AcousticGuitar
from notes.engine.instruments.drums import DrumKit
from notes.engine.instruments.fm import FM, Bell, EPiano
from notes.engine.instruments.noise import Noise
from notes.engine.instruments.rock_drums import RockKit
from notes.engine.instruments.strings import BassGuitar, ElectricGuitar, Pluck
from notes.engine.instruments.synth import Bass, Lead, Pad, Synth

__all__ = [
    "FM",
    "AcousticGuitar",
    "Bass",
    "BassGuitar",
    "Bell",
    "DrumKit",
    "EPiano",
    "ElectricGuitar",
    "Lead",
    "Noise",
    "Pad",
    "Pluck",
    "RockKit",
    "Synth",
]
