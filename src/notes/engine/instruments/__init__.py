"""Built-in instruments; importing this package registers them."""

from notes.engine.instruments.drums import DrumKit
from notes.engine.instruments.fm import FM, Bell, EPiano
from notes.engine.instruments.rock_drums import RockKit
from notes.engine.instruments.strings import BassGuitar, ElectricGuitar, Pluck
from notes.engine.instruments.synth import Bass, Lead, Pad, Synth

__all__ = [
    "FM",
    "Bass",
    "BassGuitar",
    "Bell",
    "DrumKit",
    "EPiano",
    "ElectricGuitar",
    "Lead",
    "Pad",
    "Pluck",
    "RockKit",
    "Synth",
]
