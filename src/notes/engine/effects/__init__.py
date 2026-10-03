"""Built-in effects; importing this package registers them."""

from notes.engine.effects.dynamics import Compressor
from notes.engine.effects.room import Room
from notes.engine.effects.space import Chorus, Delay, Reverb
from notes.engine.effects.tone import EQ, Cabinet, Drive, Filter, GuitarAmp

__all__ = ["EQ", "Cabinet", "Chorus", "Compressor", "Delay", "Drive", "Filter", "GuitarAmp", "Reverb", "Room"]
