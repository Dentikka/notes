"""General MIDI conventions shared by the notation, the drum synth and the MIDI exporter.

Drum hits travel through the IR as ordinary notes whose pitch is the General MIDI drum
number and whose ``sound`` parameter names the sound; the parameter also marks them as
unpitched, so pitch transformations leave them alone.
"""

from __future__ import annotations

__all__ = ["GM_DRUMS", "GM_DRUM_NAMES", "drum_number"]

#: Canonical drum sound names and their General MIDI (channel 10) note numbers.
GM_DRUMS: dict[str, int] = {
    "kick": 36,
    "rim": 37,
    "snare": 38,
    "clap": 39,
    "hat": 42,
    "pedal_hat": 44,
    "open_hat": 46,
    "tom_lo": 45,
    "tom_mid": 47,
    "tom_hi": 50,
    "crash": 49,
    "ride": 51,
    "cowbell": 56,
}

#: Short aliases in the TidalCycles/Strudel tradition.
_ALIASES: dict[str, str] = {
    "bd": "kick",
    "rs": "rim",
    "sd": "snare",
    "cp": "clap",
    "hh": "hat",
    "ch": "hat",
    "ph": "pedal_hat",
    "oh": "open_hat",
    "lt": "tom_lo",
    "mt": "tom_mid",
    "ht": "tom_hi",
    "cr": "crash",
    "rd": "ride",
    "cb": "cowbell",
}

#: Reverse map: GM note number -> canonical name.
GM_DRUM_NAMES: dict[int, str] = {number: name for name, number in GM_DRUMS.items()}


def drum_number(name: str) -> tuple[str, int]:
    """Resolve a drum name or alias ('kick', 'bd', 'oh', ...) to (canonical name, GM note)."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in GM_DRUMS:
        known = ", ".join([*GM_DRUMS, *_ALIASES])
        raise ValueError(f"Unknown drum sound {name!r}. Known: {known}")
    return key, GM_DRUMS[key]
