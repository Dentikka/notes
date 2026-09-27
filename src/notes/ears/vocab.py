"""Label vocabularies and prompt templates for zero-shot listening.

Each category is scored separately; a label's text embedding is the mean over its
templates (prompt ensembling), which makes zero-shot ranking less sensitive to wording.
"""

from __future__ import annotations

__all__ = ["TEMPLATES", "VOCABULARY"]

VOCABULARY: dict[str, tuple[str, ...]] = {
    "genre": (
        "electronic", "synthwave", "rock", "pop", "hip hop", "jazz", "classical", "ambient", "metal", "funk",
        "disco", "house", "techno", "drum and bass", "lo-fi", "folk", "blues", "reggae", "cinematic",
        "video game music", "chiptune",
    ),
    "instrument": (
        "drums", "drum machine", "bass guitar", "synth bass", "electric guitar", "distorted electric guitar",
        "clean electric guitar", "acoustic guitar", "harp", "piano", "electric piano", "organ",
        "synthesizer pad", "synth lead", "bells", "strings", "brass", "saxophone", "flute", "vocals",
    ),
    "mood": (
        "energetic", "calm", "dark", "happy", "sad", "melancholic", "aggressive", "dreamy", "epic",
        "relaxing", "tense", "uplifting",
    ),
    "production": (
        "a professional studio recording", "an amateur home recording", "a live concert recording",
        "a synthetic MIDI rendition", "a lo-fi recording", "an 8-bit chiptune",
    ),
}

TEMPLATES: dict[str, tuple[str, ...]] = {
    "genre": ("This is a {} song.", "This is {} music."),
    "instrument": ("This is the sound of {}.", "A recording of {}.", "Music played on {}."),
    "mood": ("This is a {} song.", "The mood of this music is {}."),
    "production": ("This is {}.",),
}
