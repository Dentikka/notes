"""Ears: how a render sounds to a model. Optional; needs ``pip install 'notes[ears]'``."""

from notes.ears.clap import CLAP_SAMPLE_RATE, DEFAULT_MODEL, Ear, Tag, cosine, format_report, to_clap_rate, windows
from notes.ears.vocab import TEMPLATES, VOCABULARY

__all__ = [
    "CLAP_SAMPLE_RATE",
    "DEFAULT_MODEL",
    "TEMPLATES",
    "VOCABULARY",
    "Ear",
    "Tag",
    "cosine",
    "format_report",
    "to_clap_rate",
    "windows",
]
