"""Ears: how a render sounds to a model (CLAP, needs ``pip install 'notes[ears]'``), and how it
measures up against a reference recording (`compare`, numpy and scipy only)."""

from notes.ears.clap import CLAP_SAMPLE_RATE, DEFAULT_MODEL, Ear, Tag, cosine, format_report, to_clap_rate, windows
from notes.ears.compare import ANALYSIS_RATE, Comparison, compare, format_comparison, listening, load
from notes.ears.vocab import TEMPLATES, VOCABULARY

__all__ = [
    "ANALYSIS_RATE",
    "CLAP_SAMPLE_RATE",
    "DEFAULT_MODEL",
    "TEMPLATES",
    "VOCABULARY",
    "Comparison",
    "Ear",
    "Tag",
    "compare",
    "cosine",
    "format_comparison",
    "format_report",
    "listening",
    "load",
    "to_clap_rate",
    "windows",
]
