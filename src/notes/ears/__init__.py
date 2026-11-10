"""Ears: how a render sounds to a model (CLAP, needs ``pip install 'notes[ears]'``), and how it
measures up against a reference recording (`compare`, numpy and scipy only)."""

from notes.ears.aesthetics import AestheticsEar, fetch_aesthetics
from notes.ears.clap import CLAP_SAMPLE_RATE, DEFAULT_MODEL, Ear, Tag, cosine, format_report, to_clap_rate, windows
from notes.ears.compare import ANALYSIS_RATE, Comparison, compare, format_comparison, listening, load
from notes.ears.fit import FitResult, fit_loss, fit_reference
from notes.ears.inharmonicity import InharmonicPartial, StringInharmonicity, string_inharmonicity
from notes.ears.mert import MertEar
from notes.ears.note_profile import PartialDecay, partial_decays
from notes.ears.polyphonic import TemplateTrack, notes_from_templates, template_track
from notes.ears.transcribe import tracked_music
from notes.ears.tuning import KnownNoteTuning, known_note_tuning
from notes.ears.vocab import TEMPLATES, VOCABULARY

__all__ = [
    "ANALYSIS_RATE",
    "CLAP_SAMPLE_RATE",
    "DEFAULT_MODEL",
    "TEMPLATES",
    "VOCABULARY",
    "Comparison",
    "AestheticsEar",
    "Ear",
    "FitResult",
    "InharmonicPartial",
    "KnownNoteTuning",
    "MertEar",
    "PartialDecay",
    "StringInharmonicity",
    "Tag",
    "TemplateTrack",
    "compare",
    "cosine",
    "fit_loss",
    "fetch_aesthetics",
    "fit_reference",
    "format_comparison",
    "format_report",
    "listening",
    "known_note_tuning",
    "load",
    "notes_from_templates",
    "partial_decays",
    "string_inharmonicity",
    "template_track",
    "to_clap_rate",
    "tracked_music",
    "windows",
]
