"""Ears: how a render sounds to a model (CLAP, needs ``pip install 'notes[ears]'``), and how it
measures up against a reference recording (`compare`, numpy and scipy only)."""

from notes.ears.aesthetics import AestheticsEar, fetch_aesthetics
from notes.ears.clap import CLAP_SAMPLE_RATE, DEFAULT_MODEL, Ear, Tag, cosine, format_report, to_clap_rate, windows
from notes.ears.compare import ANALYSIS_RATE, Comparison, compare, format_comparison, listening, load
from notes.ears.excitation_fit import ExcitationBasis, ExcitationGainFit, excitation_basis, fit_excitation_gains
from notes.ears.fit import FitResult, fit_loss, fit_reference
from notes.ears.idmt import idmt_music
from notes.ears.inharmonicity import InharmonicPartial, StringInharmonicity, string_inharmonicity
from notes.ears.mert import MertEar
from notes.ears.modal_pluck_fit import ModalPluckGeometryFit, ModalPluckObservation, ModalPluckTrial, fit_modal_pluck
from notes.ears.note_profile import PartialDecay, harmonic_shape_distance, infer_expression_points, partial_decays
from notes.ears.partial_envelopes import PartialEnvelopes, PartialGainFit, fit_partial_gains, partial_envelopes
from notes.ears.percussion import BandPeakTrajectory, band_peak_trajectory, compare_peak_trajectories
from notes.ears.polyphonic import TemplateTrack, notes_from_templates, template_track
from notes.ears.texture import (
    InterharmonicEnergy,
    RoughnessEstimate,
    TextureFitResult,
    estimate_roughness,
    fit_roughness,
    interharmonic_energy,
)
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
    "BandPeakTrajectory",
    "AestheticsEar",
    "Ear",
    "ExcitationBasis",
    "ExcitationGainFit",
    "FitResult",
    "InharmonicPartial",
    "InterharmonicEnergy",
    "KnownNoteTuning",
    "MertEar",
    "ModalPluckGeometryFit",
    "ModalPluckObservation",
    "ModalPluckTrial",
    "PartialDecay",
    "PartialEnvelopes",
    "PartialGainFit",
    "RoughnessEstimate",
    "TextureFitResult",
    "StringInharmonicity",
    "Tag",
    "TemplateTrack",
    "compare",
    "band_peak_trajectory",
    "compare_peak_trajectories",
    "cosine",
    "fit_loss",
    "excitation_basis",
    "fit_excitation_gains",
    "fit_modal_pluck",
    "fit_partial_gains",
    "fetch_aesthetics",
    "fit_reference",
    "fit_roughness",
    "estimate_roughness",
    "format_comparison",
    "format_report",
    "listening",
    "infer_expression_points",
    "harmonic_shape_distance",
    "idmt_music",
    "known_note_tuning",
    "interharmonic_energy",
    "load",
    "notes_from_templates",
    "partial_decays",
    "partial_envelopes",
    "string_inharmonicity",
    "template_track",
    "to_clap_rate",
    "tracked_music",
    "windows",
]
