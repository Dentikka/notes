"""Independent generated PCM, unknown phases, honest fit diagnostics and bad data."""

from dataclasses import replace

import numpy as np
import pytest
from notes import Filter
from notes.ears import ModalPluckObservation, fit_modal_pluck, partial_envelopes
from notes.engine import RenderContext
from notes.engine.instruments.strings import fret_scale
from notes.engine.mechanical_string import _mechanics
from notes.engine.palm_contact import palm_string


def observations(reference="open", source_sr=22050):
    sr = source_sr
    items = []
    for pitch in [64, 71, 74]:
        f = 440 * 2 ** ((pitch - 69) / 12)
        scale = fret_scale(pitch, 1)
        y = palm_string(
            f,
            sr,
            sr,
            position=0.08,
            contact_width=0.04,
            resistance=0.0,
            peak=0.0,
            modes=32,
            mechanical_band=7000.0,
            pick=0.137 / scale if reference == "open" else 0.137,
            pickup=0.079 / scale,
            pickup_width=0.03 / scale,
            cutoff=6555.0,
            rounding="spatial",
            stiffness=4e-5,
            length=0.65 * scale,
            sustain=9.0,
            treble_decay=2.5,
        )
        y = Filter(cutoff=4000, q=0.7).process(y[None, :], RenderContext(sr, 60))[0]
        M, K, C, h = _mechanics(f, modes=8, peak=0.0, stiffness=4e-5, length=0.65 * scale)
        gamma = np.diag(C) / (2 * np.diag(M))
        hz = np.sqrt(np.diag(K) / np.diag(M) - gamma * gamma) / (2 * np.pi)
        from scipy.signal import resample_poly

        analysis_sr = 22050
        if sr != analysis_sr:
            y = resample_poly(y, analysis_sr, sr)
        profile = partial_envelopes(y, analysis_sr, f, partials=hz / f, minimum_window=0.12, cycles=3)
        items.append(ModalPluckObservation(pitch=pitch, string=1, f0=f, stiffness=4e-5, envelopes=profile))
    return items


@pytest.mark.parametrize("reference", ["open", "vibrating"])
def test_recovers_normalized_shape_of_independently_generated_pcm(reference):
    data = observations(reference)
    fit = fit_modal_pluck(data, pickup_width=0.03, max_partials=8, pick_reference=reference)
    assert fit.converged and fit.cost < 1
    assert abs(fit.pick_position - 0.137) < 0.002
    assert abs(fit.pickup_position - 0.079) < 0.002
    assert abs(fit.pick_cutoff / 6555.0 - 1) < 0.05
    assert fit.observations == 3 and fit.supported_cells > 100
    assert len(fit.trials) >= 3 and not hasattr(fit, "phase")
    control = fit.note_controls(71, 1)
    assert control["string"] == 1
    assert control["pick_position"] == pytest.approx(
        fit.pick_position * (fret_scale(71, 1) if reference == "vibrating" else 1)
    )


def test_short_budget_reports_unconverged_instead_of_claiming_success():
    fit = fit_modal_pluck(observations(), pickup_width=0.03, max_partials=8, max_nfev=1, polish_nfev=0)
    assert not fit.converged and all(not t.converged for t in fit.trials)
    assert fit.cost > 0


def test_auto_compares_reference_models_by_training_cost_only():
    fit = fit_modal_pluck(observations(), pickup_width=0.03, max_partials=8, pick_reference="auto")
    assert fit.pick_reference == "open"
    assert {t.pick_reference for t in fit.trials} == {"open", "vibrating"}


def test_renderer_rate_can_differ_from_resampled_analysis_rate():
    fit = fit_modal_pluck(observations(source_sr=44100), pickup_width=0.03, max_partials=8, model_sample_rate=44100)
    assert fit.model_sample_rates == (44100,)
    assert fit.cost < 1 and fit.converged
    assert abs(fit.pick_position - 0.137) < 0.002 and abs(fit.pickup_position - 0.079) < 0.002
    assert abs(fit.pick_cutoff / 6555.0 - 1) < 0.05


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pick_reference": "bad"},
        {"sustain": 0},
        {"treble_decay": np.nan},
        {"pickup_width": -0.1},
        {"electrical_q": 0},
        {"start": 0.7, "end": 0.3},
        {"max_partials": 2},
        {"max_partials": True},
        {"max_nfev": 0},
        {"polish_nfev": -1},
        {"pick_bounds": (0.3, 0.1)},
        {"pickup_bounds": (0.001, 0.12)},
        {"cutoff_bounds": (1, np.inf)},
        {"coarse_seeds": None},
        {"model_sample_rate": True},
        {"model_sample_rate": 2000},
    ],
)
def test_invalid_source_fit_options(kwargs):
    with pytest.raises(ValueError):
        fit_modal_pluck([], **kwargs)


def test_bad_profiles_and_support_rejected():
    data = observations()
    for item in [
        replace(data[0], string=0),
        replace(data[0], stiffness=-0.1),
        replace(data[0], stiffness=None),
        replace(data[0], f0=None),
        replace(data[0], pitch=None),
        replace(data[0], valid=np.ones_like(data[0].envelopes.amplitude)),
        replace(data[0], valid=np.zeros_like(data[0].envelopes.amplitude, dtype=bool)),
        replace(data[0], envelopes=replace(data[0].envelopes, amplitude=-data[0].envelopes.amplitude)),
        replace(data[0], partial_t60s=(0.00000001,)),
    ]:
        with pytest.raises(ValueError):
            fit_modal_pluck([item, data[1]])
    with pytest.raises(ValueError, match="different pitches"):
        fit_modal_pluck([data[0], data[0]])
    with pytest.raises(ValueError):
        fit_modal_pluck(data, end=2.0)
