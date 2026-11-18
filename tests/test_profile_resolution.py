"""Frequency/time resolution and broad modal-transition diagnostics."""

import json

import numpy as np
import pytest
from notes.ears.note_profile import audit_profile_set, compare_profiles, harmonic_shape_distance, profile_note
from notes.ears.percussion import band_peak_trajectory

SR = 22050


def hybrid():
    t = np.arange(2 * SR) / SR
    return (
        6 * np.sin(2 * np.pi * 490.9 * t + 0.2) * np.exp(-20 * t)
        + 4 * np.sin(2 * np.pi * 722 * t + 0.4) * np.exp(-28 * t)
        + np.sin(2 * np.pi * 603.2 * t) * np.exp(-1.5 * t)
        + 0.2 * np.sin(2 * np.pi * 1206.4 * t) * np.exp(-2 * t)
    ) * (1 - np.exp(-t / 0.002))


@pytest.mark.parametrize("mode,version", [("fixed", 4), ("tracked", 5)])
@pytest.mark.parametrize("window", [0.08, 0.12, 0.24])
def test_resolved_window_finds_actual_in_prior_component_instead_of_a_smeared_peak(mode, version, window):
    y = hybrid()
    legacy = profile_note(y, SR, 587.5, harmonics=4, partial_mode=mode)
    resolved = profile_note(y, SR, 587.5, harmonics=4, partial_mode=mode, analysis_window=window)
    # Known constructed modal frequencies: the old broad main lobes can place
    # a peak where the signal has no oscillator at all.
    assert min(abs(legacy.f0 - f) for f in [490.9, 603.2, 722.0]) > 20
    assert resolved.f0 == pytest.approx(603.2, abs=0.05)
    assert resolved.onset == legacy.onset and resolved.attack_seconds == legacy.attack_seconds
    assert resolved.analysis_window == window
    assert compare_profiles(resolved, resolved)["profile_metric_version"] == version


def test_broad_peaks_reveal_the_early_low_mode_and_late_known_note_component():
    p = band_peak_trajectory(hybrid(), SR, band=(293.75, 881.25), windows=((0.02, 0.15), (0.4, 1.0)))
    assert p.supported.all()
    np.testing.assert_allclose(p.frequency_hz, [490.9, 603.2], atol=0.05)


def test_missing_in_prior_fundamental_returns_no_pitch_comparison_instead_of_a_false_zero():
    t = np.arange(2 * SR) / SR
    a = profile_note(np.sin(2 * np.pi * 440 * t), SR, 440., analysis_window=.12)
    b = profile_note(np.sin(2 * np.pi * 880 * t), SR, 440., analysis_window=.12)
    assert a.frequency_supported is True and b.frequency_supported is False
    assert b.f0 == 440.  # Only the supplied anchor remains available for bands.
    report = compare_profiles(a, b)
    assert report['pitch_supported'] is False and report['pitch_cents'] is None
    audit = audit_profile_set({'x': b}, {'x': b}, {'x': b})
    assert audit['regressions'] == [] and audit['no_regression'] is False
    assert audit['frequency_unverified_cases'] == ['x']


def test_explicit_window_uses_supported_local_peaks_not_an_unresolved_time_average():
    with pytest.raises(ValueError, match='analysis_window'):
        profile_note(hybrid(), SR, 587.5, analysis_window=False)
    p = profile_note(hybrid(), SR, 587.5, analysis_window=.12, partial_mode='fixed')
    assert p.frequency_supported is True and p.pitch_tracking_valid.any()
    use = (p.time >= .25) & (p.time <= .4) & p.pitch_tracking_valid
    assert np.median(p.fundamental_hz[use]) == pytest.approx(603.2, abs=.1)


def test_new_resolution_preserves_gain_and_onset_nuisance_handling():
    # Keep the onset away from the file boundary: the symmetric RMS detector
    # otherwise cannot recover its normal pre-onset margin from a cropped clip.
    y = np.pad(hybrid(), (SR // 10, 0))
    a = profile_note(y, SR, 587.5, analysis_window=0.12)
    b = profile_note(np.pad(y * 0.1, (round(SR * 0.23), 0)), SR, 587.5, analysis_window=0.12)
    result = compare_profiles(a, b)
    assert abs(result["pitch_cents"]) < 0.01 and result["harmonic_mae_db"] < 0.05
    assert b.onset - a.onset == pytest.approx(0.23, abs=0.005)


def test_incompatible_resolution_settings_cannot_mix_in_comparisons():
    y = hybrid()
    a = profile_note(y, SR, 587.5)
    b = profile_note(y, SR, 587.5, analysis_window=0.12)
    c = profile_note(y, SR, 587.5, analysis_window=0.24)
    for other in [a, c]:
        for function in [compare_profiles, harmonic_shape_distance]:
            with pytest.raises(ValueError, match="same measurement mode and analysis window"):
                function(b, other)
        with pytest.raises(ValueError, match="same measurement mode and analysis window"):
            audit_profile_set({"x": b}, {"x": b}, {"x": other})
    assert audit_profile_set({"x": b}, {"x": b}, {"x": b})["profile_metric_version"] == 4


@pytest.mark.parametrize("window", [0.0, -0.1, 0.001, 0.51, np.nan, np.inf, True, "bad"])
def test_bad_explicit_windows_are_rejected(window):
    with pytest.raises(ValueError, match="analysis_window"):
        profile_note(hybrid(), SR, 587.5, analysis_window=window)


def test_window_requires_enough_nominal_cycles_and_actual_post_onset_samples():
    with pytest.raises(ValueError, match="three nominal cycles"):
        profile_note(hybrid(), SR, 27.5, analysis_window=0.05)
    t = np.arange(SR // 10) / SR
    y = np.sin(2 * np.pi * 600 * t)
    assert profile_note(y, SR, 600).analysis_window is None
    with pytest.raises(ValueError, match="post-onset"):
        profile_note(y, SR, 600, analysis_window=0.12)


@pytest.mark.parametrize("f0", [110.0, 880.0])
def test_longer_tracked_windows_keep_relative_energies_under_vibrato(f0):
    t = np.arange(2 * SR) / SR
    phase = 2 * np.pi * np.cumsum(f0 * 2 ** (20 * np.sin(2 * np.pi * 5 * t) / 1200)) / SR
    amps = 1 / np.arange(1, 6)
    y = sum(a * np.sin(h * phase) for h, a in enumerate(amps, 1))
    p = profile_note(y, SR, f0, harmonics=5, partial_mode="tracked", analysis_window=0.12)
    use = (p.time >= 0.3) & (p.time <= 1.6)
    levels = p.partial_db[use]
    levels -= 10 * np.log10(np.sum(10 ** (levels / 10), axis=1, keepdims=True))
    target = 20 * np.log10(amps) - 10 * np.log10(np.sum(amps * amps))
    assert abs(levels - target).mean() < 0.3


def test_cli_explicit_resolution_and_broad_windows_are_reported_with_their_scope(tmp_path):
    from notes.cli import main
    from notes.engine.audio import write_wav

    t = np.arange(2 * SR) / SR
    reference = tmp_path / "reference.wav"
    candidate = tmp_path / "candidate.wav"
    output = tmp_path / "audit.json"
    write_wav(reference, 0.02 * np.sin(2 * np.pi * 587.5 * t) * np.exp(-t), SR)
    write_wav(candidate, 0.02 * hybrid(), SR)
    assert (
        main(
            [
                "audit-note",
                str(reference),
                str(candidate),
                "--pitch",
                "D5",
                "--partial-mode",
                "tracked",
                "--analysis-window",
                ".12",
                "--peak-window",
                ".02",
                ".15",
                "--peak-window",
                ".4",
                "1.",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["profile_metric_version"] == 5 and report["analysis_window"] == 0.12
    assert 40 < report["pitch_cents"] < 50
    assert "not fundamental" in report["broad_peaks"]["scope"]
    np.testing.assert_allclose(report["broad_peaks"]["render"]["frequency_hz"], [490.9, 603.2], atol=0.1)
