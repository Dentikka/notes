import numpy as np
import pytest

from notes.ears.note_profile import compare_profiles, profile_note

SR = 22050


def tone(f=220, second=0.3, tau=1.0, silence=0.0):
    t = np.arange(SR*2)/SR
    y = (np.sin(2*np.pi*f*t)+second*np.sin(4*np.pi*f*t))*np.exp(-t/tau)*(1-np.exp(-t/.005))
    return np.pad(y, (round(silence*SR), 0))


def test_gain_and_leading_silence_are_nuisance_variables():
    a = profile_note(tone(), SR, 220)
    b = profile_note(tone(silence=.23)*.2, SR, 220)
    report = compare_profiles(a, b)
    assert abs(report["pitch_cents"]) < 1
    assert report["harmonic_mae_db"] < .3
    assert b.onset-a.onset == pytest.approx(.23, abs=.005)


def test_wrong_timbre_pitch_and_decay_are_visible():
    a = profile_note(tone(), SR, 220)
    bright = compare_profiles(a, profile_note(tone(second=1.8), SR, 220))
    detuned = compare_profiles(a, profile_note(tone(f=224), SR, 220))
    short = compare_profiles(a, profile_note(tone(tau=.3), SR, 220))
    assert bright["harmonic_mae_db"] > 4
    assert detuned["pitch_cents"] > 25
    assert short["envelope_mae_db"] > 4


def test_fast_decay_followed_by_exact_silence_has_finite_envelope():
    t = np.arange(SR*2)/SR
    x = np.sin(2*np.pi*220*t)*np.exp(-t*15)
    x[SR:] = 0
    with np.errstate(invalid="raise"):
        measured = profile_note(x, SR, 220)
    assert np.isfinite(measured.rms_db).all()
    report = compare_profiles(measured, measured)
    assert np.isfinite(list(report.values())).all()
    assert report["envelope_mae_db"] < 1e-9


@pytest.mark.parametrize("x", [np.zeros(SR), np.zeros((2, SR)), np.full(SR, np.nan)])
def test_invalid_note_rejected(x):
    with pytest.raises(ValueError):
        profile_note(x, SR, 220)


def test_cli_audits_known_notes_without_claiming_realism(tmp_path):
    import json

    from notes.cli import main
    from notes.engine.audio import write_wav
    ref, candidate = tmp_path/"reference.wav", tmp_path/"candidate.wav"
    write_wav(ref, tone()*0.3, SR)
    write_wav(candidate, tone(second=1.8)*0.2, SR)
    output = tmp_path/"audit/report.json"
    assert main(["audit-note", str(ref), str(candidate), "--pitch", "A3", "-o", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "diagnostic-only"
    assert report["harmonic_mae_db"] > 4
    with pytest.raises(ValueError):
        main(["audit-note", str(ref), str(candidate), "--pitch", "A3", "-o", str(ref)])


def test_spurious_render_partial_is_not_hidden_by_reference_mask():
    t = np.arange(SR*2)/SR
    reference = tone()
    extra = .35*np.sin(2*np.pi*220*15*t)*np.exp(-t)*(1-np.exp(-t/.005))
    report = compare_profiles(profile_note(reference, SR, 220),
                              profile_note(reference+extra, SR, 220))
    assert report["harmonic_mae_db"] > 3


def test_out_of_harmonic_range_noise_is_visible_in_broadband_error():
    t = np.arange(SR*2)/SR
    extra = .3*np.sin(2*np.pi*7000*t)*np.exp(-t)
    a = profile_note(tone(), SR, 220)
    b = profile_note(tone()+extra, SR, 220)
    assert compare_profiles(a, b)["broadband_mae_db"] > 3


def test_set_audit_catches_regression_hidden_by_better_average():
    from notes.ears.note_profile import audit_profile_set
    ref = profile_note(tone(), SR, 220)
    bad = profile_note(tone(second=1.8), SR, 220)
    smaller_error = profile_note(tone(second=.5), SR, 220)
    references = {"improved":ref,"regressed":ref}
    old = {"improved":bad,"regressed":ref}
    candidate = {"improved":ref,"regressed":smaller_error}
    report = audit_profile_set(references, old, candidate)
    assert report["candidate_mean_loss"] < report["baseline_mean_loss"]
    assert report["no_regression"] is False
    assert report["regressions"] == ["regressed"]
    assert audit_profile_set(references, old, old)["no_regression"] is True
    with pytest.raises(ValueError):
        audit_profile_set(references, old, {"improved":ref})
    with pytest.raises(ValueError):
        audit_profile_set(references, old, old, regression_tolerance=float("nan"))


def test_partial_decays_recover_independent_losses_and_ignore_gain_and_silence():
    from notes.ears.note_profile import partial_decays

    time = np.arange(SR*2)/SR
    wave = (np.sin(2*np.pi*220*time)*np.exp(-time/.8)
            + .5*np.sin(2*np.pi*440*time)*np.exp(-time/.3))
    expected = np.log(1000)*np.array([.8, .3])
    for audio in (wave, np.pad(wave*.1, (SR//4, 0))):
        decay = partial_decays(profile_note(audio, SR, 220), start=.2, end=.6)
        np.testing.assert_allclose([p.t60_seconds for p in decay[:2]], expected, rtol=.02)
        assert all(p.reliable for p in decay[:2])
        assert not decay[-1].reliable


def test_decay_windows_expose_modulation_growth_and_changing_loss():
    from notes.ears.note_profile import partial_decays

    time = np.arange(SR*2)/SR
    carrier = np.sin(2*np.pi*220*time)
    # Growth and audible beating must not be reported as reliable exponential decay.
    for envelope in (np.exp(time), np.exp(-time)*(1+.8*np.sin(2*np.pi*5*time))):
        measured = partial_decays(profile_note(carrier*envelope, SR, 220), start=.2, end=1.2)[0]
        assert not measured.reliable and measured.t60_seconds is None
    envelope = np.exp(-np.minimum(time,.8)/.8-np.maximum(time-.8,0)/.2)
    profile = profile_note(carrier*envelope, SR, 220)
    early = partial_decays(profile, start=.2, end=.6)[0]
    late = partial_decays(profile, start=1., end=1.4)[0]
    assert early.reliable and late.reliable
    assert late.decay_db_per_second/early.decay_db_per_second == pytest.approx(4., rel=.02)


@pytest.mark.parametrize("options", [{"start":-.1}, {"end":3.}, {"end":.2}, {"end":.21},
                                     {"audibility_db":float("nan")}, {"residual_limit_db":0}])
def test_invalid_decay_window_and_thresholds_are_rejected(options):
    from notes.ears.note_profile import partial_decays

    with pytest.raises(ValueError):
        partial_decays(profile_note(tone(), SR, 220), **options)


def test_cli_reports_multiple_decay_windows_and_preserves_old_metrics(tmp_path):
    import json

    from notes.cli import main
    from notes.engine.audio import write_wav

    reference, candidate = tmp_path/"ref.wav", tmp_path/"candidate.wav"
    write_wav(reference, tone(tau=.8)*.2, SR)
    write_wav(candidate, tone(tau=.3)*.2, SR)
    output = tmp_path/"decays.json"
    assert main(["audit-note", str(reference), str(candidate), "--pitch", "A3", "-o", str(output),
                 "--decay-window", ".2", ".6", "--decay-window", "1", "1.4"]) == 0
    report = json.loads(output.read_text())
    assert report["profile_metric_version"] == 2 and report["status"] == "diagnostic-only"
    assert len(report["decay_windows"]) == 2
    for window in report["decay_windows"]:
        assert window["reference"][0]["t60_seconds"] > window["render"][0]["t60_seconds"]
