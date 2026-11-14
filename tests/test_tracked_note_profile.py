"""Known partial amplitudes under FM, plus noise/missing/extra-partial controls."""
import json

import numpy as np
import pytest

from notes.ears.note_profile import audit_profile_set, compare_profiles, harmonic_shape_distance, profile_note

SR = 22050


def controlled(f0=196., depth=0., *, drift=False, amplitudes=None, noise=0.):
    t = np.arange(3*SR)/SR
    cents = depth*t/3 if drift else depth*np.sin(2*np.pi*5*t)
    phase = 2*np.pi*np.cumsum(f0*2**(cents/1200))/SR
    amplitudes = (1/np.arange(1, min(20, int(SR*.45/f0))+1) if amplitudes is None
                  else np.asarray(amplitudes))
    y = sum(a*np.sin(h*phase+.13*h) for h, a in enumerate(amplitudes, 1))
    y += noise*np.random.default_rng(17).standard_normal(len(t))
    return y*np.minimum(1, t/.03), amplitudes


@pytest.mark.parametrize('f0', [65.4064, 110., 196., 440., 880., 1320.])
@pytest.mark.parametrize('depth,drift', [(0., False), (20., False), (40., False), (20., True)])
def test_moving_bands_recover_known_relative_partial_energies(f0, depth, drift):
    y, amplitudes = controlled(f0, depth, drift=drift)
    p = profile_note(y, SR, f0, harmonics=len(amplitudes), partial_mode='tracked')
    target = 20*np.log10(amplitudes)-10*np.log10(np.sum(amplitudes**2))
    keep = (p.time >= .4) & (p.time <= 1.8)
    levels = p.partial_db[keep]
    levels = levels-10*np.log10(np.sum(10**(levels/10), axis=1, keepdims=True))
    assert abs(levels-target).mean() < .15
    assert abs(levels-target).max() < 1.
    assert p.pitch_tracking_valid[keep].mean() > .95
    assert p.fundamental_hz.shape == p.pitch_tracking_valid.shape == p.time.shape


def test_fixed_mode_bias_is_reproduced_and_new_mode_does_not_reclassify_pitch():
    y, _ = controlled(440., 20.)
    steady, _ = controlled(440., 0.)
    old, old_steady = profile_note(y, SR, 440.), profile_note(steady, SR, 440.)
    assert harmonic_shape_distance(old_steady, old, start=.4, end=1.8) > 2.
    new, new_steady = (profile_note(x, SR, 440., partial_mode='tracked') for x in (y, steady))
    assert harmonic_shape_distance(new_steady, new, start=.4, end=1.8) < .05
    assert new.f0 == old.f0
    assert compare_profiles(new, new)['profile_metric_version'] == 3
    assert compare_profiles(old, old)['profile_metric_version'] == 2


def test_added_missing_partials_and_broadband_noise_remain_visible():
    amplitudes = np.zeros(20)
    amplitudes[:2] = [1., .3]
    ref, _ = controlled(220., 20., amplitudes=amplitudes)
    extra_amplitudes = amplitudes.copy()
    extra_amplitudes[14] = .35
    extra, _ = controlled(220., 20., amplitudes=extra_amplitudes)
    noisy, _ = controlled(220., 20., amplitudes=amplitudes, noise=.15)
    a, b, c = (profile_note(x, SR, 220., partial_mode='tracked') for x in (ref, extra, noisy))
    assert compare_profiles(a, b)['harmonic_mae_db'] > 3.
    assert compare_profiles(b, a)['harmonic_mae_db'] > 3.
    assert compare_profiles(a, b)['spurious_partial_cells'] > 0
    assert compare_profiles(b, a)['missing_partial_cells'] > 0
    assert compare_profiles(a, c)['broadband_mae_db'] > 3.


def test_gain_leading_silence_and_true_pitch_difference_remain_separate():
    y, _ = controlled(220., 20.)
    a = profile_note(y, SR, 220., partial_mode='tracked')
    b = profile_note(np.pad(y*.1, (SR//4, 0)), SR, 220., partial_mode='tracked')
    assert b.onset-a.onset == pytest.approx(.25, abs=.005)
    assert harmonic_shape_distance(a, b, start=.4, end=1.8) < .1
    other, _ = controlled(224.)
    detuned = profile_note(other, SR, 220., partial_mode='tracked')
    assert compare_profiles(a, detuned)['pitch_cents'] > 25.


def test_unsupported_fundamental_and_silent_tail_are_flagged():
    y, _ = controlled(220., amplitudes=[0., 1., .3])
    p = profile_note(y, SR, 220., harmonics=3, partial_mode='tracked')
    keep = (p.time >= .4) & (p.time <= 1.8)
    assert not p.pitch_tracking_valid[keep].any()
    y, _ = controlled(220., 20.)
    y[SR:] = 0.
    p = profile_note(y, SR, 220., partial_mode='tracked')
    assert not p.pitch_tracking_valid[p.time > 1.2].any()
    assert np.isfinite(p.partial_db).all() and np.isfinite(p.fundamental_hz).all()


def test_tracker_does_not_accept_a_peak_clipped_by_the_known_note_search_boundary():
    y, _ = controlled(440.*2**(2/12))
    p = profile_note(y, SR, 440., partial_mode='tracked')
    keep = (p.time >= .4) & (p.time <= 1.8)
    assert not p.pitch_tracking_valid[keep].any()


def test_modes_cannot_be_silently_mixed_in_comparison_or_set_audit():
    y, _ = controlled()
    a, b = (profile_note(y, SR, 196., partial_mode=mode) for mode in ('fixed', 'tracked'))
    for function in (compare_profiles, harmonic_shape_distance):
        with pytest.raises(ValueError, match='same measurement mode'):
            function(a, b)
    with pytest.raises(ValueError, match='same measurement mode'):
        audit_profile_set({'a': a}, {'a': b}, {'a': b})
    assert audit_profile_set({'a': b}, {'a': b}, {'a': b})['profile_metric_version'] == 3
    for mode in ('bad', None, True):
        with pytest.raises(ValueError):
            profile_note(y, SR, 196., partial_mode=mode)


def test_cli_records_tracked_measurement_mode_and_version(tmp_path):
    from notes.cli import main
    from notes.engine.audio import write_wav
    y, _ = controlled(220., 20.)
    path = tmp_path/'input.wav'
    write_wav(path, y*.1, SR)
    output = tmp_path/'report.json'
    assert main(['audit-note', str(path), str(path), '--pitch', 'A3', '--partial-mode', 'tracked',
                 '-o', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['partial_mode'] == 'tracked' and report['profile_metric_version'] == 3
    assert report['harmonic_mae_db'] == 0.
