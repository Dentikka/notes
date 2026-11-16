"""Amplitude truth, moving frequency, noise-only and unresolved-mode controls."""
import numpy as np
import pytest

from notes.ears import partial_envelopes


@pytest.mark.parametrize('f0', [66., 220., 770.])
@pytest.mark.parametrize('moving', [False, True])
def test_recovers_known_amplitudes_with_motion_am_noise_and_unrelated_phases(f0, moving):
    sr = 8000
    t = np.arange(sr*2)/sr
    frequency = f0*2**((30*np.sin(2*np.pi*5.3*t)+6*t)/1200) if moving else np.full_like(t, f0)
    phase = 2*np.pi*np.concatenate(([0.], np.cumsum(frequency[:-1]/sr)))
    count = min(8, int(sr*.44/max(frequency)))
    amplitudes = .7/np.arange(1, count+1)
    envelope = 1+.2*np.sin(2*np.pi*1.1*t)
    rng = np.random.default_rng(7)
    x = .3+sum(a*envelope*np.sin(h*phase+phi)
               for h, (a, phi) in enumerate(zip(amplitudes, rng.uniform(0, 6, count), strict=True), 1))
    x += .01*rng.normal(size=len(t))
    result = partial_envelopes(x, sr, frequency, partials=count)
    truth = np.interp(result.time, t, envelope)[:, None]*amplitudes
    error_db = 20*np.log10(result.amplitude/truth)
    assert np.max(abs(error_db)) < .25
    assert np.mean(result.above_white_noise_floor) == 1
    assert np.median(result.residual_rms) > .005
    assert np.max(result.design_condition) < 3
    assert not hasattr(result, 'phase')


def test_known_inharmonic_modes_do_not_get_forced_into_harmonic_bands():
    sr, f0 = 8000, 220.
    h = np.arange(1, 13)
    ratios = h*np.sqrt((1+.001*h*h)/1.001)
    t = np.arange(sr)/sr
    amplitudes = .6/h
    x = sum(a*np.sin(2*np.pi*f0*r*t+.23*i) for i, (a, r) in enumerate(zip(amplitudes, ratios, strict=True)))
    result = partial_envelopes(x, sr, f0, partials=ratios)
    np.testing.assert_allclose(result.amplitude, np.broadcast_to(amplitudes, result.amplitude.shape), atol=1e-12)
    assert max(result.residual_rms) < 1e-11


def test_eighty_moving_modes_are_resolved_at_the_reference_analysis_rate():
    sr, f0 = 22050, 66.
    time = np.arange(round(sr*.8))/sr
    frequency = f0*2**(20*np.sin(2*np.pi*5.3*time)/1200)
    phase = 2*np.pi*np.concatenate(([0.], np.cumsum(frequency[:-1]/sr)))
    amplitudes = .7/np.arange(1, 81)
    rng = np.random.default_rng(11)
    x = sum(a*np.sin(h*phase+phi) for h, (a, phi) in
            enumerate(zip(amplitudes, rng.uniform(0, 6, 80), strict=True), 1))
    x += .0005*rng.normal(size=len(time))
    result = partial_envelopes(x, sr, frequency, partials=80, step=.08)
    assert result.amplitude.shape[1] == 80
    assert np.max(abs(20*np.log10(result.amplitude/amplitudes))) < .08
    assert np.all(result.above_white_noise_floor)


def test_white_noise_dc_silence_and_missing_modes_do_not_become_supported_tones():
    sr = 8000
    noise = partial_envelopes(np.random.default_rng(9).normal(size=sr*2), sr, 220., partials=12)
    assert np.mean(noise.above_white_noise_floor) < .01
    for x in (np.zeros(sr), np.ones(sr)*.7):
        result = partial_envelopes(x, sr, 220., partials=8)
        assert not np.any(result.above_white_noise_floor)
    t = np.arange(sr)/sr
    result = partial_envelopes(np.sin(2*np.pi*440*t), sr, 220., partials=5)
    assert np.all(result.above_white_noise_floor[:, 1])
    assert not np.any(result.above_white_noise_floor[:, [0, 2, 3, 4]])


def test_wrong_frequency_prior_leaves_large_residual_instead_of_silently_retuning():
    sr = 8000
    t = np.arange(sr)/sr
    x = np.sin(2*np.pi*220*t)+.5*np.sin(2*np.pi*660*t)
    correct = partial_envelopes(x, sr, 220., partials=3)
    wrong = partial_envelopes(x, sr, 230., partials=3)
    assert max(correct.residual_rms) < 1e-10
    assert np.median(wrong.residual_rms) > .1


def test_unresolved_modes_raise_instead_of_exposing_unstable_amplitudes():
    with pytest.raises(ValueError, match='unresolved'):
        partial_envelopes(np.ones(8000), 8000, 220., partials=[1., 1.000000001])


@pytest.mark.parametrize('kwargs', [
    {'partials': 0}, {'partials': 81}, {'partials': True}, {'partials': []}, {'partials': [2., 1.]},
    {'partials': [1., 50.]}, {'partials': [1., np.nan]}, {'step': 0}, {'cycles': 2},
    {'minimum_window': .001}, {'fundamental_hz': np.ones(10)*220}, {'sr': 8000.},
    {'fundamental_hz': 10.}, {'fundamental_hz': np.nan},
])
def test_invalid_analysis_arguments(kwargs):
    options = {'sr': 8000, 'fundamental_hz': 220., **kwargs}
    with pytest.raises(ValueError):
        partial_envelopes(np.ones(8000), **options)


def test_analysis_clock_and_nyquist_truncation_are_explicit():
    result = partial_envelopes(np.ones(8000), 8000, 1000., partials=80)
    np.testing.assert_array_equal(result.ratios, [1, 2, 3])
    assert result.time[0] >= result.window_seconds/2-1/8000
    assert result.time[-1]+result.window_seconds/2 <= 1+1/8000
    with pytest.raises(ValueError, match='three full'):
        partial_envelopes(np.ones(500), 8000, 220.)
