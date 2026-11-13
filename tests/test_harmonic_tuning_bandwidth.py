"""The allowed cent offset must not move upper partials out of their filters."""
import numpy as np
import pytest

from notes.engine.tuning import known_note_tuning


@pytest.mark.parametrize('sr', [22050, 44100])
@pytest.mark.parametrize('expected', [220., 659.255, 1046.502, 2093.005])
@pytest.mark.parametrize('cents', [-40., 40.])
def test_detuned_noisy_partial_series_is_measured_without_octave_or_filter_bias(sr, expected, cents):
    t = np.arange(3*sr)/sr
    frequency = expected*2**(cents/1200)
    source = sum(np.sin(2*np.pi*h*frequency*t+.19*h)/h for h in range(1, 4))
    source += .003*np.random.default_rng(17).normal(size=len(t))
    result = known_note_tuning(source, sr, expected, start=1.2, end=2.4, harmonic_bandwidth=True)
    assert result.reliable
    assert abs(1200*np.log2(result.f0/frequency)) < .05
    assert result.harmonic_spread_cents < .1


def test_narrow_filter_rejects_a_valid_high_detuned_noisy_note():
    sr, expected = 22050, 2093.005
    t = np.arange(3*sr)/sr
    frequency = expected*2**(40/1200)
    source = sum(np.sin(2*np.pi*h*frequency*t+.19*h)/h for h in range(1, 4))
    source += .003*np.random.default_rng(17).normal(size=len(t))
    assert not known_note_tuning(source, sr, expected, start=1.2, end=2.4).reliable
    assert known_note_tuning(source, sr, expected, start=1.2, end=2.4, harmonic_bandwidth=True).reliable


@pytest.mark.parametrize('expected', [55., 110., 440., 1046.5])
def test_noise_and_missing_fundamentals_remain_unreliable(expected):
    sr = 22050
    t = np.arange(3*sr)/sr
    noise = np.random.default_rng(17).normal(size=len(t))
    assert not known_note_tuning(noise, sr, expected, start=1.2, end=2.4, harmonic_bandwidth=True).reliable
    source = .5*np.sin(2*np.pi*2*expected*t)+.3*np.sin(2*np.pi*3*expected*t)+noise*.003
    assert not known_note_tuning(source, sr, expected, start=1.2, end=2.4, harmonic_bandwidth=True).reliable


def test_prior_limit_and_moving_pitch_are_still_rejected():
    sr, expected = 22050, 659.255
    t = np.arange(3*sr)/sr
    wrong = np.sin(2*np.pi*expected*2**(75/1200)*t)
    assert not known_note_tuning(wrong, sr, expected, start=1.2, end=2.4, harmonic_bandwidth=True).reliable
    phase = np.cumsum(expected*2**(.8*np.sin(2*np.pi*5*t)/12)/sr)
    moving = np.sin(2*np.pi*phase)
    assert not known_note_tuning(moving, sr, expected, start=1.2, end=2.4, harmonic_bandwidth=True).reliable


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_bandwidth_mode_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match='harmonic_bandwidth'):
        known_note_tuning(np.ones(22050), 22050, 440., harmonic_bandwidth=value)


def test_default_and_explicit_legacy_mode_are_identical():
    sr, expected = 22050, 659.255
    t = np.arange(3*sr)/sr
    source = np.sin(2*np.pi*expected*2**(20/1200)*t)+.05*np.random.default_rng(31).normal(size=len(t))
    default = known_note_tuning(source, sr, expected, start=1.2, end=2.4)
    legacy = known_note_tuning(source, sr, expected, start=1.2, end=2.4, harmonic_bandwidth=False)
    assert default == legacy
