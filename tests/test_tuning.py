import numpy as np
import pytest

from notes.ears import known_note_tuning


@pytest.mark.parametrize("sr", [22050, 44100])
@pytest.mark.parametrize("frequency", [82.4069, 164.8138, 246.9417])
def test_known_frequency_survives_nonlinear_distortion(sr, frequency):
    time = np.arange(2*sr)/sr
    source = sum(np.sin(2*np.pi*frequency*h*time+.17*h)/h for h in range(1, 12))*np.exp(-time/.8)
    for wave in (source, np.tanh(12*source+.3)-np.tanh(.3)):
        estimate = known_note_tuning(wave, sr, frequency)
        assert estimate.reliable
        assert abs(1200*np.log2(estimate.f0/frequency)) < .3


def test_pure_sine_and_gain_invariance():
    sr = 22050
    source = np.sin(2*np.pi*440*np.arange(sr)/sr)
    a = known_note_tuning(source, sr, 440)
    b = known_note_tuning(source*.25, sr, 440)
    assert a.reliable and b.reliable
    assert a.f0 == pytest.approx(440, abs=.01)
    assert a.f0 == pytest.approx(b.f0, abs=1e-8)


@pytest.mark.parametrize("seed", [1, 17, 31])
def test_noise_does_not_become_a_reliable_prior(seed):
    source = np.random.default_rng(seed).normal(size=44100)
    assert not known_note_tuning(source, 22050, 110).reliable


def test_wrong_prior_and_invalid_inputs():
    sr = 22050
    source = np.sin(2*np.pi*440*np.arange(sr)/sr)
    assert not known_note_tuning(source, sr, 415.3).reliable
    for wave in (np.zeros(sr), np.ones(sr), np.full(sr, np.nan)):
        with pytest.raises(ValueError):
            known_note_tuning(wave, sr, 440)
    with pytest.raises(ValueError, match="window"):
        known_note_tuning(source[:1000], sr, 440)
