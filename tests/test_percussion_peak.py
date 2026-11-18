import numpy as np
import pytest

from notes.ears import band_peak_trajectory, compare_peak_trajectories


def hit(sr, frequency=100.):
    t = np.arange(sr)/sr
    return np.exp(-t/.3)*(np.sin(2*np.pi*frequency*t)+.1*np.sin(2*np.pi*170*t))


def test_damped_peak_frequency_and_gain_invariance():
    a = band_peak_trajectory(hit(8000), 8000)
    b = band_peak_trajectory(hit(8000)*.003, 8000)
    assert np.all(a.supported) and np.max(abs(a.frequency_hz-100)) < .5
    np.testing.assert_allclose(a.frequency_hz, b.frequency_hz, atol=1e-8)
    np.testing.assert_allclose(a.contrast_db, b.contrast_db, atol=1e-8)
    np.testing.assert_allclose(a.relative_rms_db, b.relative_rms_db, atol=1e-8)


@pytest.mark.parametrize('gain', [1e-200, 1e200])
def test_extreme_finite_gain_does_not_overflow_or_erase_relative_peaks(gain):
    a = band_peak_trajectory(hit(8000)*gain, 8000)
    assert np.all(a.supported) and np.max(abs(a.frequency_hz-100)) < .5


def test_moving_peak_is_compared_on_its_own_clock_not_a_constant_pitch():
    sr = 8000
    t = np.arange(sr)/sr
    x = np.exp(-t/.4)*np.sin(2*np.pi*(90*t+60*.1*(1-np.exp(-t/.1))))
    a = band_peak_trajectory(x, sr)
    assert a.frequency_hz[0] > a.frequency_hz[-1]*1.1
    assert np.all(a.supported) and compare_peak_trajectories(a, a)['passed']


def test_wrong_dominant_mode_rejected_even_with_equal_total_energy():
    a, b = hit(8000), hit(8000, 170.)
    b *= np.linalg.norm(a)/np.linalg.norm(b)
    report = compare_peak_trajectories(band_peak_trajectory(a, 8000), band_peak_trajectory(b, 8000))
    assert not report['passed'] and report['maximum_relative_error'] > .6


def test_silence_and_missing_target_support_do_not_pass():
    a = band_peak_trajectory(hit(8000), 8000)
    silent = band_peak_trajectory(np.zeros(8000), 8000)
    assert not np.any(silent.supported)
    report = compare_peak_trajectories(a, silent)
    assert not report['passed'] and report['missing_candidate_windows'] == 3
    assert not compare_peak_trajectories(silent, silent)['passed']


def test_sample_rate_change_preserves_physical_peak_windows():
    a, b = [band_peak_trajectory(hit(sr), sr) for sr in (8000, 22050)]
    assert compare_peak_trajectories(a, b, maximum_relative_error=.001)['passed']
    with pytest.raises(ValueError, match='clock'):
        compare_peak_trajectories(a, band_peak_trajectory(hit(8000), 8000, windows=((.1, .2), (.2, .4))))


@pytest.mark.parametrize('kwargs', [{'band': (0, 100)}, {'band': (100, 5000)},
    {'windows': ((.1, .2), (.05, .2))}, {'windows': ((.1, .1001),)}, {'windows': ((.5, 1.1),)},
    {'minimum_contrast_db': -1}, {'minimum_level_db': 2}])
def test_invalid_analysis_configuration(kwargs):
    with pytest.raises(ValueError):
        band_peak_trajectory(hit(8000), 8000, **kwargs)


@pytest.mark.parametrize('x,sr', [(np.zeros((2, 8000)), 8000), (np.full(8000, np.nan), 8000),
                                (np.zeros(8000), 0), (np.zeros(8000), 8000.5)])
def test_invalid_audio_and_rate(x, sr):
    with pytest.raises(ValueError):
        band_peak_trajectory(x, sr)
