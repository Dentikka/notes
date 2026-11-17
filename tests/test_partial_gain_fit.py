"""Shape recovery with differing frequencies and unrelated phases, plus support guards."""
from dataclasses import replace

import numpy as np
import pytest

from notes.ears import fit_partial_gains, partial_envelopes


def envelopes(amplitudes, *, f0=220., stiffness=.001, phase=.31):
    sr = 8000
    t = np.arange(round(sr*1.6))/sr
    h = np.arange(1, len(amplitudes)+1)
    ratios = h*np.sqrt((1+stiffness*h*h)/(1+stiffness))
    x = sum(a*np.sin(2*np.pi*f0*r*t+phase*k) for k, (a, r) in enumerate(zip(amplitudes, ratios, strict=True), 1))
    return partial_envelopes(x, sr, f0, partials=ratios)


def test_recovers_shape_with_distinct_source_and_reference_frequencies_and_phases():
    target = np.array([.3, .15, .2, .09])
    source = np.array([.4, .2, .1, .06])
    a = envelopes(target)
    b = envelopes(source, f0=180., stiffness=.0001, phase=1.19)
    assert not np.array_equal(a.ratios, b.ratios)
    fit = fit_partial_gains(a, b)
    expected = (target/np.linalg.norm(target))/(source/np.linalg.norm(source))
    np.testing.assert_allclose(fit.gains, expected, atol=2e-12)
    reconstructed = source*np.array(fit.gains)
    np.testing.assert_allclose(reconstructed/np.linalg.norm(reconstructed), target/np.linalg.norm(target), atol=2e-12)
    assert fit.upper_clipped_cells == 0 and fit.supported_cells == fit.total_cells
    assert isinstance(fit.gains, tuple) and not hasattr(fit, 'phase')


def test_loudness_is_not_fitted_and_gain_limit_is_reported():
    a = envelopes([.3, .2, .1, .5])
    b = envelopes([.3, .2, .1, 1e-6])
    fit = fit_partial_gains(a, b)
    assert fit.gains[-1] == 16 and fit.upper_clipped_cells > 0
    identity = fit_partial_gains(a, envelopes([3., 2., 1., 5.]))
    np.testing.assert_allclose(identity.gains, np.ones(4), atol=2e-12)


def test_unsupported_reference_mode_does_not_copy_noise_projection():
    a = envelopes([.3, .2, .1, .05])
    amplitude = a.amplitude.copy()
    amplitude[:, -1] = .5
    floor = np.full_like(amplitude, .001)
    supported = np.ones_like(amplitude, dtype=bool)
    supported[:, -1] = False
    noisy = replace(a, amplitude=amplitude, white_noise_floor=floor, above_white_noise_floor=supported)
    fit = fit_partial_gains(noisy, a)
    assert fit.gains[-1] < .1 and fit.supported_cells == fit.used_frames*3


def test_noise_only_and_silent_candidate_are_rejected():
    a = envelopes([.3, .2, .1])
    noise = partial_envelopes(np.random.default_rng(7).normal(size=12800), 8000, 220., partials=3)
    with pytest.raises(ValueError, match='supported frames'):
        fit_partial_gains(noise, a)
    with pytest.raises(ValueError, match='supported frames'):
        fit_partial_gains(a, replace(a, amplitude=np.zeros_like(a.amplitude)))


@pytest.mark.parametrize('window', [{'start': -1.}, {'start': .4, 'end': .2}, {'end': 10.}, {'end': np.nan}])
def test_invalid_windows_are_rejected(window):
    a = envelopes([.3, .2, .1])
    with pytest.raises(ValueError, match='window'):
        fit_partial_gains(a, a, **window)


def test_misaligned_frames_missing_modes_and_nonfinite_data_are_rejected():
    a = envelopes([.3, .2, .1])
    with pytest.raises(ValueError, match='frame clocks'):
        fit_partial_gains(a, replace(a, time=a.time+.001))
    with pytest.raises(ValueError, match='corresponding modes'):
        fit_partial_gains(a, envelopes([.3, .2]))
    bad = a.amplitude.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        fit_partial_gains(a, replace(a, amplitude=bad))
