"""Independent modal motion, frequency assumptions and ill-conditioned bases."""
import numpy as np
import pytest
from notes.ears import fit_partial_gains, partial_envelopes


def signal(tracks, amplitudes, sr, phases=None):
    phase = 2*np.pi*np.vstack((np.zeros((1, tracks.shape[1])), np.cumsum(tracks[:-1]/sr, axis=0)))
    phases = np.arange(tracks.shape[1])*.57 if phases is None else phases
    return np.sum(amplitudes*np.sin(phase+phases), axis=1)


def test_independent_chirps_recover_amplitudes_instead_of_false_decay():
    sr = 8000
    t = np.arange(sr*2)/sr
    h = np.arange(1, 7)
    tracks = 110*h+(35*np.exp(-t[:, None]/.15)+12*np.sin(2*np.pi*.6*t[:, None]))*(h/6)**2
    amplitudes = .7/h
    envelope = 1+.1*np.sin(2*np.pi*1.1*t)
    x = signal(tracks, amplitudes*envelope[:, None], sr)
    x += .001*np.random.default_rng(17).normal(size=len(t))
    actual = partial_envelopes(x, sr, tracks[:, 0], partials=6, partial_hz=tracks)
    shared = partial_envelopes(x, sr, tracks[:, 0], partials=6)
    truth = amplitudes*np.interp(actual.time, t, envelope)[:, None]
    assert np.max(abs(20*np.log10(actual.amplitude/truth))) < .1
    assert np.max(abs(20*np.log10(shared.amplitude/truth))) > .5
    centers = np.round(actual.time*sr).astype(int)
    np.testing.assert_array_equal(actual.partial_hz, tracks[centers])
    assert np.all(actual.above_white_noise_floor)
    assert shared.partial_hz is None
    assert not hasattr(actual, 'phase')
    assert not np.shares_memory(actual.partial_hz, tracks)


def test_explicit_tracks_define_band_and_count_independently_of_nominal_labels():
    sr = 8000
    tracks = np.tile([230., 510., 800., 1150., 1400.], (sr, 1))
    x = signal(tracks, np.array([.5, .3, .2, .1, .07]), sr)
    # Five nominal harmonics of 1000 Hz would ordinarily be truncated to three.
    actual = partial_envelopes(x, sr, 1000., partials=5, partial_hz=tracks)
    assert actual.amplitude.shape[1] == 5
    np.testing.assert_allclose(actual.amplitude,
                               np.broadcast_to([.5, .3, .2, .1, .07], actual.amplitude.shape), atol=1e-12)


def test_static_shape_fit_accepts_corresponding_independently_moving_modes():
    sr = 8000
    t = np.arange(sr*2)/sr
    tracks = np.column_stack((220+2*np.sin(t*5), 443+7*np.cos(t*3), 672+4*np.sin(t*7)))
    other = tracks+np.array([7., 12., 9.])
    a = partial_envelopes(signal(tracks, [.5, .3, .2], sr), sr, 220., partials=3, partial_hz=tracks)
    b = partial_envelopes(signal(other, [.2, .4, .3], sr, phases=[1., 2., 3.]),
                         sr, 220., partials=3, partial_hz=other)
    result = fit_partial_gains(a, b)
    expected = np.array([.5, .3, .2])/np.linalg.norm([.5, .3, .2])
    expected /= np.array([.2, .4, .3])/np.linalg.norm([.2, .4, .3])
    np.testing.assert_allclose(result.gains, expected, atol=1e-10)


@pytest.mark.parametrize('tracks', [
    np.ones((8000, 2))*220, np.ones((8000, 3))*220, np.ones((2, 8000)),
    np.ones(8000)*220, np.zeros((8000, 2)), np.ones((8000, 2))*np.nan,
    np.tile([220., 3600.], (8000, 1)), np.tile([440., 220.], (8000, 1)),
    np.tile([1e-300, 220.], (8000, 1)),
])
def test_invalid_tracks_fail_before_regression(tracks):
    with pytest.raises(ValueError):
        partial_envelopes(np.ones(8000), 8000, 220., partials=2, partial_hz=tracks)


def test_unresolved_independent_tracks_fail_even_with_well_spaced_nominal_labels():
    tracks = np.tile([220., 220.000000001], (8000, 1))
    with pytest.raises(ValueError, match='unresolved'):
        partial_envelopes(np.ones(8000), 8000, 220., partials=[1., 2.], partial_hz=tracks)
