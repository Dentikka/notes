import numpy as np
import pytest
from notes.ears import ExcitationBasis, fit_excitation_gains


def conflicting_tones():
    sr = 8000
    time = np.arange(sr) / sr
    hz = np.array([200., 401., 603., 805., 1007., 1209., 1411., 1613.])
    tones = np.sin(2 * np.pi * hz[:, None] * time)
    # All controls share an acoustic readout: increasing the desired low tone
    # also increases six unwanted higher tones. The reference is outside this family.
    shared = tones[0] + .5 * tones[2:].sum(0)
    basis = ExcitationBasis(np.vstack([tones[1], *[shared / 8 for _ in range(8)]]), 0.)
    reference = tones[0] + tones[1] + .04 * tones[2:].sum(0)
    return sr, hz, basis, reference


def test_amplitude_weighting_favors_dominant_component_in_an_incompatible_family():
    sr, hz, basis, reference = conflicting_tones()
    results = [fit_excitation_gains(reference, basis, sr, reference_hz=hz, model_hz=hz,
                                    initial=(.1,) * 8, max_nfev=80, regularization=0., log_weighting=mode)
               for mode in ['uniform', 'amplitude']]
    # Independently measure the strongest requested tone in the generated audio.
    bin_index = round(hz[0])
    amplitudes = [2 * abs(np.fft.rfft(basis.audio(result.gains))[bin_index]) / sr for result in results]
    assert abs(amplitudes[1] - 1.) < abs(amplitudes[0] - 1.) / 2
    assert amplitudes[1] > amplitudes[0]


def test_explicit_uniform_preserves_default_result_exactly():
    sr, hz, basis, reference = conflicting_tones()
    kwargs = dict(reference_hz=hz, model_hz=hz, initial=(.1,) * 8, max_nfev=3)
    assert fit_excitation_gains(reference, basis, sr, **kwargs) == fit_excitation_gains(
        reference, basis, sr, **kwargs, log_weighting='uniform')


@pytest.mark.parametrize('mode', ['energy', None, 4])
def test_unknown_weighting_is_rejected(mode):
    sr, hz, basis, reference = conflicting_tones()
    with pytest.raises(ValueError, match='log_weighting'):
        fit_excitation_gains(reference, basis, sr, reference_hz=hz, model_hz=hz, log_weighting=mode)
