from dataclasses import replace

import numpy as np
import pytest
from notes import ElectricGuitar, Note, Song, Track
from notes.ears import ExcitationBasis, excitation_basis, fit_excitation_gains
from notes.ears.excitation_fit import _quadratic_bands
from notes.engine import RenderContext, Voice, render_score
from notes.engine.dsp import stable_seed
from scipy.signal import stft


def test_affine_builder_and_gram_equal_native_stft():
    sr = 12000
    ctx = RenderContext(sr, 60)
    inst = ElectricGuitar(amplified=False, pluck_sampling='spectral', dispersion_fit='bracketed',
                         partial_t60_scales=(1.4, .6, 1.1), partial_pitch_cents=(0., 2., 3.))
    voice = Voice(130.8, 48., .8, .7, {'string': 5, 'pitch_interpolation': 'sinc'})

    def render(gains):
        return replace(inst, excitation_gains=gains).voice(voice, ctx, np.random.default_rng(197))[:9600]

    basis = excitation_basis(render, controls=3)
    assert basis.linearity_error < 1e-9
    gains = np.array([2.1, .35, 3.7])
    native = render(tuple(gains))
    assert np.linalg.norm(basis.audio(gains) - native) / np.linalg.norm(native) < 1e-9
    kwargs = dict(fs=sr, nperseg=1440, noverlap=1200, nfft=8192, boundary=None, padded=False)
    hz, _, z = stft(basis.waves, **kwargs)
    masks = np.array([(hz > lo) & (hz < hi) for lo, hi in [(90, 180), (200, 310), (20, 5000)]])
    gram = _quadratic_bands(z, masks)
    coefficients = np.r_[1., gains]
    predicted = np.einsum('i,btij,j->bt', coefficients, gram, coefficients)
    actual = masks.astype(float) @ abs(stft(native, **kwargs)[2]) ** 2
    np.testing.assert_allclose(predicted, actual, rtol=2e-9, atol=1e-16)
    assert not basis.waves.flags.writeable


def test_fit_recovers_synthetic_magnitudes_without_carrier_phase_transfer():
    sr = 8000
    t = np.arange(sr) / sr
    waves = np.array([.015 * np.sin(2 * np.pi * 1700 * t),
                      np.sin(2 * np.pi * 200 * t) * np.exp(-t),
                      np.cos(2 * np.pi * 401 * t) * np.exp(-2 * t),
                      np.sin(2 * np.pi * 604 * t + .4) * np.exp(-3 * t)])
    basis = ExcitationBasis(waves, 0.)
    target = -basis.audio((1.5, .4, 2.2))  # Reference carrier phase is reversed.
    result = fit_excitation_gains(target, basis, sr, reference_hz=(200, 401, 604), model_hz=(200, 401, 604),
                                  max_nfev=60, regularization=0.)
    assert result.cost < result.initial_cost / 100
    assert result.cost < .01
    assert result.supported_cells > 30
    # Test audio against independently measured FFT powers, not just optimizer success.
    actual = basis.audio(result.gains)
    assert actual @ target < 0  # The returned voice keeps its generated carrier.
    fft = dict(fs=sr, nperseg=960, noverlap=800, nfft=8192, boundary=None, padded=False)
    _, time, a = stft(target, **fft)
    _, _, b = stft(actual, **fft)
    use = (time >= .14) & (time <= .65)
    pa, pb = abs(a[:, use]) ** 2, abs(b[:, use]) ** 2
    pa /= pa.sum()
    pb /= pb.sum()
    assert np.mean(abs(pa - pb)) < 1e-7


def test_basis_with_pickup_insert_matches_full_renderer_and_note_overrides():
    sr = 12000
    ctx = RenderContext(sr, 60)
    inst = ElectricGuitar(amplified=False, pickup_q=.7, pickup_cutoff=2300.,
                         pluck_sampling='spectral', dispersion_fit='bracketed')
    gains = (.4, 2.8, 1.2)
    note = Note(48, dur=1.2, vel=.7).param(string=5, pitch_interpolation='sinc', excitation_gains=gains)
    score = Song(Track('test', inst)(note), bpm=60, sample_rate=sr, seed=271, loudness=-35).compile()
    e = score.events[0]
    seed = stable_seed(score.seed, e.track, str(e.time), e.pitch, 0)
    v = Voice(440. * 2 ** ((48. - 69.) / 12), 48., .8, .7, {'string': 5, 'pitch_interpolation': 'sinc'})

    def render(g):
        model = replace(inst, excitation_gains=g)
        x = model.voice(v, ctx, np.random.default_rng(seed))
        for effect in model.inserts():
            x = effect.process(x, ctx)
        return x[:9600]

    basis = excitation_basis(render, controls=3)
    native = render_score(score, workers=1)
    assert native.stats['limiter_db'] == 0
    actual = native.audio.mean(axis=0)[:9600]
    expected = basis.audio(gains)
    actual /= np.linalg.norm(actual)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)


def test_fit_ignores_unrequested_tail():
    sr = 8000
    t = np.arange(2 * sr) / sr
    basis = ExcitationBasis(np.array([.1 * np.sin(2 * np.pi * 700 * t), np.sin(2 * np.pi * 200 * t)]), 0.)
    reference = basis.audio((1.,))
    reference[round(.8 * sr):] = np.random.default_rng(3).normal(0, 10, len(reference) - round(.8 * sr))
    fit = fit_excitation_gains(reference, basis, sr, reference_hz=(200.,), model_hz=(200.,), max_nfev=1)
    assert fit.cost < 1e-10


@pytest.mark.parametrize('nonlinear', [True, False])
def test_rejects_nonlinear_or_nondeterministic_renderer(nonlinear):
    t = np.linspace(0, 1, 1000)
    rng = np.random.default_rng(4)

    def render(g):
        return np.tanh(g[0] * np.sin(20 * t)) if nonlinear else g[0] * np.sin(20 * t) + rng.normal(0, .01, len(t))

    with pytest.raises(ValueError, match='affine excitation probe'):
        excitation_basis(render, controls=1)


def test_no_effect_and_bad_length_renderers_are_rejected():
    with pytest.raises(ValueError, match='does not respond'):
        excitation_basis(lambda g: np.ones(100), controls=2)
    with pytest.raises(ValueError, match='lengths must match'):
        excitation_basis(lambda g: np.ones(100 if not any(g) else 101), controls=2)


@pytest.mark.parametrize('case', ['silent', 'length', 'frequencies', 'window', 'initial', 'budget'])
def test_fit_rejects_unsupported_inputs(case):
    sr = 8000
    t = np.arange(sr) / sr
    basis = ExcitationBasis(np.array([.1 * np.sin(2 * np.pi * 700 * t), np.sin(2 * np.pi * 200 * t)]), 0.)
    target = basis.audio((1.,))
    kw = dict(reference_hz=(200.,), model_hz=(200.,))
    if case == 'silent':
        target = np.zeros(sr)
    elif case == 'length':
        target = target[:-1]
    elif case == 'frequencies':
        kw['model_hz'] = (np.nan,)
    elif case == 'window':
        kw['end'] = 2.
    elif case == 'initial':
        kw['initial'] = (20.,)
    elif case == 'budget':
        kw['max_nfev'] = 0
    with pytest.raises(ValueError):
        fit_excitation_gains(target, basis, sr, **kw)
