"""Independent spectral tuning and timing/cache contracts of source-only compensation."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from notes import ModalBowedString, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument
from notes.engine.instruments import modal_bowed
from notes.engine.instruments.modal_bowed import _mean_pitch_correction
from notes.ir import Score


def spectral_frequency(wave, sr, expected):
    # Independent of the demodulated-phase estimator used by calibration.
    signal = wave[round(1.6*sr):round(2.7*sr)]
    time = np.arange(len(signal))/sr
    weighted = signal*np.hanning(len(signal))
    def objective(freq):
        return -abs(weighted@np.exp(-2j*np.pi*freq*time))**2
    return minimize_scalar(objective, bounds=(expected*.985, expected*1.015),
                           method='bounded', options={'xatol': 1e-9}).x


@pytest.mark.parametrize('sr,freq,pitch,geometry', [
    (8000, 110., 45., {}),
    (16000, 440., 69., {'open_pitches': (55., 62., 69., 76.), 'length': .328,
                       'tension': 55., 'body_size': 1.}),
])
def test_actual_sounding_pitch_from_an_independent_spectral_peak(sr, freq, pitch, geometry):
    options = dict(modes=32, bandwidth=7000., oversample=8, **geometry)
    plain = ModalBowedString(**options)
    corrected = ModalBowedString(**options, pitch_compensation=True)
    ctx = RenderContext(sr, 60)
    voice = Voice(freq, pitch, 3., .7, {})
    before = plain.voice(voice, ctx, np.random.default_rng(17))
    after = corrected.voice(voice, ctx, np.random.default_rng(17))
    f0, f1 = (spectral_frequency(y, sr, freq) for y in (before, after))
    assert abs(1200*np.log2(f0/freq)) > 1.
    assert abs(1200*np.log2(f1/freq)) < .3
    assert len(before) == len(after) and after[-1] == 0
    assert build_instrument(corrected.to_spec()) == corrected
    by_note = plain.voice(Voice(freq, pitch, 3., .7, {'pitch_compensation': True}), ctx, np.random.default_rng(29))
    np.testing.assert_array_equal(after, by_note)


def test_calibration_is_independent_of_duration_and_invalidated_by_physical_controls():
    _mean_pitch_correction.cache_clear()
    instrument = ModalBowedString(modes=16, bandwidth=4000., oversample=8, pitch_compensation=True)
    ctx = RenderContext(8000, 60)
    def render(duration, **params):
        return instrument.voice(Voice(110., 45., duration, .7, params), ctx, np.random.default_rng(17))
    short = render(1.)
    first = _mean_pitch_correction.cache_info()
    long = render(1.5)
    second = _mean_pitch_correction.cache_info()
    assert second.misses == first.misses and second.hits == first.hits+1
    np.testing.assert_allclose(short[:6400], long[:6400], atol=1e-10)
    render(.3, open_pitch=36.)
    assert _mean_pitch_correction.cache_info().misses == first.misses+1


def test_bare_default_silence_and_score_parameters_are_preserved():
    instrument = ModalBowedString(modes=16, bandwidth=4000., oversample=8)
    ctx = RenderContext(8000, 60)
    def voice(params=None, velocity=.7):
        return instrument.voice(Voice(110., 45., .3, velocity, params or {}), ctx, np.random.default_rng(17))
    np.testing.assert_array_equal(voice(), voice({'pitch_compensation': False}))
    assert not voice({'pitch_compensation': True}, velocity=0).any()
    assert not voice({'pitch_compensation': True, 'force_ratio': 0.}).any()
    score = Song(Track('bow', instrument)(Note('A2', dur=1.).param(pitch_compensation=True)),
                 bpm=60, sample_rate=8000, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param('pitch_compensation') is True
    with pytest.raises(ValueError, match='boolean'):
        voice({'pitch_compensation': 'yes'})
    with pytest.raises(ValueError, match='boolean'):
        ModalBowedString(pitch_compensation=1)


def test_unreliable_calibration_does_not_turn_into_a_zero_correction(monkeypatch):
    _mean_pitch_correction.cache_clear()
    def unpitched(freq, n, sr, **controls):
        return np.random.default_rng(17).normal(size=n), {}
    monkeypatch.setattr(modal_bowed, 'modal_bowed_string', unpitched)
    instrument = ModalBowedString(modes=16, bandwidth=4000., oversample=8, pitch_compensation=True)
    with pytest.raises(ValueError, match='unstable or missing'):
        instrument.voice(Voice(110., 45., .3, .7, {}), RenderContext(8000, 60), np.random.default_rng(17))
    _mean_pitch_correction.cache_clear()


def test_detuned_e2_preserves_contact_regime_and_corrects_late_sounding_pitch():
    # This continuous-frequency case defeats changing the bare oscillator:
    # small input-frequency changes can select a different transient/regime.
    freq, sr = 81.28183080765055, 22050
    instrument = ModalBowedString(pitch_compensation=True)
    wave = instrument.voice(Voice(freq, 40., 3., .65, {}), RenderContext(sr, 60), np.random.default_rng(17))
    measured = spectral_frequency(wave, sr, freq)
    assert abs(1200*np.log2(measured/freq)) < 1.
