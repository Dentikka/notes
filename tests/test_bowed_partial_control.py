"""Harmonic curves must be independent of noise, pitch, RNG and legacy mode."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import welch

from notes import Cello, Note, Song, Track, Violin
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.ir import Score


def source(instrument, params=None, sr=22050):
    rng = np.random.default_rng(17)
    audio = instrument.voice(Voice(220., 57., 1., .65, params or {}), RenderContext(sr, 60), rng)
    return audio, rng.random(10)


@pytest.mark.parametrize('cls', [Violin, Cello])
@pytest.mark.parametrize('sr', [8000, 22050, 44100])
def test_unit_curves_empty_override_and_ir_roundtrip_preserve_source(cls, sr):
    original = cls(vibrato=.2)
    wave, state = source(original, sr=sr)
    configured = replace(original, partial_gain_points=((0., 1., 1.), (.3, 1., 1.)))
    np.testing.assert_array_equal(wave, source(configured, sr=sr)[0])
    changed = replace(original, partial_gain_points=((0., .1, .3), (.4, 2., 3.)))
    changed_wave, changed_state = source(changed, sr=sr)
    assert np.linalg.norm(changed_wave-wave) > .01
    assert np.array_equal(state, changed_state)
    assert np.array_equal(wave, source(changed, {'partial_gain_points': []}, sr=sr)[0])
    assert build_instrument(changed.to_spec()) == changed
    assert changed_wave[-1] == 0


def test_one_harmonic_gain_does_not_equalize_other_harmonics_or_touch_noise():
    instrument = Violin(vibrato=0, bow_noise=0)
    plain, _ = source(instrument)
    changed, _ = source(instrument, {'partial_gain_points': ((0., 2.),)})
    def band(x, h):
        hz, power = welch(x[4410:19845], fs=22050, nperseg=8192)
        return power[abs(hz-h*220)<20].sum()
    assert 10*np.log10(band(changed, 1)/band(plain, 1)) == pytest.approx(6.0206, abs=.02)
    for h in (2, 3, 6, 10):
        assert 10*np.log10(band(changed, h)/band(plain, h)) == pytest.approx(0., abs=.02)
    noisy = replace(instrument, bow_noise=.1, bow_noise_seed=41)
    a, state = source(noisy)
    b, other = source(noisy, {'partial_gain_points': ((0., 2.),)})
    np.testing.assert_allclose(b-a, changed-plain, atol=1e-15)
    assert np.array_equal(state, other)


def test_time_varying_partial_changes_spectral_shape_within_a_held_note():
    instrument = Cello(vibrato=0, bow_noise=0)
    plain, _ = source(instrument)
    changed, _ = source(instrument, {'partial_gain_points': ((0., 1., 1., .25), (.35, 1., 1., .25),
                                                            (.5, 1., 1., 2.))})
    def ratio(start, end):
        def power(x):
            y = x[round(start*22050):round(end*22050)]
            hz, p = welch(y, fs=22050, nperseg=2048)
            return p[abs(hz-660)<30].sum()
        return np.sqrt(power(changed)/power(plain))
    assert ratio(.2, .32) == pytest.approx(.25, abs=.02)
    assert ratio(.65, .9) == pytest.approx(2., abs=.02)


def test_note_curve_survives_full_score_serialization_and_expression():
    points = ((0., 1., .5), (.4, .5, 2.))
    note = Note(57, .6).param(partial_gain_points=points, expression_points=((0., .3), (.5, 1.)))
    score = Song(Track('strings', Violin())(note), bpm=60, sample_rate=8000, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param('partial_gain_points')
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize('points', [None, True, [0., 1.], [[0.]], [[0., 1.], [.1, 1., 2.]],
    [[-.1, 1.]], [[0., 1.], [0., 2.]], [[0., -1.]], [[0., 16.1]], [[0., np.nan]],
    [[0., 1.]*42], [[float(i), 1.] for i in range(513)]])
def test_invalid_curves_rejected_in_constructor_and_note(points):
    with pytest.raises(ValueError, match='partial_gain_points'):
        Violin(partial_gain_points=points)
    with pytest.raises(ValueError, match='partial_gain_points'):
        source(Violin(), {'partial_gain_points': points})
