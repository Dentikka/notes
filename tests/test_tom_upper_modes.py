from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import welch
from scipy.special import jn_zeros

from notes import Note, RockKit, Song, Track
from notes.engine import RenderContext, Voice, render_score
from notes.engine.drum_physics import _circular_membrane_modes, _membrane_upper_modes
from notes.engine.dsp import stable_seed
from notes.engine.instruments.rock_drums import _tom
from notes.ir import Score


def test_circular_roots_cover_the_lowest512_against_an_independent_larger_grid():
    ratios, orders = _circular_membrane_modes()
    expected = sorted((float(z), m) for m in range(129) for z in jn_zeros(m, 64))[:512]
    np.testing.assert_allclose(ratios, np.array([z for z, m in expected])/jn_zeros(0, 1)[0], rtol=1e-13)
    assert np.array_equal(orders, [m for z, m in expected]) and np.all(np.diff(ratios) > 0)
    assert np.allclose(ratios[:7], [1, 1.594, 2.136, 2.296, 2.653, 2.918, 3.156], atol=.002)
    with pytest.raises(ValueError):
        ratios[0] = 4


def upper(count=8, damping=0., f0=100., glide=0.):
    state = np.random.default_rng(17).bit_generator.state
    return _membrane_upper_modes(np.arange(8000)/8000, f0, .16, state, 8000,
        n_modes=count, damping=damping, glide=glide, glide_tau=.08)


def test_first_added_mode_has_the_expected_bessel_frequency_and_damping_changes_only_its_loss():
    rng = np.random.default_rng(stable_seed('tom-extra-modes', np.random.default_rng(17).bit_generator.state))
    jitter = rng.normal(0, .01, 512)[7]
    expected = 100*jn_zeros(2, 2)[-1]/jn_zeros(0, 1)[0]*(1+jitter)
    a, b = upper(), upper(damping=1.6)
    for y in [a, b]:
        f, p = welch(y, 8000, nperseg=8000)
        assert abs(f[np.argmax(p)]-expected) < 1.
    def tail_ratio(y):
        return np.mean(y[1600:2400]**2)/np.mean(y[160:320]**2)
    assert tail_ratio(a) > 1000*tail_ratio(b)


def test_upper_layer_is_zero_when_all_modes_exceed_maximum_glide_band():
    assert not np.any(upper(count=512, f0=600., glide=1.))
    assert not np.any(upper(count=7))


def test_upper_layer_does_not_redraw_the_original_head_stick_or_noise():
    sr = 8000
    a, b = np.random.default_rng(17), np.random.default_rng(17)
    state = a.bit_generator.state
    old = _tom(sr, a, .7, 100., 1., .16)
    rich = _tom(sr, b, .7, 100., 1., .16, mode_count=128)
    added = _membrane_upper_modes(np.arange(len(old))/sr, 100., .16, state, sr,
        n_modes=128, damping=1.6, glide=.22*.7, glide_tau=.08)
    np.testing.assert_allclose(rich-old, added, atol=2e-14)
    assert a.random() == b.random()


@pytest.mark.parametrize('sound', ['tom_lo', 'tom_mid', 'tom_hi'])
def test_note_controls_match_constructor_with_extended_modes(sound):
    v = Voice(220., 47., .1, .7, {'sound': sound, 'tom_modes': 128, 'tom_damping': .8})
    ctx = RenderContext(8000, 60)
    a = RockKit().voice(v, ctx, np.random.default_rng(17))
    b = RockKit(tom_modes=128, tom_damping=.8).voice(
        replace(v, params={'sound': sound}), ctx, np.random.default_rng(17))
    assert np.array_equal(a, b)


@pytest.mark.parametrize('kwargs', [{'tom_modes': 6}, {'tom_modes': 513}, {'tom_modes': True},
    {'tom_modes': 7.5}, {'tom_damping': -1}, {'tom_damping': np.nan}])
def test_invalid_upper_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        RockKit(**kwargs)
    with pytest.raises(ValueError):
        RockKit().release_time_for(kwargs)


def test_rich_head_json_pcm_and_non_tom_isolation(tmp_path):
    kit = RockKit(tom_modes=128, tom_damping=.8)
    ctx, v = RenderContext(8000, 60), Voice(220., 38., .1, .7, {})
    assert np.array_equal(kit.voice(v, ctx, np.random.default_rng(17)),
                          RockKit().voice(v, ctx, np.random.default_rng(17)))
    with pytest.raises(ValueError, match='tom sound'):
        RockKit().voice(replace(v, params={'tom_modes': 128}), ctx, np.random.default_rng(17))
    score = Song(Track('tom', kit)(Note(47, .1)), bpm=60, sample_rate=8000, seed=117, loudness=None).compile()
    path = tmp_path/'score.json'
    score.to_json(path)
    assert np.array_equal(render_score(score).audio, render_score(Score.from_json(path)).audio)
