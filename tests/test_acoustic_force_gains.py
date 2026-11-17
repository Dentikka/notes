"""Initial force coefficients, consistent static loading and public acoustic IR."""
from dataclasses import replace

import numpy as np
import pytest

from notes import AcousticGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.instruments.strings import _t60
from notes.engine.modal_acoustic import pluck_force
from notes.ir import Score


def test_removing_one_force_mode_matches_an_independent_analytic_component():
    sr, freq, n, pick, hardness = 8000, 220., 4000, .18, .7
    options = dict(pick=pick, hardness=hardness, polarization=0., roughness=0., stiffness=0.)
    base, load = pluck_force(freq, n, sr, np.random.default_rng(1), **options)
    cut, changed_load = pluck_force(freq, n, sr, np.random.default_rng(1), force_gains=(1., 0.), **options)
    hz = 2*freq
    cutoff = min(1500*2**(3*hardness), sr*.45)
    amplitude = 2/np.pi*np.sin(2*np.pi*pick)/2/(1+(hz/cutoff)**2)
    t60 = _t60(7., freq)
    loss = 1/t60+max(0, 1/3.-1/t60)*(hz*hz-freq*freq)/(4000**2-freq*freq)
    gamma, omega = np.log(1000)*loss, 2*np.pi*hz
    t = np.arange(n)/sr
    expected = amplitude*np.exp(-gamma*t)*(np.cos(omega*t)+gamma/omega*np.sin(omega*t))
    np.testing.assert_allclose(base-cut, expected, atol=5e-15)
    assert load-changed_load == pytest.approx(amplitude, abs=1e-15)
    assert cut[0] == pytest.approx(changed_load, abs=1e-15)


def test_force_identity_preserves_random_stream_and_missing_gains_are_one():
    a, b = np.random.default_rng(8), np.random.default_rng(8)
    raw, static = pluck_force(220., 1000, 8000, a)
    identity, held = pluck_force(220., 1000, 8000, b, force_gains=(1.,)*80)
    assert np.array_equal(raw, identity) and static == held and a.random() == b.random()
    x, sx = pluck_force(220., 1000, 8000, np.random.default_rng(8), force_gains=(.4,))
    y, sy = pluck_force(220., 1000, 8000, np.random.default_rng(8), force_gains=(.4,)+(1.,)*79)
    assert np.array_equal(x, y) and sx == sy


@pytest.mark.parametrize('phase', ['designed', 'minimum'])
@pytest.mark.parametrize('twelve', [False, True])
def test_scaling_force_and_preload_scales_attack_body_and_noteoff(phase, twelve):
    ctx = RenderContext(8000, 60)
    inst = AcousticGuitar(model='modal', body_phase=phase, twelve=twelve)
    voice = Voice(220., 57., .08, .7, {})
    a = inst.voice(voice, ctx, np.random.default_rng(9))
    b = replace(inst, force_gains=(2.,)*80).voice(voice, ctx, np.random.default_rng(9))
    np.testing.assert_allclose(b, 2*a, atol=2e-13, rtol=2e-13)
    assert np.linalg.norm(a[ctx.samples(voice.dur):]) > 1e-5
    silence = replace(inst, force_gains=(0.,)*80).voice(voice, ctx, np.random.default_rng(9))
    assert np.max(abs(silence)) == 0


@pytest.mark.parametrize('bad', [None, [np.nan], [-1.], [16.1], [[1.]], [1.]*81])
def test_invalid_force_gains_rejected_before_empty_output(bad):
    with pytest.raises(ValueError, match='force_gains'):
        pluck_force(220., 0, 8000, np.random.default_rng(1), force_gains=bad)
    with pytest.raises(ValueError, match='force_gains'):
        AcousticGuitar(model='modal', force_gains=bad)


@pytest.mark.parametrize('model', ['waveguide', 'coupled'])
def test_other_models_do_not_silently_ignore_nonempty_force_gains(model):
    with pytest.raises(ValueError, match='requires'):
        AcousticGuitar(model=model, force_gains=(1.,))
    with pytest.raises(ValueError, match='requires'):
        AcousticGuitar(model=model).voice(Voice(220., 57., .1, .7, {'force_gains': [1.]}),
            RenderContext(8000, 60), np.random.default_rng(1))


def test_constructor_note_override_and_json_pcm_roundtrip(tmp_path):
    inst = AcousticGuitar(model='modal', force_gains=[.7, 1.2])
    assert inst.force_gains == (.7, 1.2) and build_instrument(inst.to_spec()) == inst
    ctx, v = RenderContext(8000, 60), Voice(220., 57., .1, .7, {'force_gains': [1.3, .4]})
    a = inst.voice(v, ctx, np.random.default_rng(1))
    b = replace(inst, force_gains=(1.3, .4)).voice(replace(v, params={}), ctx, np.random.default_rng(1))
    assert np.array_equal(a, b)
    score = Song(Track('g', inst)(Note('A3', .25).param(force_gains=[1.3, .4])),
                 bpm=60, sample_rate=8000, seed=17, loudness=None).compile()
    path = tmp_path/'score.json'
    score.to_json(path)
    assert np.array_equal(render_score(score).audio, render_score(Score.from_json(path)).audio)
