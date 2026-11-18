from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from notes import Note, RockKit, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.drum_physics import modes
from notes.ir import Score


def test_modal_pitch_sag_matches_independent_frequency_integration():
    sr, freq, glide, relaxation = 8000, 200., .5, .2
    t = np.arange(1600)/sr
    phase = np.random.default_rng(17).uniform(0, 2*np.pi)
    reference = solve_ivp(lambda time, _: [2*np.pi*freq*(1+glide*np.exp(-time/relaxation))],
        [0., .2], [phase], t_eval=t, rtol=1e-11, atol=1e-12).y[0]
    y = modes(t, np.array([freq]), np.array([1.]), np.array([.3]), np.random.default_rng(17),
              glide=glide, glide_tau=relaxation)
    np.testing.assert_allclose(y, np.exp(-t/.3)*np.sin(reference), atol=2e-7)


@pytest.mark.parametrize('coupling', [0., .4, 1.])
def test_both_head_glides_follow_declared_velocity_blend_without_changing_damping(coupling, monkeypatch):
    captured = []
    def membrane(t, f0, tau, rng, **kwargs):
        captured.append((tau, kwargs['glide'], kwargs['glide_tau']))
        return np.zeros_like(t)
    monkeypatch.setattr('notes.engine.instruments.rock_drums.membrane', membrane)
    kit = RockKit(tom_glide=.6, tom_glide_time=.2, tom_velocity_glide=coupling)
    for velocity in [.25, 1.]:
        kit.voice(Voice(220., 47., .1, velocity, {}), RenderContext(8000, 60), np.random.default_rng(17))
        strength = (1-coupling)+coupling*velocity
        assert captured[-2][1] == pytest.approx(.6*strength)
        assert captured[-1][1] == pytest.approx(.6*strength*(.05/.22))
        assert captured[-2][0] == pytest.approx(.16) and captured[-1][0] == pytest.approx(.176)
        assert captured[-2][2] == captured[-1][2] == .2


@pytest.mark.parametrize('sound', ['tom_lo', 'tom_mid', 'tom_hi'])
def test_note_controls_match_constructor_and_preserve_rng(sound):
    options = {'tom_glide': .6, 'tom_glide_time': .2, 'tom_velocity_glide': 0.}
    v = Voice(220., 47., .1, .3, dict(options, sound=sound))
    a, b = np.random.default_rng(17), np.random.default_rng(17)
    x = RockKit().voice(v, RenderContext(8000, 60), a)
    y = RockKit(**options).voice(replace(v, params={'sound': sound}), RenderContext(8000, 60), b)
    assert np.array_equal(x, y) and a.random() == b.random()


@pytest.mark.parametrize('sound', ['kick', 'snare', 'hat', 'open_hat', 'crash', 'ride', 'rim', 'pedal_hat'])
def test_tom_constructor_controls_leave_other_pieces_unchanged(sound):
    v, ctx = Voice(220., 38., .1, .7, {'sound': sound}), RenderContext(8000, 60)
    a = RockKit().voice(v, ctx, np.random.default_rng(17))
    b = RockKit(tom_glide=.6, tom_glide_time=.2, tom_velocity_glide=0).voice(v, ctx, np.random.default_rng(17))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match='tom sound'):
        RockKit().voice(replace(v, params=dict(v.params, tom_glide=.6)), ctx, np.random.default_rng(17))


@pytest.mark.parametrize('options', [{'tom_glide': -1}, {'tom_glide': np.nan}, {'tom_glide_time': 0},
    {'tom_glide_time': np.inf}, {'tom_velocity_glide': -.1}, {'tom_velocity_glide': 1.1}, {'tom_glide': None}])
def test_invalid_global_and_note_glide_controls(options):
    with pytest.raises(ValueError):
        RockKit(**options)
    with pytest.raises(ValueError):
        RockKit().release_time_for(options)


def test_public_ir_and_tail_contract_with_tom_glide(tmp_path):
    inst = RockKit(tom_glide=.6, tom_glide_time=.2, tom_velocity_glide=0.)
    assert build_instrument(inst.to_spec()) == inst
    assert inst.release_time_for({'tom_glide_time': .5}) == inst.release_time()
    score = Song(Track('kit', inst)(Note(47, .1).param(tom_glide=.4)), bpm=60,
                 sample_rate=8000, seed=117, loudness=None).compile()
    p = tmp_path/'score.json'
    score.to_json(p)
    assert np.array_equal(render_score(score).audio, render_score(Score.from_json(p)).audio)
