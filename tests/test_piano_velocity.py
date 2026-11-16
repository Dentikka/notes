from dataclasses import replace

import numpy as np
import pytest

from notes import Note, Piano, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.hammer import struck_string
from notes.ir import Score


def test_explicit_original_speed_preserves_nonlinear_contact_bit_exact():
    for frequency in (98., 440., 880.):
        for velocity in (.03, .5, .9):
            kwargs = dict(freq=frequency, n=4410, sr=22050, velocity=velocity, radiation=True)
            original = struck_string(**kwargs)
            assert np.array_equal(original, struck_string(**kwargs, hammer_speed=.6+3.4*velocity))
            assert np.array_equal(original, struck_string(**kwargs, hammer_speed=None))


def test_absolute_speed_changes_contact_not_just_output_gain():
    opts = dict(freq=440., n=11025, sr=22050, velocity=.6, radiation=True)
    slow = struck_string(**opts, hammer_speed=.3)
    fast = struck_string(**opts, hammer_speed=3.)
    slow /= np.linalg.norm(slow)
    fast /= np.linalg.norm(fast)
    assert float(slow@fast) < .99
    # A velocity change cannot leak through as a second gain law when an
    # absolute impact speed is given. The note-off exception remains explicit.
    assert np.array_equal(struck_string(**opts, hammer_speed=2.),
                          struck_string(**{**opts, 'velocity': .1}, hammer_speed=2.))
    assert not np.any(struck_string(**opts, hammer_speed=0.))
    assert not np.any(struck_string(**{**opts, 'velocity': 0.}, hammer_speed=2.))


@pytest.mark.parametrize('pitch', [36, 60, 84])
def test_zero_anchored_speed_curve_has_a_continuous_quiet_range(pitch):
    p = Piano(model='hammer_damped', hammer_speed_points=((0., 0.), (1., 4.)))
    context = RenderContext(22050, 60)
    energy = []
    for velocity in (.0001, .01, .1, .5, 1.):
        y = p.voice(Voice(440*2**((pitch-69)/12), pitch, .3, velocity, {}), context,
                    np.random.default_rng(1))
        assert np.isfinite(y).all()
        energy.append(float(y@y))
    assert all(a < b for a, b in zip(energy[:-1], energy[1:], strict=True))
    assert energy[0] < energy[-1]*1e-5


@pytest.mark.parametrize('model', ['hammer', 'hammer_damped'])
def test_speed_curve_override_reset_and_ir_roundtrip(model):
    curve = ((0., 0.), (.5, 1.), (1., 4.))
    inst = Piano(model=model, hammer_speed_points=curve)
    plain = Piano(model=model)
    assert build_instrument(inst.to_spec()) == inst
    context, voice = RenderContext(8000, 60), Voice(440., 69., .25, .75, {})
    def run(p, params):
        return p.voice(replace(voice, params=params), context, np.random.default_rng(3))
    assert np.array_equal(run(inst, {}), run(plain, {'hammer_speed': 2.5}))
    assert np.array_equal(run(inst, {'hammer_speed': None}), run(plain, {}))
    assert np.array_equal(run(inst, {'hammer_speed_points': []}), run(plain, {}))
    assert np.array_equal(run(inst, {'hammer_speed': .8}), run(plain, {'hammer_speed': .8}))
    song = Song(Track('piano', plain)(Note(69, .2).param(hammer_speed_points=curve)),
                bpm=60, sample_rate=8000, loudness=None)
    score = song.compile()
    restored = Score.from_json(score.to_json())
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize('curve', [None, 'bad', [(0, 1)], [(0, 1), (.5, 2)],
    [(0, 1), (1, -1)], [(0, 1), (1, 5)], [(0, 2), (1, 1)], [(0, 0), (0, 1), (1, 4)],
    [(0, 0), (1, float('nan'))], [(0, 0), (1, float('inf'))], [(0, 0, 0), (1, 4)],
    [(0, 0), (1, 'bad')], [(i/64, i/16) for i in range(65)]])
def test_invalid_curves_rejected_on_instrument_and_note(curve):
    with pytest.raises(ValueError, match='hammer_speed_points'):
        Piano(model='hammer', hammer_speed_points=curve)
    with pytest.raises(ValueError, match='hammer_speed_points'):
        Piano(model='hammer').voice(Voice(440., 69., .1, .5, {'hammer_speed_points': curve}),
                                    RenderContext(8000, 60), np.random.default_rng(0))


@pytest.mark.parametrize('speed', [-1., 4.1, float('inf'), float('nan'), 'bad'])
def test_invalid_absolute_speed_rejected(speed):
    with pytest.raises(ValueError, match='hammer_speed'):
        struck_string(440., 800, 8000, .5, hammer_speed=speed)


@pytest.mark.parametrize('model', ['legacy', 'soundboard'])
def test_other_pianos_cannot_silently_ignore_new_impact_controls(model):
    with pytest.raises(ValueError, match='hammer_speed'):
        Piano(model=model, hammer_speed_points=((0., 0.), (1., 4.)))
    p = Piano(model=model)
    for params in ({'hammer_speed': 2.}, {'hammer_speed_points': ((0., 0.), (1., 4.))}):
        with pytest.raises(ValueError, match='hammer_speed'):
            p.voice(Voice(440., 69., .1, .5, params), RenderContext(8000, 60), np.random.default_rng(0))
