from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from notes import Piano
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.piano_impact import impact_duration, piano_impact
from notes.ir import NoteEvent, Score, TrackSpec


@pytest.mark.parametrize('sr', [8000, 22050, 44100])
def test_impact_is_causal_repeatable_and_prefix_independent(sr):
    options = dict(decay=.18, color=.5, velocity_power=.5, delay=.02)
    n = sr
    y = piano_impact(n, sr, .7, np.random.default_rng(13), **options)
    again = piano_impact(n, sr, .7, np.random.default_rng(13), **options)
    prefix = piano_impact(round(.12*sr), sr, .7, np.random.default_rng(13), **options)
    assert np.array_equal(y, again) and np.array_equal(prefix, y[:len(prefix)])
    assert np.isfinite(y).all() and not np.any(y[:round(.02*sr)])
    assert y[-1] == 0 and np.linalg.norm(y) > 0
    slow = piano_impact(n, sr, .175, np.random.default_rng(13), **options)
    np.testing.assert_allclose(slow, y*.5, atol=2e-14, rtol=2e-14)


def test_zero_velocity_and_empty_output_do_not_draw_randomness():
    rng = np.random.default_rng(4)
    assert not np.any(piano_impact(8000, 8000, 0., rng, velocity_power=0.))
    assert len(piano_impact(0, 8000, .7, rng)) == 0
    assert rng.random() == np.random.default_rng(4).random()


def test_piano_adds_body_after_string_gate_without_altering_string_source():
    context = RenderContext(8000, 60)
    voice = Voice(440., 69., .03, .7, {})
    plain = Piano(model='hammer_damped')
    enabled = Piano(model='hammer_damped', impact_gain=.005, impact_decay=.3, impact_delay=.01)
    bare = plain.voice(voice, context, np.random.default_rng(8))
    actual = enabled.voice(voice, context, np.random.default_rng(8))
    assert len(actual) > len(bare)
    body = piano_impact(len(actual), context.sr, voice.vel, np.random.default_rng(8), decay=.3, delay=.01)
    expected = np.pad(bare, (0, len(actual)-len(bare)))+enabled.level*.005*body
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-13)
    assert np.linalg.norm(actual[len(bare):]) > 0
    rng = np.random.default_rng(8)
    reset = enabled.voice(replace(voice, params={'impact_gain': 0.}), context, rng)
    assert np.array_equal(reset, bare)
    assert rng.random() == np.random.default_rng(8).random()


@pytest.mark.parametrize('pitch', [69, 69.])
def test_body_controls_survive_registry_and_public_ir_render(pitch):
    inst = Piano(model='hammer_damped', impact_gain=.003, impact_decay=.25,
                 impact_color=-.2, impact_velocity_power=.3, impact_delay=.014)
    assert build_instrument(inst.to_spec()) == inst
    score = Score((NoteEvent(Fraction(0), Fraction(1, 4), pitch, .7, 'p'),),
                  (TrackSpec('p', inst.to_spec()),), Fraction(1), sample_rate=8000, seed=88,
                  master={'loudness': None, 'ceiling': -1.})
    restored = Score.from_json(score.to_json())
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


def test_note_overrides_match_constructor_controls():
    params = {'impact_gain': .003, 'impact_decay': .2, 'impact_color': .3,
              'impact_velocity_power': .4, 'impact_delay': .02}
    plain = Piano(model='hammer_damped')
    configured = Piano(model='hammer_damped', **params)
    context, voice = RenderContext(8000, 60), Voice(440., 69., .25, .7, {})
    a = plain.voice(replace(voice, params=params), context, np.random.default_rng(6))
    b = configured.voice(voice, context, np.random.default_rng(6))
    assert np.array_equal(a, b)


def test_note_cannot_exceed_instrument_declared_tail_silently():
    plain = Piano(model='hammer_damped', sustain=.05, release=.05)
    context = RenderContext(8000, 60)
    voice = Voice(440., 69., .01, .7, {'impact_gain': .01, 'impact_decay': .5})
    with pytest.raises(ValueError, match='tail exceeds'):
        plain.voice(voice, context, np.random.default_rng(5))
    configured = replace(plain, impact_gain=.01, impact_decay=.5)
    assert configured.release_time() >= impact_duration(.5)
    assert np.isfinite(configured.voice(voice, context, np.random.default_rng(5))).all()


@pytest.mark.parametrize('params', [
    {'impact_gain': -.1}, {'impact_gain': 1.1}, {'impact_gain': float('nan')},
    {'impact_decay': .02}, {'impact_decay': .6}, {'impact_color': -1.1}, {'impact_color': 2.1},
    {'impact_velocity_power': -.1}, {'impact_velocity_power': 2.1}, {'impact_delay': -.01},
    {'impact_delay': .061}, {'impact_color': float('inf')},
])
def test_invalid_instrument_and_note_controls_rejected(params):
    with pytest.raises(ValueError, match='impact'):
        Piano(model='hammer_damped', **params)
    with pytest.raises(ValueError, match='impact'):
        Piano(model='hammer_damped').voice(Voice(440., 69., .1, .7, params),
                                           RenderContext(8000, 60), np.random.default_rng(1))


@pytest.mark.parametrize('model', ['legacy', 'soundboard'])
def test_unsupported_pianos_reject_enabled_body(model):
    with pytest.raises(ValueError, match='contact piano'):
        Piano(model=model, impact_gain=.01)
    with pytest.raises(ValueError, match='contact piano'):
        Piano(model=model).voice(Voice(440., 69., .1, .7, {'impact_gain': .01}),
                                 RenderContext(8000, 60), np.random.default_rng(1))
