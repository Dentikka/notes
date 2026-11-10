"""Continuous held-note dynamics must survive IR and precede nonlinear inserts."""
from fractions import Fraction

import numpy as np
import pytest

from notes import Cello, ElectricGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice, render_score
from notes.engine.articulation import expression_curve
from notes.engine.dsp import stable_seed
from notes.engine.mixer import _render_voices
from notes.ir.score import Score


def score(instrument=None, points=None, bpm=60):
    note = Note('A3', dur=2, vel=.7)
    if points is not None:
        note = note.param(expression_points=points)
    return Song(Track('solo', instrument or Cello(vibrato=0))(note), bpm=bpm, sample_rate=16000,
                seed=17, loudness=None).compile()


def test_curve_has_real_time_knots_bounds_and_endpoint_hold():
    curve = expression_curve({'expression_points': ((.1, .2), (.5, 1), (.9, .4))}, 1200, 1000)
    assert curve[0] == curve[100] == .2
    assert curve[300] == pytest.approx(.6)
    assert curve[500] == 1 and curve[900] == curve[-1] == .4
    assert curve.min() >= 0 and curve.max() <= 1
    assert expression_curve({}, 10, 1000) is None


def test_crescendo_and_diminuendo_change_a_held_voice_without_restriking_it():
    points = ((0., .2), (.6, 1.), (1.4, .3))
    ir = score(points=points)
    restored = Score.from_dict(ir.to_dict())
    event = restored.events[0]
    assert event.vel == .7 and event.time == 0 and event.end == 2 and len(restored.events) == 1
    instrument = Cello(vibrato=0)
    ctx = RenderContext(16000, 60, seed=17)
    raw = instrument.voice(Voice(220., 57., 2., .7, {}), ctx,
                           np.random.default_rng(stable_seed(17, 'solo', str(Fraction(0)), 57., 0)))
    output = _render_voices(instrument, restored.events, restored.tempo_map(), ctx, len(raw), 'solo')
    expected = raw*expression_curve({'expression_points': points}, len(raw), ctx.sr)
    np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)
    # A sustained swell cannot be reproduced by one constant velocity multiplier.
    gain = float(np.dot(output, raw)/np.dot(raw, raw))
    assert np.linalg.norm(output-gain*raw)/np.linalg.norm(output) > .2


def test_unity_preserves_old_audio_and_zero_expression_is_silent():
    original = render_score(score()).audio
    np.testing.assert_array_equal(original, render_score(score(points=((0., 1.), (1., 1.)))).audio)
    assert not render_score(score(points=((0., 0.),))).audio.any()


def test_constant_expression_matches_excitation_level_before_distortion():
    from dataclasses import replace

    inst = ElectricGuitar(level=.15, drive=.8, pick_noise=0)
    expressed = render_score(score(inst, ((0., .5),))).audio
    expected = render_score(score(replace(inst, level=.075))).audio
    np.testing.assert_allclose(expressed, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('points', [[], [[0]], [[0, 1, 2]], [[-.1, .5]], [[0, 1], [0, .2]],
                                   [[1, .5], [.1, .6]], [[0, -1]], [[0, 1.1]], [[0, np.nan]],
                                   [[np.inf, .5]]])
def test_invalid_expression_points_are_rejected(points):
    with pytest.raises(ValueError, match='expression_points'):
        render_score(score(points=points))
