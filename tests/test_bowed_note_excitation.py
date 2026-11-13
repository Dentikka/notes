"""Note controls must reach the source, remain independent and survive IR."""
from dataclasses import replace

import numpy as np
import pytest

from notes import Cello, Note, Song, Track, Violin, cat
from notes.engine import RenderContext, Voice
from notes.engine.mixer import render_score
from notes.ir.score import Score


def voice(instrument, params, sr=22050, velocity=.65):
    return instrument.voice(Voice(220., 57, .3, velocity, params), RenderContext(sr, 60),
                            np.random.default_rng(17))


@pytest.mark.parametrize('cls', [Violin, Cello])
@pytest.mark.parametrize('sr', [8000, 22050, 44100])
@pytest.mark.parametrize('velocity', [.35, .9])
@pytest.mark.parametrize('controls', [{'pressure': .22}, {'bow_noise': .011}, {'pressure': 2., 'bow_noise': 0.}])
def test_note_excitation_equals_independently_configured_instrument(cls, sr, velocity, controls):
    instrument = cls(vibrato=0)
    actual = voice(instrument, controls, sr, velocity)
    expected = voice(replace(instrument, **controls), {}, sr, velocity)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, voice(instrument, {}, sr, velocity))


@pytest.mark.parametrize('cls', [Violin, Cello])
def test_default_controls_can_override_a_configured_source(cls):
    baseline = cls(vibrato=0)
    changed = replace(baseline, pressure=2., bow_noise=0.)
    assert np.array_equal(voice(baseline, {}), voice(changed, {'pressure': .55, 'bow_noise': .025}))
    assert np.array_equal(voice(baseline, {}), voice(baseline, {'pressure': .55, 'bow_noise': .025}))


@pytest.mark.parametrize('control', ['pressure', 'bow_noise'])
@pytest.mark.parametrize('value', [-.01, float('nan'), float('inf'), None, 'bad'])
def test_invalid_note_excitation_is_rejected(control, value):
    with pytest.raises(ValueError, match=control):
        voice(Violin(), {control: value})


def test_note_excitation_survives_dsl_ir_and_composes_with_expression_and_body_eq():
    phrase = cat(Note(55, .25).param(pressure=.22, bow_noise=.018, body_eq=((202., 4., -12.),)),
                 Note(69, .25).param(pressure=2., bow_noise=.011,
                                    expression_points=((0., .4), (.2, 1.), (.4, .5))))
    score = Song(Track('violin', Violin(vibrato=0))(phrase), bpm=60, sample_rate=8000,
                 seed=17, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param('pressure') == .22
    assert restored.events[1].param('bow_noise') == .011
    actual = render_score(score).audio
    assert np.array_equal(actual, render_score(restored).audio)
    plain = replace(score, events=tuple(replace(e, params=()) for e in score.events))
    other = render_score(plain).audio
    common = min(actual.shape[1], other.shape[1])
    assert np.linalg.norm(actual[:, :common]-other[:, :common]) > .01
