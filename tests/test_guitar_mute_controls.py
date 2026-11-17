from dataclasses import replace

import numpy as np
import pytest
from notes import ElectricGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice
from notes.engine.mixer import render_score
from notes.ir import Score


def wave(inst, params, *, sr=8000):
    return inst.voice(Voice(220., 57., .35, .7, params), RenderContext(sr, 60), np.random.default_rng(116))


@pytest.mark.parametrize(('name', 'value'), [('mute_decay', .12), ('mute_decay', 1.2),
                                           ('mute_brightness', 0.), ('mute_brightness', 1.)])
def test_per_note_mute_matches_instrument_control(name, value):
    inst = ElectricGuitar(pick_noise=0, pitch_attack=0)
    actual = wave(inst, {'palm_mute': True, name: value}, sr=22050)
    expected = wave(replace(inst, **{name: value}), {'palm_mute': True}, sr=22050)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, wave(inst, {'palm_mute': True}, sr=22050))


def test_shorter_mute_preserves_early_attack_and_damps_tail():
    inst = ElectricGuitar(pick_noise=0, pitch_attack=0, polarization=0)
    short = wave(inst, {'palm_mute': True, 'mute_decay': .15})
    long = wave(inst, {'palm_mute': True, 'mute_decay': 1.2})
    def drop(y):
        return 10*np.log10(np.mean(y[1200:1600]**2)/np.mean(y[160:400]**2))
    assert drop(short) < drop(long)-15


@pytest.mark.parametrize(('name', 'value'), [('mute_decay', 0.), ('mute_decay', -1.),
    ('mute_decay', np.nan), ('mute_decay', np.inf), ('mute_brightness', -.1),
    ('mute_brightness', 1.1), ('mute_brightness', np.nan), ('mute_brightness', np.inf)])
def test_invalid_per_note_mute_rejected(name, value):
    with pytest.raises(ValueError, match='finite'):
        wave(ElectricGuitar(), {'palm_mute': True, name: value})


@pytest.mark.parametrize('name', ['mute_decay', 'mute_brightness'])
def test_mute_override_cannot_silently_apply_to_open_note(name):
    with pytest.raises(ValueError, match='requires palm_mute'):
        wave(ElectricGuitar(), {name: .2})


def test_mute_controls_survive_score_roundtrip():
    score = Song(Track('guitar', ElectricGuitar())(
        Note('A3', .35).param(palm_mute=True, mute_decay=.12, mute_brightness=.4)), bpm=60).compile()
    restored = Score.from_dict(score.to_dict())
    params = dict(restored.events[0].params)
    assert params['mute_decay'] == .12 and params['mute_brightness'] == .4 and params['palm_mute']


@pytest.mark.parametrize(('name', 'value'), [('mute_decay', .12), ('mute_brightness', .25)])
def test_ir_mute_override_reaches_rendered_audio(name, value):
    inst = ElectricGuitar(pick_noise=0, pitch_attack=0)
    note = Note('A3', .35, vel=.7).param(palm_mute=True)
    override = Song(Track('guitar', inst)(note.param(**{name: value})), bpm=60,
                    sample_rate=8000, loudness=None).compile()
    instrument_control = Song(Track('guitar', replace(inst, **{name: value}))(note), bpm=60,
                              sample_rate=8000, loudness=None).compile()
    actual = render_score(Score.from_dict(override.to_dict()), workers=1).audio
    expected = render_score(instrument_control, workers=1).audio
    assert np.array_equal(actual, expected)
