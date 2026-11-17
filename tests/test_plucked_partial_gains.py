"""Static physical-coordinate readout, independent mechanics and public IR."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm, solve

from notes import ModalString, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.mechanical_string import _initial_state, _output, modal_string
from notes.engine.palm_contact import _sample, _state, palm_string
from notes.ir import Score


@pytest.mark.parametrize('resistance', [0., .04, 10.])
@pytest.mark.parametrize('peak', [0., .01])
def test_weighted_readout_matches_independent_physical_coordinates(resistance, peak):
    model = _state(220., 8, 4e-5, 125., 20., peak, 70., .5, 9., 2.5, .11, 0., resistance)
    initial = _initial_state(model, .19, 5000.)
    count = len(model['M'])
    q = model['U']@(initial[:count]/model['omega'])
    velocity = model['U']@initial[count:]
    physical = np.block([[np.zeros((count, count)), np.eye(count)],
                         [-solve(model['M'], model['K']), -solve(model['M'], model['C'])]])
    gains = (1.2, .1, 2., 0., .9)
    numbers = np.arange(1, 9)
    sensor = np.sin(np.pi*numbers*.07)*np.sinc(numbers*.03/2)
    sensor[:len(gains)] *= gains
    if peak:
        sensor = np.r_[sensor, -.07]
    output = _output(model, .07, .03, 0., True, gains)
    wave = _sample(model, initial, output, 301, 16000, None, 'spatial')
    for k in (0, 1, 30, 100, 300):
        state = expm(physical*k/16000)@np.r_[q, velocity]
        assert wave[k] == pytest.approx(sensor@state[count:], abs=2e-10)


@pytest.mark.parametrize('palm', [False, True])
@pytest.mark.parametrize('rounding', ['release', 'spatial'])
def test_identity_and_unspecified_coordinates_preserve_output(palm, rounding):
    options = {'rounding': rounding, 'modes': 12, 'peak': .01}
    render = modal_string
    if palm:
        render = palm_string
        options.update(position=.11, contact_width=.05, resistance=.03)
    base = render(220., 3000, 16000, **options)
    assert np.array_equal(base, render(220., 3000, 16000, partial_gains=(1.,)*80, **options))
    first = render(220., 3000, 16000, partial_gains=(.5,), **options)
    extended = render(220., 3000, 16000, partial_gains=(.5,)+(1.,)*79, **options)
    assert np.array_equal(first, extended)
    assert np.linalg.norm(first-base) > 1e-4


def test_zero_string_readout_keeps_the_bridge_and_leaves_mechanics_unchanged():
    model = _state(220., 8, 4e-5, 125., 20., .01, 70., .5, 9., 2.5, .11, .05, .03)
    snapshots = {k: model[k].copy() for k in ('M', 'K', 'C', 'A')}
    actual = _output(model, .07, .03, 0., True, (0.,)*80)
    expected = np.r_[np.zeros(9), -.07*model['U'][-1]]
    assert np.array_equal(actual, expected)
    assert np.linalg.norm(actual) > 0
    for key, before in snapshots.items():
        assert np.array_equal(model[key], before)


@pytest.mark.parametrize('bad', [None, 1., 'one', [np.nan], [np.inf], [-.1], [16.1], [[1.]], [1.]*81])
def test_invalid_gains_are_rejected_at_public_boundaries(bad):
    with pytest.raises(ValueError, match='partial_gains'):
        ModalString(partial_gains=bad)
    with pytest.raises(ValueError, match='partial_gains'):
        modal_string(220., 0, 16000, partial_gains=bad)
    with pytest.raises(ValueError, match='partial_gains'):
        palm_string(220., 0, 16000, position=.1, contact_width=.04, resistance=0., partial_gains=bad)


def test_per_note_gain_override_and_public_json_pcm_roundtrip(tmp_path):
    inst = ModalString(modes=12, pick_noise=0, partial_gains=[.8, 1.2], bridge_mobility=0)
    assert inst.partial_gains == (.8, 1.2)
    assert build_instrument(inst.to_spec()) == inst
    voice = Voice(220., 57., .1, .7, {'partial_gains': [1.2, .3]})
    ctx = RenderContext(16000, 120)
    actual = inst.voice(voice, ctx, np.random.default_rng(1))
    expected = replace(inst, partial_gains=(1.2, .3)).voice(replace(voice, params={}), ctx, np.random.default_rng(1))
    assert np.array_equal(actual, expected)
    score = Song(Track('test', inst)(Note('A3', 1).param(partial_gains=[1.2, .3])),
                 bpm=120, sample_rate=16000, seed=117, loudness=None).compile()
    path = tmp_path/'score.json'
    score.to_json(path)
    restored = Score.from_json(path)
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)
    with pytest.raises(ValueError, match='partial_gains'):
        inst.voice(replace(voice, params={'partial_gains': [-1.]}), ctx, np.random.default_rng(1))
