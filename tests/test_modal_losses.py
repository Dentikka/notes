"""Independent decay poles, passive coupling and public per-note round trips."""
import json

import numpy as np
import pytest
from notes import ModalString, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument
from notes.engine.mechanical_string import (
    _initial_state,
    _mechanics,
    _state,
    modal_string,
)
from notes.engine.palm_contact import _state as palm_state
from notes.ir import Score
from scipy.linalg import eigvalsh, expm


def test_isolated_pole_losses_have_assigned_times_and_preserve_other_mechanics():
    base = _state(110., 8, .0001, 125., 20., 0., 70., .65, 9., 2.5)
    modified = _state(110., 8, .0001, 125., 20., 0., 70., .65, 9., 2.5,
                      partial_t60s=(11., 1.5, None, 4.))
    np.testing.assert_array_equal(base['M'], modified['M'])
    np.testing.assert_array_equal(base['K'], modified['K'])
    np.testing.assert_array_equal(base['C'][2], modified['C'][2])
    np.testing.assert_array_equal(base['C'][4:], modified['C'][4:])
    actual = -np.log(1000)/modified['string_poles'].real
    np.testing.assert_allclose(actual[[0, 1, 3]], [11., 1.5, 4.], rtol=1e-10)
    np.testing.assert_array_equal(_initial_state(base, .15, 6400), _initial_state(modified, .15, 6400))


@pytest.mark.parametrize('contact', [False, True])
def test_modified_losses_remain_passive_with_body_and_contact(contact):
    args = (110., 8, .0001, 125., 20., .01, 70., .65, 9., 2.5)
    model = (palm_state(*args, .08, .04, .001, partial_t60s=(12., .4, None, 2.))
             if contact else _state(*args, partial_t60s=(12., .4, None, 2.)))
    assert eigvalsh(model['C']).min() > 0
    assert eigvalsh(model['A']+model['A'].T).max() < 1e-10
    initial = _initial_state(model, .2, 5000, .4)
    energy = [float((expm(model['A']*t)@initial)@(expm(model['A']*t)@initial)) for t in np.linspace(0, .1, 7)]
    assert np.all(np.diff(energy) < 0)


@pytest.mark.parametrize('contact', [None, 0.])
def test_none_and_empty_losses_preserve_original_samples(contact):
    kwargs = dict(modes=8, peak=.01, palm_resistance=contact, mechanical_band=3000.)
    base = modal_string(110., 4000, 16000, **kwargs)
    np.testing.assert_array_equal(base, modal_string(110., 4000, 16000, partial_t60s=(), **kwargs))
    np.testing.assert_array_equal(base, modal_string(110., 4000, 16000, partial_t60s=(None,)*8, **kwargs))
    changed = modal_string(110., 4000, 16000, partial_t60s=(None, .2), **kwargs)
    assert np.max(abs(changed-base)) > 1e-6


@pytest.mark.parametrize('value', [None, 1., '1', [0.], [-1.], [np.nan], [np.inf], [True], [False],
                                  [np.bool_(True)], [[1.]], [1.]*81, [1e-320]])
def test_bad_losses_rejected_by_public_entrypoints(value):
    with pytest.raises(ValueError, match='partial_t60s'):
        ModalString(partial_t60s=value)
    with pytest.raises(ValueError, match='partial_t60s'):
        modal_string(110., 4000, 16000, partial_t60s=value)


def test_constructor_and_note_overrides_and_json_are_consistent(tmp_path):
    inst = ModalString(modes=8, bridge_mobility=0, pick_noise=0, partial_t60s=[6., None, .3])
    assert inst.partial_t60s == (6., None, .3)
    assert build_instrument(inst.to_spec()).partial_t60s == inst.partial_t60s
    voice = Voice(110., 45., .25, .7, {'partial_t60s': [None, .2]})
    ctx = RenderContext(16000, 120)
    a = inst.voice(voice, ctx, np.random.default_rng(5))
    other = ModalString(modes=8, bridge_mobility=0, pick_noise=0, partial_t60s=(None, .2))
    b = other.voice(Voice(110., 45., .25, .7, {}), ctx, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    reset = inst.voice(Voice(110., 45., .25, .7, {'partial_t60s': []}), ctx, np.random.default_rng(5))
    plain = ModalString(modes=8, bridge_mobility=0, pick_noise=0)
    np.testing.assert_array_equal(reset, plain.voice(Voice(110., 45., .25, .7, {}),
                                                     ctx, np.random.default_rng(5)))
    song = Song(Track('g', inst)(Note('A2', dur=.5).param(partial_t60s=[None, .2])))
    score = song.compile()
    path = tmp_path/'score.json'
    path.write_text(json.dumps(score.to_dict()), encoding='utf-8')
    restored = Score.from_json(path)
    assert list(restored.events[0].param('partial_t60s')) == [None, .2]


def test_unavailable_upper_coordinates_are_ignored_and_not_reassigned():
    a = _mechanics(110., modes=8, partial_t60s=(None,)*8+(1.,))
    b = _mechanics(110., modes=8)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
