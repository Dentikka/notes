"""Modal readout must not alter contact mechanics, tuning, defaults or IR controls."""
from dataclasses import replace

import numpy as np
import pytest

from notes import Note, Piano, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.hammer import struck_string
from notes.ir import Score


@pytest.mark.parametrize('sr', [8000, 22050, 44100])
@pytest.mark.parametrize('freq', [110., 440., 1000.])
@pytest.mark.parametrize('radiation', [False, True])
def test_empty_and_unit_readouts_keep_contact_and_ring_bit_exact(sr, freq, radiation):
    base = struck_string(freq, round(sr*.3), sr, .65, radiation=radiation)
    unit = ((0., *([1.]*64)), (.1, *([1.]*64)))
    actual = struck_string(freq, len(base), sr, .65, radiation=radiation, partial_gain_points=unit)
    assert np.array_equal(actual, base)


def test_static_readout_superposition_includes_contact_without_changing_excitation():
    options = {'freq': 440., 'n': 22050, 'sr': 22050, 'velocity': .7, 'radiation': True}
    base = struck_string(**options)
    zero = struck_string(**options, partial_gain_points=((0., 0.),))
    double = struck_string(**options, partial_gain_points=((0., 2.),))
    # If a gain changed coupling/force instead of readout, this linear identity
    # would fail during and after the nonlinear hammer contact.
    np.testing.assert_allclose(zero+double, 2*base, atol=2e-13, rtol=1e-11)
    silent = struck_string(**options, partial_gain_points=((0., *([0.]*64)),))
    assert not np.any(silent)


def test_dynamic_gain_reads_both_unison_components_on_the_output_clock():
    sr = 22050
    options = {'freq': 220., 'n': sr, 'sr': sr, 'velocity': .7, 'radiation': True, 'detune': 6.}
    base = struck_string(**options)
    first = struck_string(**options, partial_gain_points=((0., 1., *([0.]*63)),))
    curve = ((0., .25), (.2, .25), (.5, 2.))
    actual = struck_string(**options, partial_gain_points=curve)
    gain = np.interp(np.arange(sr)/sr, [r[0] for r in curve], [r[1] for r in curve])
    expected = base+(gain-1)*first
    np.testing.assert_allclose(actual[round(.04*sr):], expected[round(.04*sr):], atol=2e-13, rtol=1e-11)
    assert np.linalg.norm(actual-base) > .01


@pytest.mark.parametrize('model', ['hammer', 'hammer_damped'])
def test_constructor_note_override_and_score_roundtrip(model):
    points = ((0., 1., .25), (.4, .5, 2.))
    instrument = Piano(model=model, partial_gain_points=points)
    assert build_instrument(instrument.to_spec()) == instrument
    voice = Voice(440., 69., .5, .65, {})
    context = RenderContext(22050, 60)
    def render(inst, params):
        return inst.voice(replace(voice, params=params), context, np.random.default_rng(17))
    original = Piano(model=model)
    assert np.array_equal(render(instrument, {}), render(original, {'partial_gain_points': points}))
    assert np.array_equal(render(instrument, {'partial_gain_points': []}), render(original, {}))
    score = Song(Track('piano', original)(Note(69, .5).param(partial_gain_points=points, pedal=True)),
                 bpm=60, sample_rate=8000, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param('partial_gain_points')
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize('model', ['legacy', 'soundboard'])
def test_unsupported_models_reject_explicit_curves(model):
    with pytest.raises(ValueError, match='partial_gain_points'):
        Piano(model=model, partial_gain_points=((0., 2.),))
    with pytest.raises(ValueError, match='partial_gain_points'):
        Piano(model=model).voice(Voice(440., 69., .5, .7, {'partial_gain_points': ((0., 2.),)}),
                                 RenderContext(8000, 60), np.random.default_rng(1))


@pytest.mark.parametrize('points', [None, [[0.]], [[0., -1.]], [[0., np.nan]], [[0., 1.], [0., 2.]]])
def test_invalid_direct_kernel_and_instrument_curves_are_rejected(points):
    with pytest.raises(ValueError, match='partial_gain_points'):
        struck_string(440., 100, 22050, .7, partial_gain_points=points)
    with pytest.raises(ValueError, match='partial_gain_points'):
        Piano(model='hammer_damped', partial_gain_points=points)


def test_default_piano_keeps_the_earlier_renderer_call_contract(monkeypatch):
    import notes.engine.hammer as hammer
    original = hammer.struck_string
    calls = []
    def historical(freq, n, sr, velocity, *, brightness, stiffness, sustain, detune, radiation,
                   attack_presence, hammer_mass_scale):
        calls.append(True)
        return original(freq, n, sr, velocity, brightness=brightness, stiffness=stiffness, sustain=sustain,
                        detune=detune, radiation=radiation, attack_presence=attack_presence,
                        hammer_mass_scale=hammer_mass_scale)
    monkeypatch.setattr(hammer, 'struck_string', historical)
    audio = Piano(model='hammer_damped').voice(Voice(440., 69., .2, .65, {}), RenderContext(8000, 60),
                                               np.random.default_rng(17))
    assert calls and np.isfinite(audio).all()
