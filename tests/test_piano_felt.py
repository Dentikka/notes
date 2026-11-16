from dataclasses import replace

import numpy as np
import pytest

from notes import Note, Piano, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.hammer import struck_string
from notes.ir import Score


def test_explicit_identity_controls_preserve_contact_output():
    for pitch in (36, 69, 108):
        options = dict(freq=440*2**((pitch-69)/12), n=4410, sr=22050, velocity=.7, radiation=True)
        assert np.array_equal(struck_string(**options), struck_string(**options,
            felt_stiffness_scale=1., contact_oversample=1, radiation_compensation=0.))


def test_radiation_compensation_is_only_one_state_derived_scalar():
    options = dict(freq=440., n=11025, sr=22050, velocity=.65, radiation=True)
    base = struck_string(**options)
    half = struck_string(**options, radiation_compensation=.5)
    full = struck_string(**options, radiation_compensation=1.)
    gain = float(base@full/(base@base))
    assert gain > 1.
    np.testing.assert_allclose(full, base*gain, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(half, base*np.sqrt(gain), rtol=2e-13, atol=2e-14)
    # This includes the integrated head, padded boundary and free tail. A change
    # of modal ratios or contact mechanics would fail the proportional identity.
    low = dict(freq=110., n=4410, sr=22050, velocity=.65, radiation=True)
    assert np.array_equal(struck_string(**low), struck_string(**low, radiation_compensation=1.))


def test_harder_felt_can_excitate_upper_key_without_output_gain():
    options = dict(freq=440*2**((108-69)/12), n=15435, sr=44100, velocity=.65, radiation=True)
    soft = struck_string(**options)
    hard = struck_string(**options, felt_stiffness_scale=16., contact_oversample=4)
    sl = slice(1764, 4410)
    ratio = np.sqrt(np.mean(hard[sl]**2)/np.mean(soft[sl]**2))
    assert np.isfinite(hard).all() and ratio > 10


def test_contact_hardness_shortens_impact_and_audit_does_not_change_sound():
    options = dict(freq=440*2**((108-69)/12), n=15435, sr=44100, velocity=.65, radiation=True)
    soft, hard = {}, {}
    y = struck_string(**options, contact_oversample=8, _audit=soft)
    assert np.array_equal(y, struck_string(**options, contact_oversample=8))
    struck_string(**options, felt_stiffness_scale=16., contact_oversample=8, _audit=hard)
    assert soft['separated'] == hard['separated'] == 1.
    assert hard['contact_seconds'] < soft['contact_seconds']
    assert hard['initial_energy'] == soft['initial_energy']
    for state in (soft, hard):
        assert 0 < state['string_energy']+state['hammer_energy'] <= 1.02*state['initial_energy']


@pytest.mark.parametrize('pitch', [84, 108])
def test_hard_felt_level_converges_under_fixed_mode_band_refinement(pitch):
    options = dict(freq=440*2**((pitch-69)/12), n=15435, sr=44100, velocity=.65,
                   radiation=True, felt_stiffness_scale=16.)
    waves = [struck_string(**options, contact_oversample=factor) for factor in (4, 8)]
    rms = [np.sqrt(np.mean(w[1764:11025]**2)) for w in waves]
    assert abs(20*np.log10(rms[0]/rms[1])) < .2


def test_constructor_note_reset_and_dsl_ir_roundtrip():
    p = Piano(model='hammer_damped', felt_stiffness_scale=4., contact_oversample=2,
              radiation_compensation=.7)
    plain = Piano(model='hammer_damped')
    assert build_instrument(p.to_spec()) == p
    context = RenderContext(8000, 60)
    v = Voice(440., 69., .2, .7, {})
    params = {'felt_stiffness_scale': 4., 'contact_oversample': 2, 'radiation_compensation': .7}
    a = p.voice(v, context, np.random.default_rng(1))
    b = plain.voice(replace(v, params=params), context, np.random.default_rng(1))
    assert np.array_equal(a, b)
    reset = {'felt_stiffness_scale': 1., 'contact_oversample': 1, 'radiation_compensation': 0.}
    assert np.array_equal(p.voice(replace(v, params=reset), context, np.random.default_rng(1)),
                          plain.voice(v, context, np.random.default_rng(1)))
    score = Song(Track('piano', plain)(Note(69, .2).param(**params)),
                 sample_rate=8000, bpm=60, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize('kwargs', [
    {'felt_stiffness_scale': .24}, {'felt_stiffness_scale': 16.1}, {'felt_stiffness_scale': float('nan')},
    {'contact_oversample': 0}, {'contact_oversample': 17}, {'contact_oversample': 2.}, {'contact_oversample': True},
    {'radiation_compensation': -.1}, {'radiation_compensation': 1.1}, {'radiation_compensation': float('inf')},
])
def test_invalid_controls_rejected_at_source_constructor_and_note(kwargs):
    with pytest.raises(ValueError):
        struck_string(440., 800, 8000, .7, radiation=True, **kwargs)
    with pytest.raises(ValueError):
        Piano(model='hammer_damped', **kwargs)
    with pytest.raises(ValueError):
        Piano(model='hammer_damped').voice(Voice(440., 69., .1, .7, kwargs),
                                           RenderContext(8000, 60), np.random.default_rng(0))


@pytest.mark.parametrize('model', ['legacy', 'soundboard'])
@pytest.mark.parametrize('params', [{'felt_stiffness_scale': 2.}, {'contact_oversample': 2},
                                    {'radiation_compensation': .5}])
def test_unsupported_models_reject_explicit_changes(model, params):
    with pytest.raises(ValueError):
        Piano(model=model, **params)
    with pytest.raises(ValueError):
        Piano(model=model).voice(Voice(440., 69., .1, .7, params), RenderContext(8000, 60), np.random.default_rng(0))


def test_plain_hammer_does_not_silently_ignore_radiation_compensation():
    with pytest.raises(ValueError):
        struck_string(440., 800, 8000, .7, radiation_compensation=.5)
    with pytest.raises(ValueError):
        Piano(model='hammer', radiation_compensation=.5)
