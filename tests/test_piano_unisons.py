"""Generated unison frequencies, readout independence, legacy contract and IR."""
from dataclasses import replace

import numpy as np
import pytest

from notes import Note, Piano, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.hammer import struck_string
from notes.ir import Score


@pytest.mark.parametrize('count', [1, 2, 3])
def test_symmetric_readout_contains_the_requested_unison_frequencies(count):
    sr, f0, detune = 8000, 440., 12.
    isolated = ((0., 1., *([0.]*63)),)
    y = struck_string(f0, 4*sr, sr, .65, sustain=12., detune=detune,
                       unison_strings=count, partial_gain_points=isolated)
    segment = y[sr:]*np.blackman(3*sr)
    hz = np.fft.rfftfreq(131072, 1/sr)
    power = abs(np.fft.rfft(segment, n=131072))**2
    offsets = np.array([-detune, 0., detune])
    energies = np.array([power[abs(hz-f0*2**(cents/1200)) < .6].sum() for cents in offsets])
    wanted = {1: [1], 2: [0, 2], 3: [0, 1, 2]}[count]
    unwanted = [i for i in range(3) if i not in wanted]
    assert min(energies[wanted]) > max(power)*.1
    if unwanted:
        assert max(energies[unwanted]) < min(energies[wanted])*1e-3
    if len(wanted) > 1:
        assert 10*np.log10(max(energies[wanted])/min(energies[wanted])) < 1.


def test_explicit_unisons_do_not_feed_back_into_hammer_contact():
    options = {'freq': 440., 'n': 22050, 'sr': 22050, 'velocity': .65, 'unison_strings': 3}
    base = struck_string(**options)
    zero = struck_string(**options, partial_gain_points=((0., 0.),))
    double = struck_string(**options, partial_gain_points=((0., 2.),))
    np.testing.assert_allclose(zero+double, 2*base, atol=2e-13, rtol=1e-11)
    legacy = struck_string(440., 22050, 22050, .65)
    # Integrated head is shared; only the post-contact readout differs.
    np.testing.assert_array_equal(base[:16], legacy[:16])
    assert np.linalg.norm(base[2205:]-legacy[2205:]) > .01


def test_constructor_note_override_reset_and_ir_are_consistent():
    inst = Piano(model='hammer_damped', unison_strings=3)
    assert build_instrument(inst.to_spec()) == inst
    v = Voice(440., 69., .3, .65, {})
    ctx = RenderContext(22050, 60)
    def source(instrument, params):
        return instrument.voice(replace(v, params=params), ctx, np.random.default_rng(17))
    legacy = Piano(model='hammer_damped')
    assert np.array_equal(source(inst, {}), source(legacy, {'unison_strings': 3}))
    assert np.array_equal(source(inst, {'unison_strings': None}), source(legacy, {}))
    score = Song(Track('piano', legacy)(Note(69, .3).param(unison_strings=3)),
                 bpm=60, sample_rate=8000, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param('unison_strings') == 3
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize('count', [0, 4, True, 2., '3', [], {}])
def test_invalid_unison_counts_rejected_in_both_entrypoints(count):
    with pytest.raises(ValueError, match='unison_strings'):
        Piano(model='hammer_damped', unison_strings=count)
    with pytest.raises(ValueError, match='unison_strings'):
        struck_string(440., 100, 22050, .65, unison_strings=count)


@pytest.mark.parametrize('model', ['legacy', 'soundboard'])
def test_unsupported_piano_models_reject_explicit_unison_setting(model):
    with pytest.raises(ValueError, match='unison_strings'):
        Piano(model=model, unison_strings=3)


def test_explicit_out_of_band_unisons_fail_instead_of_aliasing():
    with pytest.raises(ValueError, match='unison frequencies'):
        struck_string(440., 8000, 8000, .65, detune=1200., unison_strings=3)
