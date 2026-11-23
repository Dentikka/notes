"""Procedural period shaping, exact legacy defaults and normal DSL rendering."""
import numpy as np
import pytest
from notes import ElectricGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.string_model import _shape_excitation, steel_string
from notes.ir import Score


@pytest.mark.parametrize('length', [63, 64])
def test_excitation_shaping_scales_bin_magnitudes_and_keeps_generated_phases(length):
    burst = np.random.default_rng(4).normal(size=length)
    before = np.fft.rfft(burst)
    shaped = _shape_excitation(burst, (2., .5, 0., 1.3))
    after = np.fft.rfft(shaped)
    expected = before.copy()
    expected[1:5] *= (2., .5, 0., 1.3)
    np.testing.assert_allclose(after, expected, atol=1e-12, rtol=1e-12)
    assert _shape_excitation(burst, ()) is burst
    assert _shape_excitation(burst, (1.,)*80) is burst
    assert _shape_excitation(burst, (1.,)*(length//2)+(7.,)*5) is burst


def test_shaping_has_no_hidden_peak_normalization():
    burst = np.random.default_rng(3).normal(size=32)
    burst -= burst.mean()
    np.testing.assert_allclose(_shape_excitation(burst, (2.,)*16), 2*burst, atol=1e-12)


@pytest.mark.parametrize('freq', [82.4, 329.6, 659.3])
def test_empty_and_unity_controls_preserve_samples_and_rng(freq):
    streams = [np.random.default_rng(17) for _ in range(3)]
    waves = [steel_string(freq, 6000, 22050, rng, excitation_gains=gains)
             for rng, gains in zip(streams, [(), (1.,)*12, (1.,)*80], strict=True)]
    for wave in waves[1:]:
        np.testing.assert_array_equal(wave, waves[0])
    assert streams[0].bit_generator.state == streams[1].bit_generator.state == streams[2].bit_generator.state


def test_generated_basis_is_linear_through_both_polarizations_and_loop_filters():
    def render(gains):
        return steel_string(220., 8000, 22050, np.random.default_rng(7), excitation_gains=gains)
    constant = render((0.,)*4)
    gains = (1., .2, 3., .7)
    expected = constant.copy()
    for i, gain in enumerate(gains):
        unit = [0.]*4
        unit[i] = 1.
        expected += gain*(render(tuple(unit))-constant)
    np.testing.assert_allclose(render(gains), expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('gains', [None, 'bad', (-1.,), (17.,), (np.nan,), (np.inf,), [[1.]], [1.]*81])
def test_invalid_controls_are_rejected_on_source_instrument_and_note(gains):
    with pytest.raises(ValueError, match='excitation_gains'):
        steel_string(220., 0, 22050, np.random.default_rng(1), excitation_gains=gains)
    with pytest.raises(ValueError, match='excitation_gains'):
        ElectricGuitar(excitation_gains=gains)
    with pytest.raises(ValueError, match='excitation_gains'):
        ElectricGuitar().voice(Voice(220., 57., .1, .7, {'excitation_gains': gains}),
                              RenderContext(22050, 60), np.random.default_rng(1))


def test_note_override_reset_and_json_roundtrip_use_the_public_renderer():
    gains = (1., .3, 2., .5)
    inst = ElectricGuitar(amplified=False, excitation_gains=gains, pitch_attack=0.)
    plain = ElectricGuitar(amplified=False, pitch_attack=0.)
    ctx = RenderContext(22050, 60)
    a = inst.voice(Voice(220., 57., .15, .7, {}), ctx, np.random.default_rng(1))
    b = plain.voice(Voice(220., 57., .15, .7, {'excitation_gains': gains}), ctx, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    reset = inst.voice(Voice(220., 57., .15, .7, {'excitation_gains': []}), ctx, np.random.default_rng(1))
    np.testing.assert_array_equal(reset, plain.voice(Voice(220., 57., .15, .7, {}), ctx, np.random.default_rng(1)))
    assert build_instrument(inst.to_spec()) == inst
    score = Song(Track('g', inst)(Note('A3', .15, .7)), sample_rate=22050, bpm=60, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    np.testing.assert_array_equal(render_score(score).audio, render_score(restored).audio)


def test_initial_noise_is_independent_of_pick_click_and_can_be_set_per_note():
    inst = ElectricGuitar(amplified=False, excitation_noise=0., pick_noise=0., pitch_attack=0.)
    voice = Voice(220., 57., .15, .7, {})
    ctx = RenderContext(22050, 60)
    a = inst.voice(voice, ctx, np.random.default_rng(1))
    b = inst.voice(voice, ctx, np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)
    plain = ElectricGuitar(amplified=False, pick_noise=0., pitch_attack=0.)
    override = Voice(220., 57., .15, .7, {'excitation_noise': 0.})
    np.testing.assert_array_equal(a, plain.voice(override, ctx, np.random.default_rng(1)))
    assert build_instrument(inst.to_spec()) == inst


@pytest.mark.parametrize('noise', [-.1, 1.1, np.nan, np.inf, True, None, 'bad'])
def test_initial_noise_validation_on_instrument_and_note(noise):
    with pytest.raises(ValueError, match='excitation_noise'):
        ElectricGuitar(excitation_noise=noise)
    with pytest.raises(ValueError, match='excitation_noise'):
        ElectricGuitar().voice(Voice(220., 57., .1, .7, {'excitation_noise': noise}),
                              RenderContext(22050, 60), np.random.default_rng(1))
