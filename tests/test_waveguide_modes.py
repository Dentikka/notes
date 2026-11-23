"""Selective waveguide losses: closed-form evolution, poles, legacy and DSL contracts."""
import numpy as np
import pytest
from notes import ElectricGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice, render_score
from notes.engine.dsp import feedback_delay
from notes.engine.string_model import _loop, steel_string
from notes.engine.waveguide_modes import decay_correction, loop_modes
from notes.ir import Score


@pytest.mark.parametrize('sr,f,stiffness', [(22050, 82.4, 4e-5), (22050, 659.3, 4e-5),
                                        (44100, 82.4, 4e-5), (44100, 220., 4e-5),
                                        (22050, 220., 0.), (22050, 659.3, .001)])
def test_newton_modes_match_independent_full_polynomial_roots(sr, f, stiffness):
    length, b, a = _loop(f, sr, 12., 3., stiffness, 'bracketed')
    den = np.zeros(length+max(len(a), len(b)))
    den[:len(a)] += a
    den[length:length+len(b)] -= b
    dense = np.roots(np.trim_zeros(den, 'b'))
    poles, factors = loop_modes(length, tuple(b), tuple(a), 12)
    assert len(poles) == 12 and np.all(abs(poles) < 1)
    assert np.max(np.min(abs(poles[:, None]-dense), axis=1)) < 1e-10
    burst = np.random.default_rng(17).normal(size=length)
    # Independent derivative of the complete denominator, in negative powers.
    q = 1/poles
    numerator = np.convolve(burst, a)
    derivative = np.arange(1, len(den))*den[1:]
    expected = -poles*np.polynomial.polynomial.polyval(q, numerator)/np.polynomial.polynomial.polyval(q, derivative)
    got = np.polynomial.polynomial.polyval(q, burst)*factors
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=1e-8)
    assert not poles.flags.writeable and not factors.flags.writeable


@pytest.mark.parametrize('scale', [.2, 3.])
def test_one_mode_has_exact_requested_decay_and_unchanged_frequency_phase(scale):
    length, sr, t60 = 32, 8000, .5
    rate = np.log(1000)/(sr*t60)
    phase, h, n = .4, 3, 16000
    at = np.arange(length)
    burst = np.exp(-rate*at)*np.cos(2*np.pi*h*at/length+phase)
    b, a = np.array([np.exp(-rate*length)]), np.array([1.])
    impulse = np.zeros(n)
    impulse[:length] = burst
    before = feedback_delay(impulse, length, 1., b, a)
    scales = (1., 1., scale)
    delta = decay_correction(burst, length, b, a, scales, n)
    time = np.arange(n)
    expected = np.exp(-rate*time/scale)*np.cos(2*np.pi*h*time/length+phase)
    np.testing.assert_allclose(before+delta, expected, atol=2e-10, rtol=2e-9)
    assert delta[0] == 0


@pytest.mark.parametrize('sampling', ['grid', 'spectral'])
def test_empty_unity_and_unavailable_scales_preserve_waveform_and_rng(sampling):
    streams = [np.random.default_rng(17) for _ in range(3)]
    waves = [steel_string(220., 6000, 22050, rng, pluck_sampling=sampling, partial_t60_scales=scales)
             for rng, scales in zip(streams, [(), (1.,)*12, (1.,)*80], strict=True)]
    for wave in waves[1:]:
        np.testing.assert_array_equal(wave, waves[0])
    assert streams[0].bit_generator.state == streams[1].bit_generator.state == streams[2].bit_generator.state
    # A short loop has fewer than 80 oscillatory modes; out-of-band edits do nothing.
    args = (659.3, 4000, 22050)
    a = steel_string(*args, np.random.default_rng(5), pluck_sampling=sampling)
    b = steel_string(*args, np.random.default_rng(5), pluck_sampling=sampling,
                     partial_t60_scales=(1.,)*79+(.2,))
    np.testing.assert_array_equal(a, b)


def test_changed_losses_keep_random_draws_and_render_through_json_with_note_reset():
    ctx = RenderContext(22050, 60)
    instrument = ElectricGuitar(amplified=False, partial_t60_scales=(1., .2, .5))
    original = ElectricGuitar(amplified=False)
    v = Voice(220., 57., .2, .7, {})
    a, b = np.random.default_rng(11), np.random.default_rng(11)
    changed = instrument.voice(v, ctx, a)
    before = original.voice(v, ctx, b)
    assert a.bit_generator.state == b.bit_generator.state
    assert np.isfinite(changed).all() and not np.array_equal(changed, before)
    override = Voice(220., 57., .2, .7, {'partial_t60_scales': (1., .2, .5)})
    np.testing.assert_array_equal(changed, original.voice(override, ctx, np.random.default_rng(11)))
    reset = Voice(220., 57., .2, .7, {'partial_t60_scales': []})
    np.testing.assert_array_equal(before, instrument.voice(reset, ctx, np.random.default_rng(11)))
    score = Song(Track('g', instrument)(Note(57, .2, .7)), sample_rate=22050, bpm=60).compile()
    np.testing.assert_array_equal(render_score(score).audio, render_score(Score.from_json(score.to_json())).audio)


@pytest.mark.parametrize('scales', [None, 'bad', (0.,), (-1.,), (np.inf,), (np.nan,), (True,), [1.]*81])
def test_invalid_scales_rejected_before_source_and_note_render(scales):
    with pytest.raises(ValueError, match='partial_t60_scales'):
        steel_string(220., 0, 22050, np.random.default_rng(1), partial_t60_scales=scales)
    with pytest.raises(ValueError, match='partial_t60_scales'):
        ElectricGuitar(partial_t60_scales=scales)
    with pytest.raises(ValueError, match='partial_t60_scales'):
        ElectricGuitar().voice(Voice(220., 57., .1, .7, {'partial_t60_scales': scales}),
                              RenderContext(22050, 60), np.random.default_rng(1))


def test_noncontractive_loop_is_rejected_instead_of_editing_unstable_modes():
    with pytest.raises(ValueError, match='contractive'):
        loop_modes(32, (1.001,), (1.,), 3)
