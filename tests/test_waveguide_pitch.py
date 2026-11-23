"""Static modal retuning: closed-form phase continuity, losses, guardrails and DSL."""
import numpy as np
import pytest
from notes import ElectricGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice, render_score
from notes.engine.dsp import feedback_delay
from notes.engine.string_model import steel_string
from notes.engine.waveguide_modes import decay_correction, mode_correction
from notes.ir import Score


@pytest.mark.parametrize('cents,scale', [(7., 1.), (-9., 1.), (7., .2), (-9., 3.)])
def test_retuning_and_loss_change_follow_one_continuous_closed_form_mode(cents, scale):
    length, sr, n = 32, 8000, 16000
    rate = np.log(1000)/(sr*.5)
    h, phase = 3, .4
    omega = 2*np.pi*h/length
    at = np.arange(length)
    burst = np.exp(-rate*at)*np.cos(omega*at+phase)
    b, a = np.array([np.exp(-rate*length)]), np.array([1.])
    impulse = np.zeros(n)
    impulse[:length] = burst
    before = feedback_delay(impulse, length, 1., b, a)
    delta = mode_correction(burst, length, b, a, (1., 1., scale), n, pitch_cents=(0., 0., cents))
    time = np.arange(n)
    elapsed = np.maximum(time-length, 0)
    angle = omega*np.minimum(time, length)+omega*2**(cents/1200)*elapsed+phase
    expected = np.exp(-rate*(np.minimum(time, length)+elapsed/scale))*np.cos(angle)
    np.testing.assert_allclose(before+delta, expected, atol=3e-10, rtol=3e-9)
    np.testing.assert_array_equal(delta[:length+1], np.zeros(length+1))


@pytest.mark.parametrize('length,index,cents', [(128, 29, 100.), (128, 29, -100.), (64, 30, 100.)])
def test_crossing_an_unedited_neighbor_or_nyquist_is_rejected(length, index, cents):
    burst = np.ones(length)
    offsets = [0.]*(index+1)
    offsets[index] = cents
    with pytest.raises(ValueError, match='cross modes or exceed Nyquist'):
        mode_correction(burst, length, np.array([.9]), np.array([1.]), (), 512, pitch_cents=tuple(offsets))


def test_zero_offsets_keep_the_old_nondefault_decay_path_exactly():
    burst = np.random.default_rng(17).normal(size=32)
    b, a, scales = np.array([.9]), np.array([1.]), (1., .2, 3.)
    before = decay_correction(burst, 32, b, a, scales, 4000)
    after = mode_correction(burst, 32, b, a, scales, 4000, pitch_cents=(0.,)*12)
    assert before.tobytes() == after.tobytes()


def test_frequency_change_keeps_strongly_damped_release_state_bounded():
    length = 32
    burst = np.cos(2*np.pi*np.arange(length)/length)
    delta = mode_correction(burst, length, np.array([.01**length]), np.array([1.]), (), 256,
                            pitch_cents=(3.,))
    np.testing.assert_array_equal(delta[:length+1], np.zeros(length+1))
    assert np.isfinite(delta).all() and abs(delta).max() < 2.


def test_unavailable_offsets_do_not_change_a_short_loop():
    args = (659.3, 4000, 22050)
    a = steel_string(*args, np.random.default_rng(17))
    b = steel_string(*args, np.random.default_rng(17), partial_pitch_cents=(0.,)*79+(50.,))
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize('losses', [(), (1., .2, 3.)])
def test_public_zero_offsets_preserve_samples_and_rng_including_modal_losses(losses):
    a, b = np.random.default_rng(17), np.random.default_rng(17)
    x = steel_string(220., 6000, 22050, a, partial_t60_scales=losses)
    y = steel_string(220., 6000, 22050, b, partial_t60_scales=losses, partial_pitch_cents=(0.,)*80)
    assert x.tobytes() == y.tobytes() and a.bit_generator.state == b.bit_generator.state


def test_constructor_note_override_reset_json_and_unchanged_random_draws():
    inst = ElectricGuitar(amplified=False, partial_pitch_cents=(0., 3., -1.))
    old = ElectricGuitar(amplified=False)
    ctx = RenderContext(22050, 60)
    v = Voice(220., 57., .2, .7, {})
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    changed, before = inst.voice(v, ctx, a), old.voice(v, ctx, b)
    assert a.bit_generator.state == b.bit_generator.state and np.isfinite(changed).all()
    assert not np.array_equal(changed, before)
    override = Voice(220., 57., .2, .7, {'partial_pitch_cents': (0., 3., -1.)})
    np.testing.assert_array_equal(changed, old.voice(override, ctx, np.random.default_rng(9)))
    reset = Voice(220., 57., .2, .7, {'partial_pitch_cents': []})
    np.testing.assert_array_equal(before, inst.voice(reset, ctx, np.random.default_rng(9)))
    score = Song(Track('g', inst)(Note(57, .2, .7)), sample_rate=22050, bpm=60).compile()
    np.testing.assert_array_equal(render_score(score).audio, render_score(Score.from_json(score.to_json())).audio)


@pytest.mark.parametrize('offsets', [None, 'bad', (101.,), (-101.,), (np.nan,), (np.inf,), (True,), [0.]*81])
def test_invalid_offsets_fail_at_all_public_entries(offsets):
    with pytest.raises(ValueError, match='partial_pitch_cents'):
        steel_string(220., 0, 22050, np.random.default_rng(1), partial_pitch_cents=offsets)
    with pytest.raises(ValueError, match='partial_pitch_cents'):
        ElectricGuitar(partial_pitch_cents=offsets)
    with pytest.raises(ValueError, match='partial_pitch_cents'):
        ElectricGuitar().voice(Voice(220., 57., .1, .7, {'partial_pitch_cents': offsets}),
                              RenderContext(22050, 60), np.random.default_rng(1))
