"""Continuous pluck coordinates, physical spectral zeros and public rendering."""
import numpy as np
import pytest
from notes import ElectricGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice, render_score
from notes.engine.string_model import _pluck, _spectral_shape, steel_string
from notes.ir import Score


@pytest.mark.parametrize('length', [257, 258])
def test_continuous_coordinate_retains_physical_zeros_without_grid_zeros(length):
    # A real 1/6 coordinate has a sixth-bin zero at either period length.
    physical = abs(np.fft.rfft(_spectral_shape(length, 1/6, 'velocity')))
    assert physical[6]/physical[1] < 1e-12
    position = .15*2**(2/12)
    smooth = abs(np.fft.rfft(_spectral_shape(length, position, 'velocity')))
    assert smooth[6]/smooth[1] > .005
    if length == 258:
        grid = abs(np.fft.rfft(_pluck(length, 44100, position, .7, 0., np.random.default_rng(1))))
        assert grid[6]/grid[1] < 1e-12


def test_motion_across_a_grid_boundary_is_continuous():
    boundary = 43.5/258
    pairs = {}
    for mode in ['grid', 'spectral']:
        a, b = [_pluck(258, 44100, boundary+offset, .7, 0., np.random.default_rng(1), sampling=mode)
                for offset in [-1e-8, 1e-8]]
        pairs[mode] = np.linalg.norm(a-b)/np.linalg.norm(a)
    assert pairs['grid'] > .01
    assert pairs['spectral'] < 1e-4


@pytest.mark.parametrize('excitation', ['velocity', 'displacement'])
def test_underresolved_spectral_period_is_rejected_instead_of_normalizing_zero(excitation):
    with pytest.raises(ValueError, match='at least three delay samples'):
        _spectral_shape(2, .5, excitation)


@pytest.mark.parametrize('length', [127, 128])
def test_displacement_and_velocity_satisfy_the_fourier_derivative(length):
    displacement = np.fft.rfft(_spectral_shape(length, .173, 'displacement'))
    velocity = np.fft.rfft(_spectral_shape(length, .173, 'velocity'))
    # The self-conjugate Nyquist bin cannot carry the derivative's imaginary part.
    h = np.arange(1, (length+1)//2)
    np.testing.assert_allclose(velocity[h], 2j*np.pi*h*displacement[h], atol=1e-12, rtol=1e-12)
    assert abs(displacement[0]-length*.5) < 1e-12


def test_optional_sampling_preserves_random_draws_and_legacy_default():
    rng = [np.random.default_rng(29) for _ in range(3)]
    a = steel_string(220., 4000, 22050, rng[0])
    b = steel_string(220., 4000, 22050, rng[1], pluck_sampling='grid')
    c = steel_string(220., 4000, 22050, rng[2], pluck_sampling='spectral')
    np.testing.assert_array_equal(a, b)
    assert rng[0].bit_generator.state == rng[1].bit_generator.state == rng[2].bit_generator.state
    assert np.isfinite(c).all() and not np.array_equal(a, c)


def test_constructor_note_override_and_json_renderer_agree():
    instrument = ElectricGuitar(amplified=False, pluck_sampling='spectral')
    old = ElectricGuitar(amplified=False)
    ctx = RenderContext(22050, 60)
    v = Voice(165., 52., .15, .7, {})
    overridden = Voice(v.freq, v.pitch, v.dur, v.vel, {'pluck_sampling': 'spectral'})
    np.testing.assert_array_equal(instrument.voice(v, ctx, np.random.default_rng(9)),
                                  old.voice(overridden, ctx, np.random.default_rng(9)))
    reset = Voice(v.freq, v.pitch, v.dur, v.vel, {'pluck_sampling': 'grid'})
    np.testing.assert_array_equal(instrument.voice(reset, ctx, np.random.default_rng(9)),
                                  old.voice(v, ctx, np.random.default_rng(9)))
    score = Song(Track('g', instrument)(Note(52, .15, .7)), sample_rate=22050, bpm=60).compile()
    np.testing.assert_array_equal(render_score(score).audio, render_score(Score.from_json(score.to_json())).audio)


@pytest.mark.parametrize('sampling', ['bad', None, True])
def test_invalid_sampling_rejected_at_all_public_entries(sampling):
    with pytest.raises(ValueError, match='pluck_sampling'):
        steel_string(220., 0, 22050, np.random.default_rng(1), pluck_sampling=sampling)
    with pytest.raises(ValueError, match='pluck_sampling'):
        ElectricGuitar(pluck_sampling=sampling)
    with pytest.raises(ValueError, match='pluck_sampling'):
        ElectricGuitar().voice(Voice(220., 57., .1, .7, {'pluck_sampling': sampling}),
                              RenderContext(22050, 60), np.random.default_rng(1))
