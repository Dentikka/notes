"""Energy, independent dynamics and public IR behavior of the modal string source."""
import numpy as np
import pytest
from scipy.linalg import eigvalsh, expm, solve

from notes import ModalString, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument
from notes.engine.mechanical_string import _initial_state, _output, _tuned_state, modal_string


def test_mechanical_energy_and_independent_matrix_exponential():
    _, model = _tuned_state(110., modes=12, stiffness=.000124)
    for matrix in ('M', 'K', 'C'):
        assert eigvalsh(model[matrix])[0] > 0
        assert not model[matrix].flags.writeable
    assert eigvalsh(model['A']+model['A'].T).max() < 1e-9
    assert model['poles'].real.max() < 0
    initial = _initial_state(model, .15, 6400., release_motion=.75)
    readout = _output(model, .06, .04, 0., True)
    weights = (readout@model['V'])*solve(model['V'], initial)
    energies = []
    for time in np.linspace(0., .1, 11):
        expected = expm(model['A']*time)@initial
        actual = np.sum(weights*np.exp(model['poles']*time)).real
        assert actual == pytest.approx(readout@expected, abs=1e-10)
        energies.append(float(expected@expected))
    assert np.all(np.diff(energies) < 1e-12)


def test_release_motion_redistributes_energy_without_increasing_it():
    _, model = _tuned_state(110., modes=12)
    static = _initial_state(model, .15, None)
    for motion in (-1., -.75, -.25, 0., .25, .75, 1.):
        mixed = _initial_state(model, .15, None, motion)
        assert float(mixed@mixed) == pytest.approx(float(static@static), rel=1e-12)
    moving = _initial_state(model, .15, None, 1.)
    assert not moving[:len(model['omega'])].any()
    np.testing.assert_array_equal(moving, -_initial_state(model, .15, None, -1.))


def test_duration_cannot_change_the_earlier_excitation():
    short = modal_string(110., 16000, 16000, modes=32, release_motion=.75)
    long = modal_string(110., 32000, 16000, modes=32, release_motion=.75)
    np.testing.assert_array_equal(short, long[:len(short)])


@pytest.mark.parametrize('freq', [82.4069, 110., 123.4708, 146.8324, 195.9977, 329.6276, 659.2551])
def test_string_like_damped_mode_stays_in_tune_across_the_guitar(freq):
    _, model = _tuned_state(freq, modes=16)
    measured = model['string_poles'][0].imag/(2*np.pi)
    assert abs(1200*np.log2(measured/freq)) < .001


def test_public_voice_preserves_seed_silence_release_and_ir_roundtrip():
    instrument = ModalString(modes=16, release_motion=.75)
    assert build_instrument(instrument.to_spec()) == instrument
    ctx = RenderContext(16000, 60)
    def voice(params=None, vel=.7, seed=17):
        return instrument.voice(Voice(110., 45., .3, vel, params or {}), ctx, np.random.default_rng(seed))
    a = voice()
    np.testing.assert_array_equal(a, voice())
    assert np.isfinite(a).all() and np.max(abs(a)) > 1e-4
    assert not voice(vel=0).any()
    assert not np.array_equal(a[:96], voice(seed=29)[:96])
    np.testing.assert_array_equal(a[96:], voice(seed=29)[96:])
    assert np.max(abs(a[-10:])) < np.max(abs(a))*.01
    assert not np.allclose(a, voice({'release_motion': 0.}))
    assert not np.allclose(a, voice({'palm_mute': True}))
    legato = voice({'legato': True})
    np.testing.assert_array_equal(legato, voice({'legato': True}, seed=29))
    score = Song(Track('guitar', instrument)(Note('A2', dur=.2).param(release_motion=-1., bridge_scale=.5)),
                 bpm=60, sample_rate=16000).compile()
    assert score.events[0].param('release_motion') == -1.
    # Exercise the actual registered source in the normal renderer.
    from notes.engine import render_score
    result = render_score(score)
    assert np.isfinite(result.audio).all() and result.stats['limiter_db'] == 0


def test_steady_pitch_offset_and_time_varying_curve_are_explicit():
    instrument = ModalString(modes=16, pick_noise=0)
    ctx = RenderContext(16000, 60)
    shifted = instrument.voice(Voice(110., 45., .2, .7, {'pitch_points': ((0., .25),)}),
                               ctx, np.random.default_rng(1))
    direct = instrument.voice(Voice(110.*2**(.25/12), 45., .2, .7, {}), ctx, np.random.default_rng(1))
    np.testing.assert_array_equal(shifted, direct)
    with pytest.raises(ValueError, match='steady pitches'):
        instrument.voice(Voice(110., 45., .2, .7, {'bend': .5}), ctx, np.random.default_rng(1))


@pytest.mark.parametrize('options', [{'modes': 4}, {'stiffness': -1}, {'bridge_q': 0},
                                    {'pickup_position': 0}, {'pickup_width': .5}, {'release_motion': 2},
                                    {'level': np.nan}, {'brightness': np.inf}])
def test_invalid_instrument_controls_are_rejected(options):
    with pytest.raises(ValueError):
        ModalString(**options)


@pytest.mark.parametrize('params', [{'release_motion': 2.}, {'bridge_scale': -1.},
                                   {'pick_hardness': np.nan}, {'pick_position': 1.}, {'stiffness': -1.}])
def test_invalid_per_note_controls_are_rejected(params):
    with pytest.raises(ValueError):
        ModalString(modes=16).voice(Voice(110., 45., .2, .7, params), RenderContext(16000, 60),
                                    np.random.default_rng(1))
