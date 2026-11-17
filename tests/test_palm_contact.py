"""Independent mechanical, numerical-observation and public voice contracts."""
from dataclasses import replace

import numpy as np
import pytest
from notes import ModalString, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.mechanical_string import _initial_state, _output, _point, modal_string
from notes.engine.palm_contact import _sample, _state, palm_string
from notes.ir import Score
from scipy.linalg import eig, expm, solve


def model(*, resistance=.03, width=.06, peak=0., position=.11):
    return _state(146.8324, 12, 4e-5, 125., 20., peak, 70., .5, 30., 2.5,
                  position, width, resistance)


@pytest.mark.parametrize('peak', [0., .01])
def test_contact_is_passive_and_matches_independent_physical_coordinate_dynamics(peak):
    m = model(peak=peak)
    for key in ('M', 'K', 'C'):
        assert np.linalg.eigvalsh(m[key]).min() > 0
        assert not m[key].flags.writeable
    assert np.linalg.eigvalsh(m['A']+m['A'].T).max() < 1e-9
    initial = _initial_state(m, .2, 3000.)
    q0, v0 = m['U']@(initial[:len(m['omega'])]/m['omega']), m['U']@initial[len(m['omega']):]
    size = len(m['M'])
    physical = np.block([[np.zeros((size, size)), np.eye(size)],
                         [-solve(m['M'], m['K']), -solve(m['M'], m['C'])]])
    readout = _point(m['h'], .1, body=peak > 0, relative=True, width=.04)
    sampled = _sample(m, initial, _output(m, .1, .04, 0., True), 2401, 24000, None, 'spatial')
    energies = []
    for k in (0, 120, 480, 1200, 2400):
        state = expm(physical*(k/24000))@np.r_[q0, v0]
        assert sampled[k] == pytest.approx(readout@state[size:], abs=1e-9)
        energies.append((state[size:]@m['M']@state[size:]+state[:size]@m['K']@state[:size])/2)
    assert max(np.diff(energies)) <= energies[0]*1e-10


def test_finite_width_damps_a_mode_with_a_node_at_the_contact_center():
    base = model(position=.25, width=0, resistance=0)
    point = model(position=.25, width=0, resistance=.05)
    strip = model(position=.25, width=.1, resistance=.05)
    assert abs(point['C'][3, 3]-base['C'][3, 3]) < 1e-15
    assert strip['C'][3, 3]-base['C'][3, 3] > .001


def test_overdamped_real_poles_and_release_filter_are_not_discarded():
    m = model(resistance=10., width=0)
    real = np.flatnonzero(m['poles'].imag == 0)
    assert len(real) > 0
    cutoff = float(-m['poles'][real[0]].real/(2*np.pi))
    initial = _initial_state(m, .2, None)
    output = _output(m, .1, .04, 0., True)
    actual = _sample(m, initial, output, 301, 24000, cutoff, 'release')
    c = 2*np.pi*cutoff
    A = np.pad(m['A'], ((0, 2), (0, 2)))
    A[-2, :-2], A[-2, -2], A[-1, -2], A[-1, -1] = c*output, -c, c, -c
    for k in (0, 1, 10, 100, 300):
        expected = (expm(A*k/24000)@np.r_[initial, 0., 0.])[-1]
        assert actual[k] == pytest.approx(expected, abs=1e-9)


def test_defective_modal_basis_falls_back_to_exact_state_evolution():
    A = np.array([[0., 1.], [-1., -2.]])
    poles, vectors = eig(A)
    assert np.linalg.cond(vectors) > 1e10
    m = {'A': A, 'poles': poles, 'V': vectors}
    for rounding in ('spatial', 'release'):
        actual = _sample(m, np.array([1., 0.]), np.array([0., 1.]), 101, 100, 1/(2*np.pi), rounding)
        t = np.arange(101)/100
        expected = -t*np.exp(-t) if rounding == 'spatial' else -t**3*np.exp(-t)/6
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_physical_band_is_independent_of_output_rate_and_prefix_length():
    options = dict(position=.1, contact_width=.06, resistance=.03, modes=64, peak=0.,
                   cutoff=2000., rounding='spatial')
    low = palm_string(146.8324, 2205, 22050, **options)
    high = palm_string(146.8324, 4410, 44100, **options)
    np.testing.assert_allclose(low, high[::2], atol=1e-12)
    short = palm_string(146.8324, 800, 8000, **options)
    long = palm_string(146.8324, 1600, 8000, **options)
    np.testing.assert_allclose(short, long[:len(short)], atol=1e-12)


@pytest.mark.parametrize('kwargs', [dict(resistance=-1), dict(resistance=np.nan), dict(contact_width=-.1),
    dict(position=0), dict(position=.01, contact_width=.1), dict(pickup=2), dict(mechanical_band=100),
    dict(cutoff=0), dict(rounding='unknown')])
def test_invalid_contact_controls_rejected(kwargs):
    options = dict(position=.1, contact_width=.06, resistance=.03, modes=12)
    options.update(kwargs)
    with pytest.raises(ValueError):
        palm_string(146.8324, 100, 22050, **options)


def test_public_voice_contact_controls_roundtrip_and_reach_dsp():
    instrument = ModalString(modes=24, palm_resistance=.03, palm_position=.1, palm_width=.06,
                             pick_cutoff=2000., pluck_rounding='spatial', pick_noise=0, bridge_mobility=0)
    assert build_instrument(instrument.to_spec()) == instrument
    ctx = RenderContext(22050, 60)
    voice = Voice(146.8324, 50., .3, .7, {'palm_mute': True})
    a = instrument.voice(voice, ctx, np.random.default_rng(1))
    b = instrument.voice(replace(voice, params={'palm_mute': True, 'palm_resistance': .06}),
                         ctx, np.random.default_rng(1))
    assert np.isfinite(a).all() and not np.array_equal(a, b)
    assert not instrument.voice(replace(voice, vel=0), ctx, np.random.default_rng(1)).any()
    score = Song(Track('guitar', instrument)(Note('D3', .3).param(palm_mute=True, palm_resistance=.06)),
                 bpm=60, sample_rate=22050, loudness=None).compile()
    first = render_score(score, workers=1).audio
    restored = render_score(Score.from_dict(score.to_dict()), workers=1).audio
    np.testing.assert_array_equal(first, restored)


def test_contact_none_and_zero_are_distinct_and_open_note_is_not_muted():
    ctx = RenderContext(16000, 60)
    base = ModalString(modes=16, pick_noise=0)
    note = Voice(146.8324, 50., .3, .7, {})
    legacy = base.voice(note, ctx, np.random.default_rng(1))
    configured = replace(base, palm_resistance=.03)
    np.testing.assert_array_equal(legacy, configured.voice(note, ctx, np.random.default_rng(1)))
    none = base.voice(replace(note, params={'palm_mute': True}), ctx, np.random.default_rng(1))
    zero = base.voice(replace(note, params={'palm_mute': True, 'palm_resistance': 0}), ctx, np.random.default_rng(1))
    assert not np.array_equal(none, zero)
    with pytest.raises(ValueError, match='palm_mute'):
        configured.voice(replace(note, params={'palm_resistance': .03}), ctx, np.random.default_rng(1))


def test_low_level_legacy_none_preserves_original_path():
    a = modal_string(110., 1600, 16000, modes=16)
    b = modal_string(110., 1600, 16000, modes=16, palm_resistance=None)
    np.testing.assert_array_equal(a, b)
