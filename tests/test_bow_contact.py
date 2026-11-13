"""Independent equations, work budget and real renderer integration of modal bowing."""
import numpy as np
import pytest
from scipy.linalg import solve
from scipy.optimize import brentq

from notes import ModalBowedString, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.bow_contact import _model, _soft_contact, modal_bowed_string
from notes.engine.dsp import vel_gain
from notes.ir import Score


def test_soft_contact_matches_an_independent_root_and_is_dissipative():
    a, normal, mobility = 100., 1.2, .04
    def phi(u):
        return np.sqrt(2*a)*u*np.exp(.5-a*u*u)
    for free in np.r_[np.linspace(-2., 2., 61), -20., 0., 20.]:
        force, relative = _soft_contact(float(free), mobility, normal, a)
        expected = brentq(lambda u, free=free: u+mobility*normal*phi(u)-free,
                          min(0., free), max(0., free), xtol=1e-14)
        assert relative == pytest.approx(expected, abs=1e-12)
        assert force == pytest.approx(normal*phi(expected), abs=1e-10)
        assert force*relative >= -1e-14


def test_modal_update_matches_dense_equations_and_the_work_identity():
    freq, sr, factor, modes = 110., 8000, 16, 8
    B, tension, length = 1e-5, 110., .69
    model = _model(freq, sr, factor, modes, 5000., B, tension, length, 12., 1., .08, .012)
    h = np.arange(1, modes+1)
    hz = freq*h*np.sqrt((1+B*h*h)/(1+B))
    omega = 2*sr*factor*np.tan(np.pi*hz/(sr*factor))
    zero = np.zeros((modes, modes))
    A = np.block([[zero, np.diag(omega)], [-np.diag(omega), -2*np.diag(model.gamma)]])
    forcing = np.r_[np.zeros(modes), model.coupling]
    initial = np.random.default_rng(17).normal(size=2*modes)*.01
    force = .8
    identity = np.eye(2*modes)
    expected = solve(identity-model.dt*A/2, (identity+model.dt*A/2)@initial+model.dt*force*forcing)
    x, v = initial[:modes], initial[modes:]
    xn = model.p11*x+model.p12*v+model.qx*force
    vn = model.p21*x+model.p22*v+model.qv*force
    np.testing.assert_allclose(np.r_[xn, vn], expected, atol=1e-16)
    midpoint = (v+vn)/2
    energy_change = .5*(xn@xn+vn@vn-initial@initial)
    damping = model.dt*np.sum(2*model.gamma*midpoint**2)
    assert energy_change+damping == pytest.approx(model.dt*force*(model.coupling@midpoint), abs=1e-16)
    assert not model.coupling.flags.writeable


def test_work_budget_stopped_bow_and_duration_prefix():
    options = dict(modes=16, bandwidth=4000., oversample=8, attack=.05, gate=.2, audit=True)
    short, a = modal_bowed_string(110., 3200, 8000, **options)
    long, b = modal_bowed_string(110., 6400, 8000, **options)
    np.testing.assert_array_equal(short[:-64], long[:len(short)-64])
    for stats in (a, b):
        assert stats['uniqueness_margin'] > 0
        assert stats['maximum_energy_balance_error'] < 1e-10
        assert stats['friction_loss'] >= 0 and stats['damping_loss'] >= 0
        assert stats['final_energy'] == pytest.approx(
            stats['bow_work']-stats['friction_loss']-stats['damping_loss'], abs=1e-9)
    assert b['final_energy'] <= a['final_energy']
    assert np.isfinite(long).all() and np.max(abs(long)) > .001


def test_silence_empty_and_automatic_unique_root_resolution():
    for options in ({'bow_speed': 0.}, {'force_ratio': 0.}, {'gate': 0.}):
        audio, _ = modal_bowed_string(110., 800, 8000, modes=16, bandwidth=4000., **options)
        assert not audio.any()
    assert len(modal_bowed_string(110., 0, 8000, modes=16, bandwidth=4000.)[0]) == 0
    audio, stats = modal_bowed_string(110., 800, 8000, modes=16, bandwidth=4000., oversample=1)
    assert stats['oversample'] >= 2 and stats['uniqueness_margin'] > 0
    assert stats['bow_work'] is None
    assert np.isfinite(audio).all()


def test_public_voice_note_controls_velocity_seed_and_ir_roundtrip():
    instrument = ModalBowedString(modes=16, bandwidth=4000., oversample=8, attack=.05)
    assert build_instrument(instrument.to_spec()) == instrument
    ctx = RenderContext(8000, 60)
    def voice(velocity=.7, params=None, seed=17):
        return instrument.voice(Voice(110., 45., .3, velocity, params or {}), ctx, np.random.default_rng(seed))
    y = voice()
    np.testing.assert_array_equal(y, voice(seed=29))
    np.testing.assert_allclose(voice(.3), y*vel_gain(.3)/vel_gain(.7), atol=1e-15)
    assert not voice(0).any() and not voice(params={'force_ratio': 0.}).any()
    assert y[-1] == 0 and np.max(abs(y)) > 1e-4
    assert not np.allclose(y, voice(params={'open_pitch': 36.}))
    with pytest.raises(ValueError, match='open_pitch'):
        voice(params={'open_pitch': 57.})
    with pytest.raises(ValueError, match='steady pitches'):
        voice(params={'bend': .5})
    offset = voice(params={'pitch_points': ((0., .25),)})
    direct = instrument.voice(Voice(110.*2**(.25/12), 45., .3, .7, {}), ctx, np.random.default_rng(17))
    np.testing.assert_array_equal(offset, direct)
    score = Song(Track('bow', instrument)(Note('A2', dur=.3).param(bow_position=.09)),
                 bpm=60, sample_rate=8000, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    np.testing.assert_array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize('options', [
    {'modes': 4}, {'modes': True}, {'oversample': 3}, {'oversample': 16.},
    {'bandwidth': 0}, {'position': 1.}, {'width': .5}, {'soft_a': 0}, {'bow_speed': -.1},
    {'force_ratio': float('nan')}, {'stiffness': -.001}, {'pressure_attack': 0}, {'gate': 2.},
])
def test_invalid_physical_controls_are_rejected(options):
    with pytest.raises(ValueError):
        modal_bowed_string(110., 800, 8000, **options)


@pytest.mark.parametrize('options', [
    {'open_pitches': ()}, {'open_pitches': (57., 36.)}, {'open_pitches': (float('nan'),)},
    {'body_size': 0}, {'bow_width': .5}, {'oversample': 16.}, {'pressure_attack': 0}, {'level': -.1},
])
def test_invalid_instrument_controls_are_rejected(options):
    with pytest.raises(ValueError):
        ModalBowedString(**options)
