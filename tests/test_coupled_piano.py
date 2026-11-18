"""Physical-coordinate checks, source limits and normal renderer integration."""

from dataclasses import fields

import numpy as np
import pytest
from notes import CoupledPiano, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.piano_unison import _contact, _force_gradient, _model, _Parameters, struck_unisons
from notes.ir import Score
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve
from scipy.signal import resample_poly


def test_discrete_felt_gradient_has_the_actual_potential_work_and_positive_derivative():
    a = np.r_[-0.001, 0.0, 0.001, np.geomspace(1e-8, 0.001, 50)]
    b = np.r_[0.001, 0.001, -0.001, a[3:] * (1 + 1e-7)]
    k = 2.2e9
    force, derivative = _force_gradient(a, b, k)
    expected = k * (np.maximum(b, 0) ** 3.3 - np.maximum(a, 0) ** 3.3) / 3.3
    np.testing.assert_allclose(force * (b - a), expected, rtol=2e-8, atol=1e-20)
    assert np.all(force >= 0) and np.all(derivative >= 0)


def test_moving_endpoint_matrices_match_independent_spatial_energy_and_contact_geometry():
    freq = 220.0
    p = _Parameters(unison_cents=(-2.0, 0.0, 3.0), modes=6, stiffness=0.02, bridge_mass=0.02)
    m = _model(freq, 16000, p)
    rng = np.random.default_rng(7)
    q = rng.normal(size=len(m.M)) * 1e-5
    v = rng.normal(size=len(m.M)) * 0.01
    nodes, weights = np.polynomial.legendre.leggauss(100)
    x = (nodes + 1) / 2
    weights = weights / 2
    h = np.arange(1, 7)
    phi = np.sin(np.pi * x[:, None] * h)
    dx = np.pi * h * np.cos(np.pi * x[:, None] * h)
    dxx = -((np.pi * h) ** 2) * phi
    modal_mass = 0.012 * (220 / freq) ** 0.35 / 3
    kinetic = 0.5 * p.bridge_mass * v[-1] ** 2
    potential = 0.5 * p.bridge_mass * (2 * np.pi * p.bridge_hz) ** 2 * q[-1] ** 2
    contact = []
    for j, cents in enumerate(p.unison_cents):
        sl = slice(j * 6, (j + 1) * 6)
        f = freq * 2 ** (cents / 1200)
        tension = 8 * modal_mass * f * f / (1 + p.stiffness)
        bending = tension * p.stiffness / np.pi**2
        kinetic += 0.5 * 2 * modal_mass * np.sum(weights * (phi @ v[sl] + x * v[-1]) ** 2)
        potential += 0.5 * tension * np.sum(weights * (dx @ q[sl] + q[-1]) ** 2) + 0.5 * bending * np.sum(
            weights * (dxx @ q[sl]) ** 2
        )
        contact.append(float(np.sin(np.pi * h * 0.135) @ q[sl] + 0.135 * q[-1]))
    assert 0.5 * v @ m.M @ v == pytest.approx(kinetic, rel=1e-12)
    assert 0.5 * q @ m.K @ q == pytest.approx(potential, rel=1e-12)
    np.testing.assert_allclose(m.B.T @ q, contact, atol=1e-18)


@pytest.mark.parametrize("body", [None, 0.02])
@pytest.mark.parametrize("speed", [.3, 2.64])
def test_contact_matches_independent_continuous_physical_coordinate_ode(body, speed):
    f = 440.0
    p = _Parameters(unison_cents=(-1.4, 1.4), modes=8, contact_oversample=4, bridge_mass=body)
    m = _model(f, 16000, p)
    contact = _contact(m, f, 0.6, p, hammer_speed=speed)
    size = len(m.M)
    hammer_mass = 0.007 * (220 / f) ** 0.15
    felt = 2e9 * (0.45 + 0.65) / 2
    minvK = solve(m.M, m.K, assume_a="pos")
    minvC = solve(m.M, m.C, assume_a="pos")
    minvB = solve(m.M, m.B, assume_a="pos")

    def rhs(t, z):
        q, v, hx, hv = z[:size], z[size : 2 * size], z[-2], z[-1]
        force = felt * np.maximum(hx - m.B.T @ q, 0) ** 2.3
        return np.r_[v, -minvK @ q - minvC @ v + minvB @ force, hv, -force.sum() / hammer_mass]

    initial = np.r_[np.zeros(2 * size), 0.0, speed]
    ref = solve_ivp(rhs, (0, contact.seconds), initial, method="DOP853", rtol=1e-10, atol=1e-13).y[:, -1]
    delta = contact.initial - ref[: 2 * size]
    relative = np.sqrt((delta[:size] @ m.K @ delta[:size] + delta[size:] @ m.M @ delta[size:]) / contact.initial_energy)
    assert relative < 0.001
    assert contact.balance_error < 1e-8 and contact.iterations > 0
    assert abs(contact.initial_energy - contact.remaining_energy - contact.dissipated) < 1e-10
    audit = {}
    wave = struck_unisons(f, 2400, 16000, 0.6, _audit=audit, hammer_speed=speed,
                          **{g.name: getattr(p, g.name) for g in fields(p)})
    for sample in [400, 1200, 2200]:
        expected = m.readout @ (expm(m.A * (sample / 16000 - contact.seconds)) @ ref[: 2 * size])[size:] * 0.11
        assert wave[sample] == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("body", [None, 0.02])
def test_coincident_strings_have_the_same_aggregate_source_and_energy(body):
    a, b = {}, {}
    opts = dict(modes=8, bridge_mass=body)
    x = struck_unisons(440.0, 3000, 16000, 0.6, unison_cents=(0.0,), _audit=a, **opts)
    y = struck_unisons(440.0, 3000, 16000, 0.6, unison_cents=(0.0, 0.0, 0.0), _audit=b, **opts)
    np.testing.assert_allclose(x, y, rtol=1e-7, atol=1e-11)
    assert a["initial_energy"] == b["initial_energy"]
    assert a["modes"] == b["modes"] == 8


def test_silence_short_prefix_and_offset_order_have_defined_behavior():
    assert len(struck_unisons(440.0, 0, 8000, 0.6)) == 0
    assert not np.any(struck_unisons(440.0, 100, 8000, 0.0))
    options = dict(modes=8, unison_cents=(-1.4, 0.0, 1.4))
    full = struck_unisons(440.0, 4000, 16000, 0.6, **options)
    for n in [1, 12, 100, 1800]:
        np.testing.assert_allclose(struck_unisons(440.0, n, 16000, 0.6, **options), full[:n], atol=1e-11, rtol=1e-9)
    np.testing.assert_array_equal(full, struck_unisons(440.0, 4000, 16000, 0.6, modes=8, unison_cents=(1.4, -1.4, 0.0)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unison_cents": ()},
        {"unison_cents": [0.0] * 4},
        {"unison_cents": [True]},
        {"unison_cents": [np.nan]},
        {"unison_cents": [51]},
        {"modes": 0},
        {"modes": True},
        {"modes": 65},
        {"contact_oversample": 0},
        {"sustain": 0},
        {"sustain": 1e-320},
        {"stiffness": -1},
        {"brightness": 2},
        {"hammer_mass_scale": 0.1},
        {"felt_stiffness_scale": 20},
        {"bridge_mass": 0},
        {"bridge_mass": 1e-12},
        {"bridge_hz": np.nan},
        {"bridge_q": 0.5},
        {"radiation": "true"},
        {"radiation_loss_scale": 5},
        {"radiation": False, "radiation_loss_scale": 0.5},
    ],
)
def test_invalid_source_parameters_rejected_even_for_empty_requests(kwargs):
    with pytest.raises(ValueError):
        struck_unisons(440.0, 0, 16000, 0.6, **kwargs)
    with pytest.raises(ValueError):
        CoupledPiano(**kwargs)


def test_output_band_and_nonoscillatory_pitch_are_not_silently_certified():
    with pytest.raises(ValueError, match="band"):
        struck_unisons(3150.0, 100, 8000, 0.6, unison_cents=(50.0,))
    with pytest.raises(ValueError, match="band"):
        struck_unisons(440.0, 100, 8000, 0.6, bridge_mass=0.02, bridge_hz=5000.0, mechanical_band=4000.0)
    options = dict(unison_cents=(0.0,), modes=1, sustain=0.0001)
    y = struck_unisons(110.0, 1000, 8000, 0.6, **options)
    assert np.isfinite(y).all() and np.max(abs(y)) > 0
    with pytest.raises(ValueError, match="oscillatory"):
        struck_unisons(110.0, 1000, 8000, 0.6, pitch_compensation=True, **options)


def test_note_overrides_json_renderer_and_release_allocation(tmp_path):
    inst = CoupledPiano(modes=8, ring_time=0.3, release=0.03)
    assert build_instrument(inst.to_spec()) == inst
    ctx = RenderContext(8000, 60)
    voice = Voice(440.0, 69.0, 0.05, 0.6, {"unison_cents": [0.0], "pitch_compensation": False})
    a = inst.voice(voice, ctx, np.random.default_rng(1))
    other = CoupledPiano(modes=8, ring_time=0.3, release=0.03, unison_cents=(0.0,), pitch_compensation=False)
    np.testing.assert_array_equal(a, other.voice(Voice(440.0, 69.0, 0.05, 0.6, {}), ctx, np.random.default_rng(1)))
    note = Note("A4", 0.05, 0.6).param(undamped=True, ring_time=0.5, unison_cents=(0.0,))
    score = Song(Track("p", inst)(note), sample_rate=8000, bpm=60, loudness=None).compile()
    path = tmp_path / "score.json"
    score.to_json(path)
    restored = Score.from_json(path)
    assert restored.events[0].param("undamped") is True
    assert inst.release_time_for(dict(restored.events[0].params)) == 0.5
    result = render_score(restored, workers=1)
    assert result.duration > 0.2 and np.isfinite(result.audio).all()


def test_dynamic_contours_and_bad_note_flags_are_explicitly_rejected():
    inst = CoupledPiano(modes=8)
    ctx = RenderContext(8000, 60)
    for params in [
        {"undamped": "false"},
        {"pedal": 1},
        {"ring_time": 0},
        {"unison_strings": 3},
        {"pitch_points": ((0.0, 0.0), (0.1, 1.0))},
    ]:
        with pytest.raises(ValueError):
            inst.voice(Voice(440.0, 69.0, 0.1, 0.6, params), ctx, np.random.default_rng(1))


@pytest.mark.parametrize("velocity", [0.1, 0.6, 1.0])
def test_tuning_near_bridge_resonance_does_not_alternate_between_mode_windows(velocity):
    audit = {}
    target = 440 * 2 ** ((87 - 69) / 12)
    wave = struck_unisons(target, 800, 44100, velocity, bridge_mass=0.02, pitch_compensation=True, _audit=audit)
    assert np.isfinite(wave).all() and abs(audit["tuning"][-1]["error_cents"]) < 0.05
    assert audit["max_relative_balance_error"] < 1e-8


def test_requested_low_edge_can_use_a_lower_internal_bare_frequency():
    audit = {}
    wave = struck_unisons(20.0, 100, 8000, 0.6, unison_cents=(50.0,), modes=4, pitch_compensation=True, _audit=audit)
    assert np.isfinite(wave).all() and abs(audit["tuning"][-1]["error_cents"]) < 0.05


def test_physical_matrices_and_oversampled_prefix_are_independent_of_output_rate():
    p = _Parameters(modes=16, bridge_mass=0.02, mechanical_band=10000.0)
    a, b = _model(440.0, 8000, p), _model(440.0, 44100, p)
    for field in ["M", "K", "C", "B", "readout", "A"]:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.observation_factor > 1
    full = struck_unisons(440.0, 2000, 8000, 0.6, modes=16, bridge_mass=0.02, mechanical_band=10000.0)
    short = struck_unisons(440.0, 20, 8000, 0.6, modes=16, bridge_mass=0.02, mechanical_band=10000.0)
    np.testing.assert_allclose(short, full[:20], rtol=1e-9, atol=1e-11)


def test_low_rate_observation_matches_resampled_high_rate_on_the_same_mechanics():
    options = dict(modes=16, bridge_mass=0.02, mechanical_band=10000.0)
    high = struck_unisons(440.0, 22050, 44100, 0.6, **options)
    low = struck_unisons(440.0, 4000, 8000, 0.6, **options)
    down = resample_poly(high, 80, 441)
    use = slice(400, 3600)
    assert np.linalg.norm(low[use] - down[use]) / np.linalg.norm(down[use]) < 0.02


def test_multiple_processes_load_registered_coupled_instrument_and_preserve_audio():
    inst = CoupledPiano(modes=4, release=0.03, ring_time=0.2)
    score = Song(
        Track("a", inst)(Note("A4", 0.08, 0.6)),
        Track("b", inst)(Note("C5", 0.08, 0.5)),
        sample_rate=8000,
        bpm=60,
        loudness=None,
    ).compile()
    serial = render_score(score, workers=1)
    parallel = render_score(score, workers=2)
    np.testing.assert_allclose(serial.audio, parallel.audio, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("gate,release", [(0.1, 0.18), (0.1, 0.001), (0.0002, 0.001)])
def test_pedal_and_undamped_flags_preserve_the_complete_prefix_before_key_release(gate, release):
    inst = CoupledPiano(modes=8, release=release, ring_time=0.2)
    ctx = RenderContext(8000, 60)
    waves = [
        inst.voice(Voice(440.0, 69.0, gate, 0.6, params), ctx, np.random.default_rng(1))
        for params in [{}, {"pedal": True}, {"undamped": True}]
    ]
    for wave in waves[1:]:
        np.testing.assert_allclose(wave[: ctx.samples(gate)], waves[0][: ctx.samples(gate)], atol=1e-11, rtol=1e-9)


def test_explicit_hammer_speed_preserves_old_law_and_controls_initial_energy_and_spectrum():
    opts = dict(freq=1760., n=6000, sr=16000, velocity=.6, modes=8)
    old = struck_unisons(**opts)
    np.testing.assert_array_equal(old, struck_unisons(**opts, hammer_speed=None))
    np.testing.assert_array_equal(old, struck_unisons(**opts, hammer_speed=.6+3.4*.6))
    low, high = {}, {}
    a = struck_unisons(**opts, hammer_speed=.3, _audit=low)
    b = struck_unisons(**opts, hammer_speed=3., _audit=high)
    assert high['initial_energy']/low['initial_energy'] == pytest.approx(100.)
    # A speed input changes nonlinear excitation: a scalar output gain cannot
    # account for the resulting normalized spectrum.
    sa, sb = abs(np.fft.rfft(a)), abs(np.fft.rfft(b))
    assert np.linalg.norm(sa/np.linalg.norm(sa)-sb/np.linalg.norm(sb)) > .02
    np.testing.assert_array_equal(a, struck_unisons(**{**opts, 'velocity': .1}, hammer_speed=.3))
    assert not np.any(struck_unisons(**opts, hammer_speed=0.))
    assert not np.any(struck_unisons(**{**opts, 'velocity': 0.}, hammer_speed=3.))


@pytest.mark.parametrize('speed', [-1., 4.1, np.nan, np.inf, True, 'bad'])
def test_invalid_hammer_speed_is_rejected_before_empty_or_silent_render(speed):
    with pytest.raises(ValueError, match='hammer_speed'):
        struck_unisons(440., 0, 16000, 0., hammer_speed=speed)
    with pytest.raises(ValueError, match='hammer_speed'):
        CoupledPiano().voice(Voice(440., 69., 0., 0., {'hammer_speed': speed}),
                             RenderContext(8000, 60), np.random.default_rng(0))


def test_subnormal_hammer_energy_raises_explicitly_without_dividing_by_zero():
    with pytest.raises(ValueError, match='hammer_speed'):
        struck_unisons(440., 400, 16000, .6, hammer_speed=1e-300)


def test_speed_curve_note_priority_json_and_physical_energy_after_tuning():
    curve = ((0., 0.), (1., 4.))
    inst = CoupledPiano(modes=8, ring_time=.1, release=.03, hammer_speed_points=curve)
    plain = CoupledPiano(modes=8, ring_time=.1, release=.03)
    ctx = RenderContext(8000, 60)

    def voice(instrument, params):
        return instrument.voice(Voice(440., 69., .05, .5, params), ctx, np.random.default_rng(0))

    np.testing.assert_array_equal(voice(inst, {}), voice(plain, {'hammer_speed': 2.}))
    np.testing.assert_array_equal(voice(inst, {'hammer_speed': None}), voice(plain, {}))
    np.testing.assert_array_equal(voice(inst, {'hammer_speed_points': []}), voice(plain, {}))
    np.testing.assert_array_equal(voice(inst, {'hammer_speed': .3}), voice(plain, {'hammer_speed': .3}))
    assert build_instrument(inst.to_spec()) == inst
    score = Song(Track('p', inst)(Note('A4', .05, .5).param(hammer_speed=.3)),
                 sample_rate=8000, bpm=60, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    np.testing.assert_array_equal(render_score(score).audio, render_score(restored).audio)
    audit = {}
    struck_unisons(1760., 800, 16000, .6, pitch_compensation=True, hammer_speed=.3, _audit=audit)
    bare = audit['tuning'][-1]['bare_frequency']
    expected = .5*.007*(220/bare)**.15*.3**2
    assert audit['initial_energy'] == pytest.approx(expected)
    assert audit['max_relative_balance_error'] < 1e-8


@pytest.mark.parametrize('curve', [None, [(0., 0.)], [(0., 0.), (1., 5.)], [(0., 2.), (1., 1.)]])
def test_speed_curves_rejected_on_instrument_and_note(curve):
    with pytest.raises(ValueError, match='hammer_speed_points'):
        CoupledPiano(hammer_speed_points=curve)
    with pytest.raises(ValueError, match='hammer_speed_points'):
        CoupledPiano().voice(Voice(440., 69., .05, .5, {'hammer_speed_points': curve}),
                             RenderContext(8000, 60), np.random.default_rng(0))
