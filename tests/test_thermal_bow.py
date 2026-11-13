"""Independent convex-contact roots, thermal solves and total source work."""
import numpy as np
import pytest
from notes.engine.bowed_body import mechanics, midpoint
from notes.engine.thermal_bow import contact, heat_network, heat_step, thermal_bowed_string
from scipy.linalg import solve
from scipy.optimize import brentq


def test_elastoplastic_contact_matches_independent_velocity_root_and_energy():
    def residual(u, free, mobility, previous, limit, stiffness, dt):
        return u+mobility*np.clip(previous+stiffness*dt*u, -limit, limit)-free

    rng = np.random.default_rng(17)
    for _ in range(300):
        free, previous = rng.normal(size=2)*.3
        mobility = 10**rng.uniform(-3, 0)
        limit = rng.uniform(.01, .8)
        stiffness = 10**rng.uniform(3, 6)
        dt = 10**rng.uniform(-6, -4)
        force, slip, plastic, numerical, yielded = contact(free, mobility, previous, limit, stiffness, dt)
        root = brentq(residual, free-mobility*limit-1e-8, free+mobility*limit+1e-8,
                      args=(free, mobility, previous, limit, stiffness, dt), xtol=1e-14)
        assert slip == pytest.approx(root, abs=1e-12)
        assert abs(force) <= limit and plastic >= 0 and numerical >= 0
        change = (force**2-previous**2)/(2*stiffness)
        assert dt*force*slip == pytest.approx(change+plastic+numerical, abs=1e-16)
        if not yielded:
            assert plastic == 0.


def test_elastic_force_memory_can_return_stored_energy_without_negative_dissipation():
    force, slip, plastic, numerical, yielded = contact(-.1, .2, .2, .3, 1e5, 1e-6)
    assert force > 0 and slip < 0 and not yielded
    assert force*slip < 0 and plastic == 0 and numerical > 0
    # Zero yield releases all stored contact energy with no frictional heat.
    force, _, plastic, numerical, yielded = contact(.1, 0., .2, 0., 1e5, 1e-6)
    assert force == 0 and yielded and plastic == 0
    assert numerical == pytest.approx(.2**2/(2e5))


@pytest.mark.parametrize('normal', [.05, .55, 2.])
@pytest.mark.parametrize('dt', [1/64000, 1/352800])
@pytest.mark.parametrize('slip', [0., -.3, .3])
def test_thermal_network_matches_direct_conduction_solve_and_energy(normal, dt, slip):
    heat = heat_network(normal, dt)
    rng = np.random.default_rng(17)
    initial = rng.uniform(0, 25, len(heat.capacities))
    energy = 2e-6
    actual, loss = heat_step(heat, initial, energy, slip)
    matrix = np.diag(heat.capacities/dt)+heat.conductance
    matrix[0, 0] += heat.convection*abs(slip)
    rhs = heat.capacities*initial/dt
    rhs[0] += energy/dt
    expected = solve(matrix, rhs, assume_a='pos')
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    assert heat.capacities@(actual-initial) == pytest.approx(energy-loss, abs=1e-13)
    assert actual.min() >= 0 and loss >= 0
    assert not heat.capacities.flags.writeable and not heat.P.flags.writeable
    assert np.linalg.eigvalsh(heat.conductance).min() > 0


def test_heat_cools_without_spontaneous_heating():
    heat = heat_network(.55, .0001)
    temperature = np.full(len(heat.capacities), 10.)
    for _ in range(50):
        after, loss = heat_step(heat, temperature, 0., 0.)
        assert 0 <= after.min() <= after.max() <= 10.+1e-12
        assert heat.capacities@after <= heat.capacities@temperature
        assert loss >= 0
        temperature = after


def test_coupled_source_accounts_for_mechanical_contact_and_thermal_energy():
    trace = {}
    y, stats = thermal_bowed_string(196., 3200, 8000, gate=.16, attack=.05, modes=16,
        bandwidth=4000., tension=55., length=.328, bodies=((460., 20., .01),), audit=True, trace=trace)
    stored = sum(stats[k] for k in ('final_mechanical_energy', 'final_contact_energy', 'final_heat_energy'))
    supplied = stats['bow_work']-sum(stats[k] for k in ('mechanical_loss', 'numerical_loss', 'cooling_loss'))
    assert stored == pytest.approx(supplied, abs=1e-10)
    assert stats['maximum_energy_balance_error'] < 1e-12
    assert stats['peak_temperature'] > 0 and stats['plastic_heat'] > 0
    assert stats['numerical_loss'] > 0 and stats['numerical_loss'] < .02*stats['plastic_heat']
    assert np.isfinite(y).all() and abs(y).max() > .001
    assert trace['temperature'].min() >= 0
    assert trace['body_velocity'].shape == trace['body_acceleration'].shape == (1, 3200*stats['oversample'])


def test_changing_requested_duration_preserves_integrated_contact_prefix():
    kwargs = {'gate': .2, 'attack': .05, 'modes': 8, 'bandwidth': 4000., 'bodies': ()}
    a, b = {}, {}
    thermal_bowed_string(196., 320, 8000, trace=a, **kwargs)
    thermal_bowed_string(196., 640, 8000, trace=b, **kwargs)
    for key in ('force', 'slip', 'temperature'):
        np.testing.assert_array_equal(a[key], b[key][:len(a[key])])


def test_adaptive_integration_resolves_contact_frequency_in_hz():
    _, stats = thermal_bowed_string(196., 1, 8000, modes=8, bandwidth=4000., bodies=(), oversample=1)
    m = mechanics(196., modes=8, bandwidth=4000.)
    p = midpoint(m, 8000, stats['oversample'])
    bound_hz = np.sqrt(m.physical_omega.max()**2+1e5*(p.coupling@p.coupling))/(2*np.pi)
    assert bound_hz < .42*8000*stats['oversample']
    assert stats['oversample'] < 16


@pytest.mark.parametrize('kwargs', [{'n': 0}, {'gate': 0}, {'force_ratio': 0}, {'bow_speed': 0}])
def test_intentional_silence(kwargs):
    args = {'freq': 196., 'n': 10, 'sr': 8000, **kwargs}
    y, stats = thermal_bowed_string(**args)
    assert y.shape == (args['n'],) and not y.any() and stats['silent']


@pytest.mark.parametrize('kwargs', [
    {'n': -1}, {'n': True}, {'n': 10.5}, {'sr': 3000}, {'sr': 8000.},
    {'oversample': 3}, {'oversample': 16.}, {'oversample': True},
    {'gate': -1}, {'gate': float('nan')}, {'attack': 0}, {'attack': float('inf')},
    {'force_ratio': -1}, {'bow_speed': -1}, {'contact_stiffness': 0},
    {'hot_mu': 0}, {'hot_mu': 2}, {'cold_mu': float('nan')}, {'softening_temperature': 0},
    {'cells': 3}, {'cells': 21}, {'cells': 11.}, {'cells': True},
    {'isothermal': 1}, {'audit': 1}, {'trace': []},
])
def test_invalid_source_controls(kwargs):
    with pytest.raises(ValueError):
        thermal_bowed_string(**{'freq': 196., 'n': 10, 'sr': 8000, **kwargs})


@pytest.mark.parametrize('values', [(.1, -.1, 0., .2, 1e5, 1e-5),
    (.1, .1, 0., -.2, 1e5, 1e-5), (.1, .1, 0., .2, 0., 1e-5),
    (.1, .1, 0., .2, 1e5, 0.), (float('nan'), .1, 0., .2, 1e5, 1e-5)])
def test_invalid_contact_controls(values):
    with pytest.raises(ValueError):
        contact(*values)


@pytest.mark.parametrize('values', [(0., 1e-5, 11), (.5, 0., 11), (.5, 1e-5, 3),
    (.5, 1e-5, 21), (.5, 1e-5, True), (float('inf'), 1e-5, 11)])
def test_invalid_thermal_geometry(values):
    with pytest.raises(ValueError):
        heat_network(*values)


def test_invalid_thermal_state():
    heat = heat_network(.55, 1e-5)
    for temperature, energy, slip in [(np.zeros(2), 0., 0.),
        (-np.ones(len(heat.capacities)), 0., 0.),
        (np.full(len(heat.capacities), np.nan), 0., 0.),
        (np.zeros(len(heat.capacities)), -1., 0.),
        (np.zeros(len(heat.capacities)), 0., float('inf'))]:
        with pytest.raises(ValueError):
            heat_step(heat, temperature, energy, slip)
