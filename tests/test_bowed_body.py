"""Independent physical-coordinate equations and energy accounting."""
import numpy as np
import pytest
from scipy.linalg import solve

from notes.engine.bow_contact import _model
from notes.engine.bowed_body import mechanics, midpoint


@pytest.mark.parametrize('bodies', [(), ((460., 20., .01),), ((460., 20., .01), (2400., 2., .002))])
@pytest.mark.parametrize('sr,factor', [(8000, 16), (22050, 32)])
def test_full_physical_midpoint_equations_and_work_identity(bodies, sr, factor):
    m = mechanics(196., modes=16, bandwidth=4000., bodies=bodies)
    p = midpoint(m, sr, factor)
    n, dt = len(p.omega), p.dt
    assert np.linalg.eigvalsh(m.M).min() > 0 and np.linalg.eigvalsh(m.K).min() > 0
    assert np.linalg.eigvalsh(p.D).min() > 0 and p.mobility > 0
    rng = np.random.default_rng(17)
    z = rng.normal(size=2*n)*.01
    q, v = m.U@(z[:n]/p.omega), m.U@z[n:]
    effective_k = (m.M@m.U*p.omega**2)@m.U.T@m.M
    eye = np.eye(n)
    lhs = np.block([[eye, -dt*eye/2], [dt*effective_k/2, m.M+dt*m.C/2]])
    rhs = np.r_[q+dt*v/2, -dt*effective_k@q/2+(m.M-dt*m.C/2)@v+dt*m.force*.7]
    physical = solve(lhs, rhs)
    expected = np.r_[p.omega*(m.U.T@m.M@physical[:n]), m.U.T@m.M@physical[n:]]
    actual = p.P@z+p.Q*.7
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    midv = (z[n:]+actual[n:])/2
    change = .5*(actual@actual-z@z)
    assert change+dt*(midv@p.D@midv) == pytest.approx(dt*.7*(p.coupling@midv), abs=1e-14)
    assert not m.M.flags.writeable and not p.P.flags.writeable


def test_rigid_endpoint_reduces_to_the_existing_diagonal_bow_model():
    m = mechanics(110., modes=8, bandwidth=4000.)
    p = midpoint(m, 8000, 16)
    old = _model(110., 8000, 16, 8, 4000., 1e-5, 110., .69, 12., 1., .08, .012)
    z = np.random.default_rng(17).normal(size=16)*.001
    expected = np.r_[old.p11*z[:8]+old.p12*z[8:]+old.qx*.2,
                     old.p21*z[:8]+old.p22*z[8:]+old.qv*.2]
    actual = p.P@z+p.Q*.2
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(p.readout, old.bridge, atol=1e-15)
    assert not p.body_velocity.any()


def test_basis_is_physical_and_keeps_body_motion_coupled():
    m = mechanics(196., modes=128, bodies=((460., 20., .01), (550., 20., .005)))
    a, b = midpoint(m, 8000, 16), midpoint(m, 22050, 32)
    assert a.mechanics is b.mechanics
    assert len(a.omega) == len(b.omega)
    assert np.max(abs(m.M-np.diag(np.diag(m.M)))) > 0
    assert np.max(abs(a.D-np.diag(np.diag(a.D)))) > 0


@pytest.mark.parametrize('options', [{'modes': 3}, {'modes': True}, {'stiffness': -.1}, {'width': -.1},
    {'freq': 10}, {'freq': float('nan')}, {'position': 1.}, {'sustain': 0},
    {'bodies': ((460, 20, 0),)}, {'bodies': ((20000, 20, .01),)}, {'bodies': ((460, -2, .01),)},
    {'bodies': ((460, 2),)}, {'bodies': 'bad'}, {'bodies': ((460, 2, .01),)*9}])
def test_invalid_physical_controls_are_rejected(options):
    kwargs = {'freq': 196., **options}
    with pytest.raises(ValueError):
        mechanics(**kwargs)


@pytest.mark.parametrize('sr,factor', [(3000, 16), (8000, 3), (8000, True), (8000, 16.), (8000, 1)])
def test_invalid_or_unresolved_steps_are_rejected(sr, factor):
    with pytest.raises(ValueError):
        midpoint(mechanics(196.), sr, factor)


@pytest.mark.parametrize('weights', [(1., -.4), (0., 0.), (.3, 1.7)])
def test_weighted_port_matches_continuum_energy_and_physical_step(weights):
    bodies = ((460., 20., .01), (2400., 2., .002))
    m = mechanics(196., modes=8, bandwidth=4000., bodies=bodies, bridge_weights=weights)
    rng = np.random.default_rng(17)
    q, velocity = rng.normal(size=(2, len(m.M)))*.001
    # Independently integrate the string's actual displacement ansatz over x/L.
    x, quadrature = np.polynomial.legendre.leggauss(64)
    x, quadrature = (x+1)/2, quadrature/2
    h = np.arange(1, 9)
    mass = 110.*(1+1e-5)/(4*.69*196.**2)
    local_v = np.sin(np.pi*x[:, None]*h)@velocity[:8]+(1-x)*(np.asarray(weights)@velocity[8:])
    slope = (np.pi*h*np.cos(np.pi*x[:, None]*h))@q[:8]-np.asarray(weights)@q[8:]
    curvature = (-(np.pi*h)**2*np.sin(np.pi*x[:, None]*h))@q[:8]
    kinetic = mass*(quadrature@local_v**2)/2
    elastic = 110./(2*.69)*(quadrature@(slope**2+1e-5/np.pi**2*curvature**2))
    for j, (f, Q, peak) in enumerate(bodies):
        mb = Q/(peak*2*np.pi*f)
        kinetic += mb*velocity[8+j]**2/2
        elastic += mb*(2*np.pi*f)**2*q[8+j]**2/2
    assert .5*velocity@m.M@velocity == pytest.approx(kinetic, rel=1e-12)
    assert .5*q@m.K@q == pytest.approx(elastic, rel=1e-12)
    p = midpoint(m, 8000, 16)
    n, dt = len(p.omega), p.dt
    z = np.r_[p.omega*(m.U.T@m.M@q), m.U.T@m.M@velocity]
    effective_k = (m.M@m.U*p.omega**2)@m.U.T@m.M
    identity = np.eye(n)
    lhs = np.block([[identity, -dt*identity/2], [dt*effective_k/2, m.M+dt*m.C/2]])
    rhs = np.r_[q+dt*velocity/2, -dt*effective_k@q/2+(m.M-dt*m.C/2)@velocity+dt*m.force*.3]
    physical = solve(lhs, rhs)
    actual = p.P@z+p.Q*.3
    np.testing.assert_allclose(m.U@(actual[:n]/p.omega), physical[:n], atol=1e-13)
    np.testing.assert_allclose(m.U@actual[n:], physical[n:], atol=1e-13)
    midv = (z[n:]+actual[n:])/2
    assert .5*(actual@actual-z@z)+dt*(midv@p.D@midv) == pytest.approx(
        dt*.3*(p.coupling@midv), abs=1e-12)
    np.testing.assert_allclose(p.body_velocity, np.asarray(weights)@m.U[-2:])


def test_unit_weights_keep_default_cache_and_signed_ports_are_coordinate_congruences():
    bodies = ((460., 20., .01), (550., 20., .005))
    base = mechanics(196., modes=8, bandwidth=4000., bodies=bodies)
    assert base is mechanics(196., modes=8, bandwidth=4000., bodies=bodies, bridge_weights=(1., 1.))
    changed = mechanics(196., modes=8, bandwidth=4000., bodies=bodies, bridge_weights=(1., -1.))
    sign = np.diag(np.r_[np.ones(9), -1.])
    for name in ('M', 'K', 'C'):
        np.testing.assert_array_equal(getattr(changed, name), sign@getattr(base, name)@sign)
    np.testing.assert_array_equal(changed.force, sign@base.force)
    np.testing.assert_array_equal(changed.readout, sign@base.readout)
    assert not changed.bridge_weights.flags.writeable


@pytest.mark.parametrize('weights', [(1.,), (1., 1., 1.), (1., float('nan')), 'bad'])
def test_invalid_bridge_weights(weights):
    with pytest.raises(ValueError):
        mechanics(196., bodies=((460., 20., .01), (550., 20., .005)), bridge_weights=weights)
