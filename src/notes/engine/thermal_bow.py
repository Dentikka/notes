"""Experimental thermo-elasto-plastic bow contact on passive string/body modes.

Inspired by Woodhouse, Bowed String Simulation Using a Thermal Friction Model:
https://euphonics.org/wp-content/uploads/2022/03/Thermal_bowing.pdf
This is an independent positive finite-volume spreading-ladder approximation,
with finite tangential compliance, not that paper's exact Green-function model.
No recorded excitation, carrier phase or measured acoustic IR is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.signal import resample_poly

from notes.engine.bowed_body import BodyModes, _finite, mechanics, midpoint
from notes.engine.dsp import dc_block

DEFAULT_BODIES = ((460., 20., .01), (550., 20., .005), (2400., 2., 2/(.06*2*np.pi*2400)))


@dataclass(frozen=True)
class HeatNetwork:
    capacities: np.ndarray
    conductance: np.ndarray
    ambient: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    convection: float
    dt: float


def heat_network(normal: float, dt: float, cells: int = 11) -> HeatNetwork:
    """Positive RC volumes with expanding cross-section below the contact patch.

    Material values and nominal patch/layer scales follow the 2003 paper's
    example table. Geometry and the finite spreading ladder remain approximations.
    """
    normal, dt = _finite('normal force', normal), _finite('time step', dt)
    if (normal <= 0 or dt <= 0 or not isinstance(cells, int) or isinstance(cells, bool)
            or not 4 <= cells <= 20):
        raise ValueError('positive load/time step and four to twenty thermal cells required')
    radius = 250e-6*np.sqrt(normal)
    area = np.pi*radius**2
    layer, rosin_density, rosin_heat, rosin_k = 1e-6, 1010., 250., .113
    lengths = 1e-6*2**np.arange(cells)
    edges = np.r_[0., np.cumsum(lengths)]
    centers = (edges[:-1]+edges[1:])/2
    n = 1+2*cells
    C, L, ambient = np.zeros(n), np.zeros((n, n)), np.zeros(n)
    C[0] = area*layer*rosin_density*rosin_heat

    def connect(a: int, b: int, value: float) -> None:
        L[a, a] += value
        L[b, b] += value
        L[a, b] -= value
        L[b, a] -= value

    for side, (density, specific_heat, conductivity) in enumerate(((3000., 1000., 1.4), (600., 1680., .2))):
        start = 1+side*cells
        volumes = np.pi*((radius+edges[1:])**3-(radius+edges[:-1])**3)/3
        C[start:start+cells] = density*specific_heat*volumes
        surface_r = layer/(2*rosin_k*area)+(1/radius-1/(radius+centers[0]))/(np.pi*conductivity)
        connect(0, start, 1/surface_r)
        for i in range(cells-1):
            resistance = (1/(radius+centers[i])-1/(radius+centers[i+1]))/(np.pi*conductivity)
            connect(start+i, start+i+1, 1/resistance)
        far = conductivity*np.pi*(radius+centers[-1])
        L[start+cells-1, start+cells-1] += far
        ambient[start+cells-1] = far
    inverse = solve(np.diag(C/dt)+L, np.eye(n), assume_a='pos')
    P, Q = inverse*(C/dt)[None, :], inverse[:, 0]
    convection = 2*radius*layer*rosin_density*rosin_heat
    for array in (C, L, ambient, P, Q):
        array.flags.writeable = False
    return HeatNetwork(C, L, ambient, P, Q, float(convection), dt)


def heat_step(model: HeatNetwork, temperature: np.ndarray, energy: float, slip: float) -> tuple[np.ndarray, float]:
    """Advance a nonnegative temperature rise with prescribed plastic heat.

    The returned loss includes substrate conduction to ambient and transport
    of heated rosin. Sum(C * temperature) is thermal energy above ambient.
    """
    temperature = np.asarray(temperature, dtype=float)
    energy, slip = _finite('plastic heat', energy), _finite('slip', slip)
    if (temperature.shape != model.capacities.shape or not np.isfinite(temperature).all()
            or temperature.min() < 0 or energy < 0):
        raise ValueError('temperature rise and supplied heat must be finite and nonnegative')
    return _heat_step(model, temperature, energy, slip)


def _heat_step(model: HeatNetwork, temperature: np.ndarray, energy: float, slip: float) -> tuple[np.ndarray, float]:
    power = energy/model.dt
    free = model.P@temperature+model.Q*power
    h = model.convection*abs(slip)
    result = free-model.Q*(h*free[0])/(1+h*model.Q[0])
    loss = model.dt*(model.ambient@result+h*result[0])
    return result, float(loss)


def contact(free: float, mobility: float, previous_force: float, limit: float,
            stiffness: float, dt: float) -> tuple[float, float, float, float, bool]:
    """Unique elastic predictor/plastic return map with an explicit energy budget.

    The internal state is tangential elastic force. For fixed temperature its
    force/relative-velocity graph is monotone. Backward-Euler elastic loss is
    tracked separately from plastic heat, rather than silently heating the rosin.
    """
    free, mobility, previous_force, limit, stiffness, dt = (
        _finite(name, value) for name, value in zip(
            ('free speed', 'mobility', 'previous force', 'yield limit', 'contact stiffness', 'time step'),
            (free, mobility, previous_force, limit, stiffness, dt), strict=True))
    if min(mobility, limit) < 0 or min(stiffness, dt) <= 0:
        raise ValueError('nonnegative mobility/limit and positive contact stiffness/time step required')
    return _contact(free, mobility, previous_force, limit, stiffness, dt)


def _contact(free: float, mobility: float, previous_force: float, limit: float,
             stiffness: float, dt: float) -> tuple[float, float, float, float, bool]:
    kdt = stiffness*dt
    elastic = (previous_force+kdt*free)/(1+mobility*kdt)
    force = min(limit, max(-limit, elastic))
    slip = free-mobility*force
    increment = dt*slip-(force-previous_force)/stiffness
    plastic = force*increment if abs(elastic) > limit else 0.
    if plastic < -1e-13:
        raise ArithmeticError('negative plastic dissipation')
    numerical = (force-previous_force)**2/(2*stiffness)
    return force, slip, max(0., plastic), numerical, abs(elastic) > limit


def thermal_bowed_string(
    freq: float, n: int, sr: int, *, gate: float | None = None,
    bow_speed: float = .08, force_ratio: float = 16., attack: float = .15,
    contact_stiffness: float = 1e5, cold_mu: float = 1.2, hot_mu: float = .2,
    softening_temperature: float = 40., cells: int = 11, oversample: int = 32,
    isothermal: bool = False, audit: bool = False, trace: dict[str, np.ndarray | int] | None = None,
    modes: int = 128, bandwidth: float = 18000., stiffness: float = 1e-5,
    tension: float = 110., length: float = .69, sustain: float = 12., treble: float = 1.,
    position: float = .08, width: float = .012, bodies: BodyModes = DEFAULT_BODIES,
) -> tuple[np.ndarray, dict[str, float | int | bool | None]]:
    """Experimental sustained source with thermal friction and moving body modes.

    Bow speed has units m/s; normal force equals ``force_ratio`` times string
    impedance times bow speed. Temperature controls the plastic yield strength,
    while tangential elastic force persists across sticking/slipping steps.
    Numerical contact loss is accounted separately and never heats the rosin.

    Returns the bare endpoint-gradient signal and diagnostics. Optional traces
    expose force, slip, temperature rise, and each body's velocity/acceleration
    at integration rate. These are not calibrated acoustic pressure. ``audit``
    checks the coupled numerical energy, including heat and contact strain;
    mechanical prewarping converges to the physical system under refinement.
    Static sticking may leave nonzero mechanical/contact energy after note-off.

    Parameters describe an approximation, not measured violin rosin. No bow
    roughness, torsion, changing pitch, linked legato or radiation model is
    inferred here. This does not replace the accepted Violin/Cello presets.
    """
    if (not isinstance(n, int) or isinstance(n, bool) or n < 0
            or not isinstance(sr, int) or isinstance(sr, bool) or sr < 4000):
        raise ValueError('integer nonnegative length and sample rate >=4000 required')
    if (not isinstance(oversample, int) or isinstance(oversample, bool)
            or oversample not in (1, 2, 4, 8, 16, 32, 64)):
        raise ValueError('oversample must be an integer power of two from one to 64')
    if not isinstance(cells, int) or isinstance(cells, bool) or not 4 <= cells <= 20:
        raise ValueError('four to twenty thermal cells required')
    if not isinstance(isothermal, bool) or not isinstance(audit, bool):
        raise ValueError('isothermal and audit must be boolean')
    if trace is not None and not isinstance(trace, dict):
        raise ValueError('trace must be a dictionary or None')
    bow_speed, force_ratio, attack, contact_stiffness, cold_mu, hot_mu, softening_temperature = (
        _finite(name, value) for name, value in zip(
            ('bow speed', 'force ratio', 'attack', 'contact stiffness', 'cold mu', 'hot mu', 'softening temperature'),
            (bow_speed, force_ratio, attack, contact_stiffness, cold_mu, hot_mu, softening_temperature), strict=True))
    if (min(bow_speed, force_ratio) < 0 or min(attack, contact_stiffness, softening_temperature) <= 0
            or not 0 < hot_mu <= cold_mu):
        raise ValueError('invalid bow motion or contact coefficients')
    if gate is None:
        gate = n/sr
    gate = _finite('gate', gate)
    if gate < 0:
        raise ValueError('gate must be nonnegative')
    freq, tension, length, stiffness = (_finite(name, value) for name, value in zip(
        ('frequency', 'tension', 'length', 'stiffness'), (freq, tension, length, stiffness), strict=True))
    m = mechanics(freq, modes=modes, bandwidth=bandwidth, stiffness=stiffness, tension=tension,
                  length=length, sustain=sustain, treble=treble, position=position, width=width, bodies=bodies)
    normal = float(force_ratio*tension*np.sqrt(1+stiffness)/(2*length*freq)*bow_speed)
    if normal <= 0 or n == 0 or gate == 0:
        return np.zeros(n), {'silent': True}
    factor = oversample
    while True:
        try:
            model = midpoint(m, sr, factor)
            # Bound the extra elastic-contact frequency as well as body modes.
            highest = np.sqrt(m.physical_omega.max()**2+contact_stiffness*(model.coupling@model.coupling))
            if highest/(2*np.pi) < .42*sr*factor:
                break
        except ValueError:
            pass
        if factor >= 64:
            raise ValueError('thermoelastic contact needs more than 64x integration')
        factor *= 2
    heat = heat_network(normal, model.dt, cells)
    temperature = np.zeros(len(heat.capacities))
    count = len(model.omega)
    state = np.zeros(2*count)
    b = 0.
    output = np.empty(n*factor)
    work = mechanical_loss = plastic_heat = numerical_loss = cooling = balance = 0.
    peak_temperature = 0.
    yielded_steps = 0
    body_map = m.U[-m.body_count:] if m.body_count else np.zeros((0, count))
    if trace is not None:
        force_trace, slip_trace, temperature_trace = (np.empty_like(output) for _ in range(3))
        body_v = np.empty((m.body_count, len(output)))
        body_a = np.empty_like(body_v)
    for i in range(len(output)):
        time = i*model.dt
        speed = bow_speed*min(1., time/attack)*min(1., max(0., (gate-time)/.06))
        mu = cold_mu if isothermal else max(
            hot_mu, cold_mu-(cold_mu-hot_mu)*float(temperature[0])/softening_temperature)
        free = model.P@state
        free_speed = float(.5*model.coupling@(state[count:]+free[count:]))
        force, slip, heat_energy, numerical, yielded = _contact(speed-free_speed, model.mobility,
            b, normal*mu, contact_stiffness, model.dt)
        next_state = free+model.Q*force
        next_temperature, heat_loss = _heat_step(heat, temperature, heat_energy, slip)
        output[i] = model.readout@next_state[:count]
        peak_temperature = max(peak_temperature, float(next_temperature[0]))
        yielded_steps += int(yielded)
        bow_work = model.dt*force*speed
        work += bow_work
        plastic_heat += heat_energy
        numerical_loss += numerical
        cooling += heat_loss
        if trace is not None:
            force_trace[i], slip_trace[i], temperature_trace[i] = force, slip, next_temperature[0]
            body_v[:, i] = body_map@next_state[count:]
            body_a[:, i] = body_map@(next_state[count:]-state[count:])/model.dt
        if audit:
            midv = (state[count:]+next_state[count:])/2
            mech_loss = model.dt*float(midv@model.D@midv)
            change = .5*(next_state@next_state-state@state)+(force**2-b**2)/(2*contact_stiffness)
            change += heat.capacities@(next_temperature-temperature)
            balance = max(balance, abs(change-bow_work+mech_loss+numerical+heat_loss))
            mechanical_loss += mech_loss
            if next_temperature.min() < -1e-12:
                raise ArithmeticError('negative temperature above ambient')
        state, temperature, b = next_state, next_temperature, force
    if trace is not None:
        trace.update(force=force_trace, slip=slip_trace, temperature=temperature_trace,
                     body_velocity=body_v, body_acceleration=body_a, integration_rate=sr*factor)
    wave = resample_poly(output, 1, factor)[:n]
    return dc_block(wave, sr, 15.), {'oversample': factor, 'string_modes': count-m.body_count,
        'body_modes': m.body_count, 'normal_force': normal, 'peak_temperature': peak_temperature,
        'final_surface_temperature': float(temperature[0]), 'yielded_fraction': yielded_steps/max(1, len(output)),
        'final_mechanical_energy': float(.5*(state@state)), 'final_contact_energy': b*b/(2*contact_stiffness),
        'final_heat_energy': float(heat.capacities@temperature), 'bow_work': work,
        'mechanical_loss': mechanical_loss if audit else None, 'plastic_heat': plastic_heat,
        'numerical_loss': numerical_loss, 'cooling_loss': cooling,
        'maximum_energy_balance_error': balance if audit else None}
