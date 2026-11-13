"""Experimental passive modal string driven by a smooth bow friction curve.

The sector-bounded curve is equation 2 of Russo, Ducceschi and Bilbao,
Efficient Simulation of the Bowed String in Modal Form (DAFx 2022):
https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_14.pdf
The solver here is an independently implemented implicit midpoint method,
not that paper's non-iterative update. No recorded signal or response is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypedDict

import numpy as np
from scipy.signal import resample_poly

from notes.engine.dsp import dc_block


class BowDiagnostics(TypedDict):
    mode_count: int
    oversample: int
    uniqueness_margin: float
    initial_energy: float
    final_energy: float
    bow_work: float | None
    friction_loss: float | None
    damping_loss: float | None
    maximum_energy_balance_error: float | None


@dataclass(frozen=True)
class _Model:
    p11: np.ndarray
    p12: np.ndarray
    p21: np.ndarray
    p22: np.ndarray
    qx: np.ndarray
    qv: np.ndarray
    coupling: np.ndarray
    bridge: np.ndarray
    gamma: np.ndarray
    mobility: float
    dt: float


@lru_cache(maxsize=128)
def _model(freq: float, sr: int, factor: int, modes: int, bandwidth: float,
           stiffness: float, tension: float, length: float, sustain: float,
           treble: float, position: float, width: float) -> _Model:
    rate = sr*factor
    h = np.arange(1, modes+1, dtype=float)
    hz = freq*h*np.sqrt((1+stiffness*h*h)/(1+stiffness))
    h, hz = h[hz < bandwidth], hz[hz < bandwidth]
    if len(h) < 2:
        raise ValueError('physical modal band needs at least two string modes')
    mass = tension*(1+stiffness)/(8*length*freq*freq)
    omega = 2*rate*np.tan(np.pi*hz/rate)
    gamma = np.log(1000)*(1/sustain+max(0, 1/treble-1/sustain)*(hz/freq-1)**2
                          / max(1, (4000/freq-1)**2))
    dt = 1/rate
    s = omega*dt/2
    denominator = 1+gamma*dt+s*s
    p11 = (1+gamma*dt-s*s)/denominator
    p12 = 2*s/denominator
    p21 = -p12
    p22 = (1-gamma*dt-s*s)/denominator
    coupling = np.sin(np.pi*h*position)*np.sinc(h*width/2)/np.sqrt(mass)
    qx, qv = dt*s*coupling/denominator, dt*coupling/denominator
    bridge = tension*np.pi*h*(1+stiffness*h*h)/(length*np.sqrt(mass)*omega)
    for value in (p11, p12, p21, p22, coupling, qx, qv, bridge, gamma):
        value.flags.writeable = False
    return _Model(p11, p12, p21, p22, qx, qv, coupling, bridge, gamma,
                  float(.5*coupling@qv), dt)


def _soft_contact(free: float, mobility: float, normal: float, a: float) -> tuple[float, float]:
    """Solve the unique scalar midpoint contact equation on its physical bracket."""
    direction = float(np.sign(free))
    magnitude = abs(free)
    alpha = mobility*normal
    low, high = 0., magnitude
    u = magnitude/(1+alpha*np.sqrt(2*a)*np.exp(.5))
    for _ in range(64):
        exponential = np.sqrt(2*a)*np.exp(.5-a*u*u)
        error = u+alpha*u*exponential-magnitude
        if abs(error) < 1e-13:
            break
        if error > 0:
            high = u
        else:
            low = u
        derivative = 1+alpha*exponential*(1-2*a*u*u)
        proposed = u-error/derivative
        # At fast slip the root can equal the endpoint after underflow.
        u = proposed if low <= proposed <= high else (low+high)/2
    if abs(u+alpha*np.sqrt(2*a)*u*np.exp(.5-a*u*u)-magnitude) > 1e-12:
        raise ArithmeticError('soft contact root did not converge')
    return float(direction*normal*np.sqrt(2*a)*u*np.exp(.5-a*u*u)), direction*u


def modal_bowed_string(
    freq: float, n: int, sr: int, *, gate: float | None = None,
    bow_speed: float = .08, force_ratio: float = 16., position: float = .08,
    width: float = .012, soft_a: float = 100., attack: float = .15,
    pressure_attack: float | None = None, modes: int = 128, bandwidth: float = 18000.,
    stiffness: float = 1e-5, tension: float = 110., length: float = .69,
    sustain: float = 12., treble: float = 1., oversample: int = 16,
    audit: bool = False,
) -> tuple[np.ndarray, BowDiagnostics]:
    """Steady bridge-force readout; nonlinear friction supplies all excitation.

    ``freq`` is the bare fundamental; friction can shift the sounding pitch.
    The physical modal band is independent of output rate. The integration rate
    is doubled, at most to 64x, until every mode is resolved and the contact
    equation has a globally unique root. ``oversample`` is the minimum factor.
    Bow force is ``force_ratio * string_impedance * bow_speed``. Velocity is in
    m/s; position/width are fractions of the vibrating length. A stopped bow
    retains pressure and dissipates motion. No initial waveform is prescribed.
    ``audit`` measures the string's work/energy budget before output filtering.
    This is one polarization with no moving bridge or thermal rosin model.
    """
    if (not isinstance(n, int) or isinstance(n, bool) or n < 0
            or not isinstance(sr, int) or isinstance(sr, bool) or sr < 4000
            or not np.isfinite(freq) or not 20 <= freq < sr/4):
        raise ValueError('bowed string needs integer n/rate, sr >= 4000 and 20 <= freq < sr/4')
    if not isinstance(modes, int) or isinstance(modes, bool) or not 8 <= modes <= 128:
        raise ValueError('modes must be an integer from 8 to 128')
    if (not isinstance(oversample, int) or isinstance(oversample, bool)
            or oversample not in (1, 2, 4, 8, 16, 32, 64)):
        raise ValueError('oversample must be a power of two from 1 to 64')
    positive = (soft_a, attack, bandwidth, tension, length, sustain, treble)
    if any(not np.isfinite(v) or v <= 0 for v in positive):
        raise ValueError('bow model times, friction scale, band, tension and length must be finite and positive')
    if any(not np.isfinite(v) or v < 0 for v in (bow_speed, force_ratio, width, stiffness)):
        raise ValueError('bow speed, force, width and stiffness must be finite and nonnegative')
    if (not np.isfinite(position) or position-width/2 <= 0 or position+width/2 >= 1):
        raise ValueError('whole bow contact must lie inside the vibrating string')
    if pressure_attack is not None and (not np.isfinite(pressure_attack) or pressure_attack <= 0):
        raise ValueError('pressure_attack must be finite and positive or None')
    gate = n/sr if gate is None else gate
    if not np.isfinite(gate) or not 0 <= gate <= n/sr:
        raise ValueError('bow gate must lie inside the requested duration')
    impedance = tension*np.sqrt(1+stiffness)/(2*length*freq)
    normal = force_ratio*impedance*bow_speed
    factor = oversample
    while True:
        if bandwidth < .42*sr*factor:
            model = _model(float(freq), sr, factor, modes, bandwidth, stiffness, tension,
                           length, sustain, treble, position, width)
            margin = 1-model.mobility*normal*2/np.e*np.sqrt(2*soft_a)
            if margin > 0:
                break
        if factor == 64:
            raise ValueError('bow contact requires more than 64x integration; reduce force/band or friction slope')
        factor *= 2
    x, v = np.zeros(len(model.coupling)), np.zeros(len(model.coupling))
    output = np.empty(n*factor)
    work = friction = damping = balance_error = 0.
    for i in range(len(output)):
        time = i*model.dt
        speed = bow_speed*min(1., time/attack)*min(1., max(0., (gate-time)/.06))
        force_scale = normal*(min(1., time/pressure_attack) if pressure_attack is not None else 1.)
        free_x = model.p11*x+model.p12*v
        free_v = model.p21*x+model.p22*v
        free_speed = float(.5*model.coupling@(v+free_v))
        force, _ = _soft_contact(speed-free_speed, model.mobility, force_scale, soft_a)
        next_x, next_v = free_x+model.qx*force, free_v+model.qv*force
        output[i] = model.bridge@next_x
        if audit:
            midpoint_v = (v+next_v)/2
            loss = model.dt*float(np.sum(2*model.gamma*midpoint_v**2))
            bow_work = model.dt*force*speed
            contact_loss = model.dt*force*(speed-float(model.coupling@midpoint_v))
            delta = float(.5*(next_x@next_x+next_v@next_v-x@x-v@v))
            balance_error = max(balance_error, abs(delta-bow_work+contact_loss+loss))
            if contact_loss < -1e-12:
                raise ArithmeticError('soft contact created relative energy')
            work += bow_work
            friction += contact_loss
            damping += loss
        x, v = next_x, next_v
    wave = resample_poly(output, 1, factor)[:n] if n else np.zeros(0)
    diagnostics: BowDiagnostics = {
        'mode_count': len(x), 'oversample': factor, 'uniqueness_margin': float(margin),
        'initial_energy': 0., 'final_energy': float(.5*(x@x+v@v)),
        'bow_work': work if audit else None, 'friction_loss': friction if audit else None,
        'damping_loss': damping if audit else None,
        'maximum_energy_balance_error': balance_error if audit else None,
    }
    return dc_block(wave, sr, 15.) if n else wave, diagnostics
