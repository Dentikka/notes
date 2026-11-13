"""Passive string/body mechanics for experimental bow-contact solvers.

The sine string coordinates share a moving endpoint with positive-mass body
modes. Full non-proportional damping is retained. This is a numerical building
block, not a calibrated acoustic instrument or a measured impulse response.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh, solve

BodyModes = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Mechanics:
    M: np.ndarray
    K: np.ndarray
    C: np.ndarray
    U: np.ndarray
    physical_omega: np.ndarray
    force: np.ndarray
    readout: np.ndarray
    body_count: int


@dataclass(frozen=True)
class Midpoint:
    mechanics: Mechanics
    P: np.ndarray
    Q: np.ndarray
    omega: np.ndarray
    D: np.ndarray
    coupling: np.ndarray
    readout: np.ndarray
    body_velocity: np.ndarray
    mobility: float
    dt: float


def _finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be finite') from error
    if not np.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def mechanics(freq: float, *, modes: int = 128, bandwidth: float = 18000., stiffness: float = 1e-5,
              tension: float = 110., length: float = .69, sustain: float = 12., treble: float = 1.,
              position: float = .08, width: float = .012, bodies: BodyModes = ()) -> Mechanics:
    """Construct physical M/K/C before choosing any output or integration rate.

    Body entries are (frequency Hz, Q, peak velocity mobility m/s/N). Their
    displacement coordinates add at the string endpoint. The readout is the
    finite-basis endpoint gradient, not an exact reaction force or far-field
    pressure. Input and velocity are collocated for mechanical work accounting.
    """
    if not isinstance(modes, int) or isinstance(modes, bool) or not 8 <= modes <= 128:
        raise ValueError('modes must be an integer from 8 to 128')
    names = ('frequency', 'bandwidth', 'stiffness', 'tension', 'length', 'sustain', 'treble', 'position', 'width')
    values = tuple(_finite(k, v) for k, v in zip(names, (freq, bandwidth, stiffness, tension, length,
                                                        sustain, treble, position, width), strict=True))
    freq, bandwidth, stiffness, tension, length, sustain, treble, position, width = values
    if freq < 20 or min(bandwidth, tension, length, sustain, treble) <= 0:
        raise ValueError('frequency must be >=20Hz and physical scales positive')
    if min(stiffness, width) < 0 or not 0 < position-width/2 <= position+width/2 < 1:
        raise ValueError('stiffness/width must be nonnegative and the whole contact inside the string')
    if not isinstance(bodies, (tuple, list)) or len(bodies) > 8:
        raise ValueError('bodies must contain at most eight frequency/Q/mobility triples')
    normalized = []
    for mode in bodies:
        if not isinstance(mode, (tuple, list)) or len(mode) != 3:
            raise ValueError('body modes need frequency/Q/mobility triples')
        f, q, peak = (_finite('body mode', v) for v in mode)
        if not 0 < f < bandwidth or min(q, peak) <= 0:
            raise ValueError('body frequency/Q/mobility must be positive and frequency below the physical band')
        normalized.append((f, q, peak))
    return _mechanics(freq, modes, bandwidth, stiffness, tension, length, sustain, treble,
                      position, width, tuple(normalized))


@lru_cache(maxsize=32)
def _mechanics(freq: float, modes: int, bandwidth: float, stiffness: float, tension: float, length: float,
               sustain: float, treble: float, position: float, width: float, bodies: BodyModes) -> Mechanics:
    h = np.arange(1, modes+1, dtype=float)
    hz = freq*h*np.sqrt((1+stiffness*h*h)/(1+stiffness))
    h, hz = h[hz < bandwidth], hz[hz < bandwidth]
    if len(h) < 2:
        raise ValueError('physical band needs at least two string modes')
    count, body_count = len(h), len(bodies)
    total_mass = tension*(1+stiffness)/(4*length*freq*freq)
    gamma = np.log(1000)*(1/sustain+max(0, 1/treble-1/sustain)*(hz/freq-1)**2
                          / max(1, (4000/freq-1)**2))
    M = np.diag(np.r_[np.full(count, total_mass/2), np.zeros(body_count)])
    K = np.diag(np.r_[total_mass/2*(2*np.pi*hz)**2, np.zeros(body_count)])
    C = np.diag(np.r_[total_mass*gamma, np.zeros(body_count)])
    if body_count:
        for j, (body_hz, body_q, peak) in enumerate(bodies):
            omega = 2*np.pi*body_hz
            mass = body_q/(peak*omega)
            M[count+j, count+j] = mass
            K[count+j, count+j] = mass*omega**2
            C[count+j, count+j] = mass*omega/body_q
        M[:count, count:] = total_mass/(np.pi*h[:, None])
        M[count:, :count] = M[:count, count:].T
        M[count:, count:] += total_mass/3
        K[count:, count:] += tension/length
    values, U = eigh(K, M)
    if values.min() <= 0:
        raise ValueError('mechanical stiffness must be positive')
    force = np.r_[np.sin(np.pi*h*position)*np.sinc(h*width/2), np.full(body_count, 1-position)]
    readout = np.r_[tension*np.pi*h*(1+stiffness*h*h)/length, np.full(body_count, -tension/length)]
    omega = np.sqrt(values)
    for array in (M, K, C, U, force, readout, omega):
        array.flags.writeable = False
    return Mechanics(M, K, C, U, omega, force, readout, body_count)


def midpoint(m: Mechanics, sr: int, factor: int) -> Midpoint:
    """Dissipative midpoint operators with coupled-modal frequency prewarping.

    The conserved numerical energy uses the warped stiffness, which converges
    to physical K under refinement. M/K/C and the modal basis do not depend on
    output rate. Observation filters are outside this energy budget.
    """
    if not isinstance(sr, int) or isinstance(sr, bool) or sr < 4000:
        raise ValueError('sample rate must be an integer >=4000')
    if not isinstance(factor, int) or isinstance(factor, bool) or factor not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError('integration factor must be a power of two from 1 to 64')
    rate = sr*factor
    if m.physical_omega.max()/(2*np.pi) >= .42*rate:
        raise ValueError('integration rate does not resolve the physical coupled band')
    omega = 2*rate*np.tan(m.physical_omega/(2*rate))
    D = m.U.T@m.C@m.U
    D = (D+D.T)/2
    n = len(omega)
    zero = np.zeros_like(D)
    A = np.block([[zero, np.diag(omega)], [-np.diag(omega), -D]])
    coupling = m.U.T@m.force
    forcing = np.r_[np.zeros(n), coupling]
    dt = 1/rate
    identity = np.eye(2*n)
    left = identity-dt*A/2
    P = solve(left, identity+dt*A/2)
    Q = solve(left, dt*forcing)
    readout = (m.readout@m.U)/omega
    body_velocity = m.U[-m.body_count:].sum(axis=0) if m.body_count else np.zeros(n)
    mobility = float(.5*coupling@Q[n:])
    if mobility <= 0:
        raise ArithmeticError('collocated midpoint mobility must be positive')
    for array in (P, Q, omega, D, coupling, readout, body_velocity):
        array.flags.writeable = False
    return Midpoint(m, P, Q, omega, D, coupling, readout, body_velocity, mobility, dt)
