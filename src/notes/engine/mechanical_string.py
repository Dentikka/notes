"""Finite mechanical string/body model, including exact non-proportional damping.

y(x) = sum(q_h sin(pi*h*x)) + b*(1-x). The bridge displacement b is
an additional mechanical coordinate. M, K and C are symmetric positive
(semi)definite; the energy-coordinate state matrix is dissipative.
No recorded excitation, measured IR or stored waveform is used.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

import numpy as np
from scipy.linalg import eig, eigh, solve
from scipy.optimize import linear_sum_assignment

__all__ = ["modal_string"]


class _State(TypedDict):
    M: np.ndarray
    K: np.ndarray
    C: np.ndarray
    h: np.ndarray
    U: np.ndarray
    omega: np.ndarray
    D: np.ndarray
    A: np.ndarray
    poles: np.ndarray
    V: np.ndarray
    positive: np.ndarray
    string_poles: np.ndarray


def _mechanics(bare: float, *, modes: int = 64, stiffness: float = 4e-5,
              body_hz: float = 115., body_q: float = 20., peak: float = .02,
              tension: float = 70., length: float = .65, sustain: float = 9.,
              treble_decay: float = 2.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = np.arange(1, modes+1, dtype=float)
    mass = tension*(1+stiffness)/(4*length*bare*bare)
    hz = bare*h*np.sqrt((1+stiffness*h*h)/(1+stiffness))
    M = np.diag(np.full(modes, mass/2))
    K = np.diag(mass/2*(2*np.pi*hz)**2)
    t60 = sustain*(220/bare)**.4
    gamma = np.log(1000)*(1/t60 + max(0, 1/treble_decay-1/t60)*np.maximum(0, hz*hz-bare*bare) /
                         max(1, 4000**2-bare*bare))
    C = np.diag(mass*gamma)
    if peak > 0:
        omega = 2*np.pi*body_hz
        body_mass = body_q/(peak*omega)
        M = np.pad(M, ((0, 1), (0, 1)))
        M[-1, -1] = body_mass+mass/3
        M[:-1, -1] = M[-1, :-1] = mass/(np.pi*h)
        K = np.pad(K, ((0, 1), (0, 1)))
        K[-1, -1] = body_mass*omega*omega+tension/length
        C = np.pad(C, ((0, 1), (0, 1)))
        C[-1, -1] = body_mass*omega/body_q
    return M, K, C, h


@lru_cache(maxsize=128)
def _state(bare: float, modes: int, stiffness: float, body_hz: float, body_q: float, peak: float,
          tension: float, length: float, sustain: float, treble_decay: float) -> _State:
    M, K, C, h = _mechanics(bare, modes=modes, stiffness=stiffness, body_hz=body_hz, body_q=body_q,
                           peak=peak, tension=tension, length=length, sustain=sustain, treble_decay=treble_decay)
    omega2, U = eigh(K, M)
    omega = np.sqrt(omega2)
    D = U.T@C@U
    zero = np.zeros_like(D)
    A = np.block([[zero, np.diag(omega)], [-np.diag(omega), -D]])
    poles, V = eig(A)
    positive = np.flatnonzero(poles.imag > 0)
    if len(positive) != len(omega):
        raise ValueError('pitched modal string requires underdamped modes; increase sustain or bridge Q')
    # Associate damped pole pairs with the undamped modal energy coordinates.
    energy = abs(V[:len(omega), positive])**2 + abs(V[len(omega):, positive])**2
    row, col = linear_sum_assignment(-energy)
    pole_for_mode = np.empty(len(omega), dtype=int)
    pole_for_mode[row] = positive[col]
    original_participation = np.diag(M)[:modes, None]*U[:modes]**2
    string_modes = np.argmax(original_participation, axis=1)
    string_poles = poles[pole_for_mode[string_modes]]
    result: _State = {'M': M, 'K': K, 'C': C, 'h': h, 'U': U, 'omega': omega, 'D': D,
                      'A': A, 'poles': poles, 'V': V, 'positive': positive, 'string_poles': string_poles}
    for value in result.values():
        value.flags.writeable = False
    return result


def _tuned_state(freq: float, *, modes: int = 64, stiffness: float = 4e-5,
                body_hz: float = 115., body_q: float = 20., peak: float = .02,
                tension: float = 70., length: float = .65, sustain: float = 9.,
                treble_decay: float = 2.5) -> tuple[float, _State]:
    bare = freq
    for _ in range(12):
        model = _state(float(bare), modes, stiffness, body_hz, body_q, peak,
                      tension, length, sustain, treble_decay)
        observed = model['string_poles'][0].imag/(2*np.pi)
        if abs(1200*np.log2(observed/freq)) < .001:
            return float(bare), model
        bare *= freq/observed
    raise ValueError('could not tune the string-like mode away from the body resonance')


def _point(h: np.ndarray, position: float, *, body: bool, relative: bool = False,
          width: float = 0., spacing: float = 0.) -> np.ndarray:
    values = np.sin(np.pi*h*position)*np.sinc(h*width/2)*np.cos(np.pi*h*spacing/2)
    return np.r_[values, -position if relative else 1-position] if body else values


def _initial_state(model: _State, pick: float, cutoff: float | None,
                   release_motion: float = 0.) -> np.ndarray:
    f = _point(model['h'], pick, body=len(model['M']) > len(model['h']))
    if cutoff is not None:
        rigid_hz = np.sqrt(np.diag(model['K'])[:len(model['h'])]/np.diag(model['M'])[:len(model['h'])])/(2*np.pi)
        f[:len(model['h'])] /= 1+(rigid_hz/cutoff)**2
    static = solve(model['K'], f, assume_a='pos')
    amplitude = model['U'].T@model['M']@static
    position = model['omega']*amplitude
    moving = amplitude*np.linalg.norm(position)/np.linalg.norm(amplitude)
    velocity = np.sign(release_motion)*np.sqrt(abs(release_motion))*moving
    position = np.sqrt(1-abs(release_motion))*position
    return np.r_[position, velocity]


def _output(model: _State, pickup: float, width: float, spacing: float, relative: bool) -> np.ndarray:
    readout = _point(model['h'], pickup, width=width, spacing=spacing,
                    body=len(model['M']) > len(model['h']), relative=relative)
    return np.r_[np.zeros(len(model['omega'])), readout@model['U']]


def modal_string(freq: float, n: int, sr: int, *, pick: float = .15, pickup: float = .06,
                 width: float = .04, spacing: float = 0., relative: bool = True,
                 cutoff: float | None = 6400., rounding: str = 'release', release_motion: float = 0.,
                 modes: int = 64, stiffness: float = 4e-5, body_hz: float = 125., body_q: float = 20.,
                 peak: float = .01, tension: float = 70., length: float = .65,
                 sustain: float = 9., treble_decay: float = 2.5) -> np.ndarray:
    """Magnetic velocity readout of a finite, dissipative string/bridge model.

    ``release_motion`` is the signed fraction of initial energy in a velocity
    profile proportional to the static deflection; zero releases a held pluck,
    +/-1 starts at zero deflection with opposite velocities. Total initial
    mechanical energy is preserved when changing this control.

    ``modes`` bounds the string basis; modes near/above the audio band are omitted.
    This is a steady-pitch source, with no recording or measured response input.
    """
    if (not isinstance(n, int) or n < 0 or not isinstance(sr, int) or sr < 4000
            or not np.isfinite(freq) or not 20 <= freq < sr/4):
        raise ValueError('modal string needs integer length/rate, sr >= 4000 and 20 <= freq < sr/4')
    if not isinstance(modes, int) or isinstance(modes, bool) or not 8 <= modes <= 128:
        raise ValueError('modes must be an integer from 8 to 128')
    positive = (body_hz, body_q, tension, length, sustain, treble_decay)
    if any(not np.isfinite(value) or value <= 0 for value in positive) or body_hz >= sr*.44:
        raise ValueError('mechanical frequencies, Q, tension, length and decays must be finite and positive')
    if any(not np.isfinite(value) or value < 0 for value in (stiffness, peak, width, spacing)):
        raise ValueError('stiffness, mobility and pickup widths must be finite and nonnegative')
    if (not np.isfinite(pick) or not 0 < pick < 1 or not np.isfinite(pickup) or not 0 < pickup < 1
            or pickup-(width+spacing)/2 < 0 or pickup+(width+spacing)/2 > 1):
        raise ValueError('pluck and whole pickup aperture must lie inside the vibrating string')
    if not np.isfinite(release_motion) or not -1 <= release_motion <= 1:
        raise ValueError('release motion must be between -1 and 1')
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0):
        raise ValueError('pluck cutoff must be finite and positive')
    if n == 0:
        return np.zeros(0)
    h = np.arange(1, modes+1)
    hz = freq*h*np.sqrt((1+stiffness*h*h)/(1+stiffness))
    count = max(2, int(np.sum(hz < sr*.42)))
    _, model = _tuned_state(freq, modes=count, stiffness=stiffness, body_hz=body_hz, body_q=body_q,
                            peak=peak, tension=tension, length=length, sustain=sustain, treble_decay=treble_decay)
    if rounding not in ('spatial', 'release'):
        raise ValueError('rounding must be spatial or release')
    z0 = _initial_state(model, pick, cutoff if rounding == 'spatial' else None,
                        release_motion)
    readout = _output(model, pickup, width, spacing, relative)
    weights = (readout@model['V'])*solve(model['V'], z0)
    selected = model['positive']
    selected = selected[model['poles'][selected].imag < 2*np.pi*.44*sr]
    time = np.arange(n)/sr
    wave = np.zeros(n)
    a0, a1 = 0., 0.
    for j in selected:
        weight = weights[j]
        if rounding == 'release' and cutoff is not None:
            corner = 2*np.pi*cutoff
            weight *= corner**2/(model['poles'][j]+corner)**2
            a0 += 2*weight.real
            a1 += 2*(weight*model['poles'][j]).real
        wave += 2*(weight*np.exp(model['poles'][j]*time)).real
    if rounding == 'release' and cutoff is not None:
        corner = 2*np.pi*cutoff
        wave += (-a0+(-a1-corner*a0)*time)*np.exp(-corner*time)
    return wave
