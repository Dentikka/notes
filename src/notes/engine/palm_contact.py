"""Passive finite-area viscous contact for the mechanical plucked-string source."""
from __future__ import annotations

from functools import lru_cache
from math import ceil

import numpy as np
from scipy.linalg import eig, eigh, expm, solve
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

from notes.engine.articulation import _partial_gains, _partial_t60_steps, _partial_t60s
from notes.engine.mechanical_string import _initial_state, _mechanics, _output, _point, _State


@lru_cache(maxsize=32)
def _state(bare: float, modes: int, stiffness: float, body_hz: float, body_q: float, peak: float,
           tension: float, length: float, sustain: float, treble_decay: float,
           position: float, width: float, resistance: float,
           partial_t60s: tuple[float | None, ...] = ()) -> _State:
    M, K, C, h = _mechanics(bare, modes=modes, stiffness=stiffness, body_hz=body_hz, body_q=body_q,
                           peak=peak, tension=tension, length=length, sustain=sustain, treble_decay=treble_decay,
                           partial_t60s=partial_t60s)
    if width:
        nodes, weights = np.polynomial.legendre.leggauss(32)
        shapes = np.column_stack([_point(h, position+width*x/2, body=peak > 0) for x in nodes])
        C = C+resistance*(shapes*(weights/2))@shapes.T
    elif resistance:
        shape = _point(h, position, body=peak > 0)
        C = C+resistance*np.outer(shape, shape)
    if peak == 0:
        U = np.diag(1/np.sqrt(np.diag(M)))
        omega = np.sqrt(np.diag(K)/np.diag(M))
    else:
        omega2, U = eigh(K, M)
        omega = np.sqrt(omega2)
    D = U.T@C@U
    zero = np.zeros_like(D)
    A = np.block([[zero, np.diag(omega)], [-np.diag(omega), -D]])
    poles, V = eig(A)
    positive = np.flatnonzero(poles.imag > 0)
    real = np.flatnonzero(poles.imag == 0)
    candidates = np.r_[positive, real]
    energy = abs(V[:len(omega), candidates])**2+abs(V[len(omega):, candidates])**2
    row, col = linear_sum_assignment(-energy)
    assigned = np.empty(len(omega), dtype=int)
    assigned[row] = candidates[col]
    participation = np.diag(M)[:modes, None]*U[:modes]**2
    string_modes = np.argmax(participation, axis=1)
    result: _State = {'M': M, 'K': K, 'C': C, 'h': h, 'U': U, 'omega': omega, 'D': D, 'A': A,
                     'poles': poles, 'V': V, 'positive': positive,
                     'string_poles': poles[assigned[string_modes]]}
    for value in result.values():
        value.flags.writeable = False
    return result


def _sample(model: _State, initial: np.ndarray, output: np.ndarray,
            n: int, sr: int, cutoff: float | None, rounding: str) -> np.ndarray:
    """Sample the exact modal evolution, retaining nonoscillatory overdamped poles."""
    if np.linalg.cond(model['V']) > 1e10:
        A, z, readout = model['A'], initial, output
        if rounding == 'release' and cutoff is not None:
            c = 2*np.pi*cutoff
            A = np.pad(A, ((0, 2), (0, 2)))
            A[-2, :-2], A[-2, -2], A[-1, -2], A[-1, -1] = c*output, -c, c, -c
            z = np.r_[initial, 0., 0.]
            readout = np.r_[np.zeros(len(initial)+1), 1.]
        step = expm(A/sr)
        wave = np.empty(n)
        for k in range(n):
            wave[k] = readout@z
            z = step@z
        return wave
    weights = (output@model['V'])*solve(model['V'], initial)
    poles = model['poles']
    wave = np.empty(n)
    for at in range(0, n, 1024):
        t = np.arange(at, min(n, at+1024))/sr
        if rounding != 'release' or cutoff is None:
            wave[at:at+len(t)] = (weights@np.exp(poles[:, None]*t)).real
            continue
        c = 2*np.pi*cutoff
        delta = poles+c
        near = abs(delta) < 1e-3*c
        far = ~near
        filtered = weights[far]*c*c/delta[far]**2
        y = filtered@np.exp(poles[far, None]*t)
        y -= (filtered.sum()+t*np.sum(filtered*delta[far]))*np.exp(-c*t)
        for j in np.flatnonzero(near):
            # Stable divided difference at a pole coincident with the release filter.
            use = c*t < 700
            tt = t[use]
            q = delta[j]*tt
            kernel = tt*tt*(.5+q/6+q*q/24+q**3/120+q**4/720)
            y[use] += weights[j]*c*c*np.exp(-c*tt)*kernel
        wave[at:at+len(t)] = y.real
    return wave


def _sample_damping_steps(model: _State, changes: list[tuple[float, _State]],
                          initial: np.ndarray, output: np.ndarray, n: int, sr: int,
                          cutoff: float | None, rounding: str) -> np.ndarray:
    """Keep mechanical and optional release-filter states across loss changes."""
    filtered = rounding == 'release' and cutoff is not None
    mechanical_size = len(initial)
    state = np.r_[initial, 0., 0.] if filtered else initial.copy()
    wave = np.empty(n)
    at = 0
    for time, following in [*changes, (n/sr, model)]:
        end = min(n, round(time*sr))
        count = end-at
        if count > 0:
            part = _sample(model, state[:mechanical_size], output, count, sr, cutoff, rounding)
            A = model['A']
            if filtered:
                c = 2*np.pi*cutoff
                t = np.arange(count)/sr
                part += (state[-1]+c*state[-2]*t)*np.exp(-c*t)
                A = np.pad(A, ((0, 2), (0, 2)))
                A[-2, :-2], A[-2, -2], A[-1, -2], A[-1, -1] = c*output, -c, c, -c
            wave[at:end] = part
            state = expm(A*(count/sr))@state
            at = end
        if at == n:
            break
        model = following
    return wave


def palm_string(freq: float, n: int, sr: int, *, position: float, contact_width: float,
                resistance: float, mechanical_band: float = 8000., pick: float = .15,
                pickup: float = .06, pickup_width: float = .04, spacing: float = 0.,
                relative: bool = True, cutoff: float | None = 6400., rounding: str = 'release',
                release_motion: float = 0., modes: int = 64, stiffness: float = 4e-5,
                body_hz: float = 125., body_q: float = 20., peak: float = .01,
                tension: float = 70., length: float = .65, sustain: float = 9.,
                treble_decay: float = 2.5, partial_gains: tuple[float, ...] = (),
                partial_t60s: tuple[float | None, ...] = (),
                partial_t60_steps: tuple[tuple[float, tuple[float | None, ...]], ...] = ()) -> np.ndarray:
    """Mechanical velocity with a stationary, uniformly distributed viscous contact.

    Contact coordinates are fractions of the vibrating length; resistance is the
    total effective N*s/m over that region. Zero width selects a point contact.
    This is a passive linear approximation, not an identified hand/string model.
    The physical modal band is independent of output rate, bounded by ``modes``.
    Low output rates use an oversampled observation and polyphase decimation.
    Optional ``partial_gains`` changes physical string-coordinate readout only;
    contact mechanics, initial energy and the bridge readout remain unchanged.
    ``partial_t60s`` overrides isolated string-coordinate damping times before
    adding the positive viscous contact. None entries retain the original losses;
    the contact/body can change the actual coupled pole decay times.
    ``partial_t60_steps`` replaces the loss tuple at onset-relative seconds.
    None/missing entries restore intrinsic sustain/treble losses; an empty row
    restores all of them. States, mechanical energy and release-filter memory
    remain continuous across the switch; M/K and the initial pluck do not change.
    Changes are applied on the internal observation grid, with no recorded audio.
    """
    gains = _partial_gains(partial_gains)
    decays = _partial_t60s(partial_t60s)
    steps = _partial_t60_steps(partial_t60_steps)
    changes = []
    for time, replacement in steps:
        if time == 0:
            decays = replacement
        elif replacement != (changes[-1][1] if changes else decays):
            changes.append((time, replacement))
    losses = {'partial_t60s': decays} if decays else {}
    if (not isinstance(n, int) or isinstance(n, bool) or n < 0 or not isinstance(sr, int)
            or isinstance(sr, bool) or sr < 4000 or not np.isfinite(freq) or not 20 <= freq < sr*.44):
        raise ValueError('palm string needs integer length/rate, sr>=4000 and 20<=freq<.44*sr')
    if not isinstance(modes, int) or isinstance(modes, bool) or not 8 <= modes <= 128:
        raise ValueError('modes must be an integer from 8 to 128')
    if any(not np.isfinite(x) or x <= 0 for x in (body_hz, body_q, tension, length, sustain,
                                                 treble_decay, mechanical_band)):
        raise ValueError('frequencies, Q, tension, length and decays must be finite and positive')
    if mechanical_band <= freq or (peak > 0 and body_hz >= mechanical_band):
        raise ValueError('mechanical band must include the fundamental and bridge frequency')
    if any(not np.isfinite(x) or x < 0 for x in (stiffness, peak, pickup_width, spacing, resistance, contact_width)):
        raise ValueError('stiffness, mobility, widths and resistance must be finite and nonnegative')
    if (not np.isfinite(position) or not 0 < position < 1 or position-contact_width/2 < 0
            or position+contact_width/2 > 1):
        raise ValueError('whole palm contact must lie inside the vibrating string')
    if (not np.isfinite(pick) or not 0 < pick < 1 or not np.isfinite(pickup) or not 0 < pickup < 1
            or pickup-(pickup_width+spacing)/2 < 0 or pickup+(pickup_width+spacing)/2 > 1):
        raise ValueError('pluck and whole pickup aperture must lie inside the string')
    if not np.isfinite(release_motion) or not -1 <= release_motion <= 1:
        raise ValueError('release motion must be between -1 and 1')
    if rounding not in ('spatial', 'release') or (cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0)):
        raise ValueError('rounding must be spatial/release and cutoff must be finite and positive')
    if not n:
        return np.zeros(0)
    bare = freq
    for _ in range(12):
        h = np.arange(1, modes+1)
        hz = bare*h*np.sqrt((1+stiffness*h*h)/(1+stiffness))
        count = max(1, int((hz < mechanical_band).sum()))
        model = _state(float(bare), count, stiffness, body_hz, body_q, peak,
                       tension, length, sustain, treble_decay, position, contact_width, resistance, **losses)
        observed = model['string_poles'][0].imag/(2*np.pi)
        if not np.isfinite(observed) or observed <= 0:
            raise ValueError('palm contact overdamps the assigned fundamental; a pitched source is unavailable')
        if abs(1200*np.log2(observed/freq)) < .05:
            break
        bare *= freq/observed
    else:
        raise ValueError('could not tune the palm-damped string fundamental')
    initial = _initial_state(model, pick, cutoff if rounding == 'spatial' else None, release_motion)
    output = _output(model, pickup, pickup_width, spacing, relative, gains)
    scheduled = [(time, _state(float(bare), count, stiffness, body_hz, body_q, peak,
                              tension, length, sustain, treble_decay, position, contact_width, resistance,
                              partial_t60s=replacement)) for time, replacement in changes]
    highest = max(float(abs(s['poles'].imag).max()/(2*np.pi)) for s in [model, *(s for _, s in scheduled)])
    factor = max(1, ceil(2.5*highest/sr))
    # Future observation support avoids changing a prefix when a longer note is requested.
    guard = 16*factor if factor > 1 else 0
    if scheduled:
        wave = _sample_damping_steps(model, scheduled, initial, output, n*factor+guard, sr*factor, cutoff, rounding)
    else:
        wave = _sample(model, initial, output, n*factor+guard, sr*factor, cutoff, rounding)
    return resample_poly(wave, 1, factor)[:n] if factor > 1 else wave
