"""Experimental passive multi-string piano contact; no recorded excitation."""
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from functools import lru_cache
from math import ceil

import numpy as np
from scipy.linalg import eig, expm, solve
from scipy.optimize import brentq
from scipy.signal import resample_poly

from notes.engine.palm_contact import _sample


def _number(value: object, name: str, low: float, high: float | None = None,
            *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite number') from error
    if (isinstance(value, (bool, np.bool_)) or not np.isfinite(result) or result < low
            or (positive and result <= 0) or (high is not None and result > high)):
        raise ValueError(f'{name} is outside its supported finite range')
    return result


def _integer(value: object, name: str, low: int, high: int) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _cents(value: object) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 3:
        raise ValueError('unison_cents needs one to three offsets in [-50,50]')
    return tuple(sorted(_number(x, 'unison_cents', -50, 50) for x in value))


@dataclass(frozen=True, kw_only=True)
class _Parameters:
    unison_cents: tuple[float, ...] = (-1.4, 0., 1.4)
    modes: int = 64
    mechanical_band: float = 17640.
    stiffness: float = .00012
    sustain: float = 8.
    brightness: float = .65
    hammer_mass_scale: float = 1.
    felt_stiffness_scale: float = 1.
    contact_oversample: int = 1
    radiation: bool = True
    radiation_loss_scale: float = 1.
    bridge_mass: float | None = None
    bridge_hz: float = 1500.
    bridge_q: float = 30.

    def __post_init__(self) -> None:
        object.__setattr__(self, 'unison_cents', _cents(self.unison_cents))
        object.__setattr__(self, 'modes', _integer(self.modes, 'modes', 1, 64))
        object.__setattr__(self, 'contact_oversample', _integer(self.contact_oversample, 'contact_oversample', 1, 16))
        for name, low, high in [('stiffness', 0, 1), ('sustain', .0001, 120), ('brightness', 0, 1),
                                ('mechanical_band', 100, 48000),
                                ('hammer_mass_scale', .25, 4), ('felt_stiffness_scale', .01, 16),
                                ('radiation_loss_scale', 0, 4), ('bridge_hz', 20, 20000), ('bridge_q', 1, 1000)]:
            object.__setattr__(self, name, _number(getattr(self, name), name, low, high, positive=name == 'sustain'))
        if self.bridge_mass is not None:
            object.__setattr__(self, 'bridge_mass', _number(self.bridge_mass, 'bridge_mass', 1e-4, 10))
        if not isinstance(self.radiation, (bool, np.bool_)):
            raise ValueError('radiation must be boolean')
        object.__setattr__(self, 'radiation', bool(self.radiation))
        if not self.radiation and self.radiation_loss_scale != 1:
            raise ValueError('radiation_loss_scale requires radiation=True')


@dataclass(frozen=True)
class _Model:
    M: np.ndarray
    K: np.ndarray
    C: np.ndarray
    B: np.ndarray
    readout: np.ndarray
    A: np.ndarray
    poles: np.ndarray
    V: np.ndarray
    Lv: np.ndarray
    Lq: np.ndarray
    LF: np.ndarray
    rate: int
    highest: float
    minimum_mass_eigenvalue: float
    eigenbasis_condition: float
    observation_factor: int


@lru_cache(maxsize=16)
def _model(freq: float, sr: int, p: _Parameters) -> _Model:
    offsets = np.array(p.unison_cents)
    strings = len(offsets)
    h = np.arange(1, p.modes+1)
    frequencies = freq*2**(offsets[:, None]/1200)*h*np.sqrt((1+p.stiffness*h*h)/(1+p.stiffness))
    if np.max(frequencies[:, 0]) >= .4*sr:
        raise ValueError('unison fundamental exceeds the observation band')
    keep = np.all(frequencies < p.mechanical_band, axis=0)
    if not keep.any():
        raise ValueError('mechanical band must include every fundamental')
    h, frequencies = h[keep], frequencies[:, keep]
    modes, count = len(h), frequencies.size
    omega = 2*np.pi*frequencies.ravel()
    mass = .012*(220/freq)**.35
    modal_mass = mass/strings
    body = p.bridge_mass is not None
    size = count+int(body)
    M = np.diag(np.r_[np.full(count, modal_mass), p.bridge_mass] if body else np.full(count, modal_mass))
    K = np.diag(np.r_[modal_mass*omega**2, p.bridge_mass*(2*np.pi*p.bridge_hz)**2]
                if body else modal_mass*omega**2)
    register = float(np.clip(np.log2(freq/110)/2.6, 0, 1)) if p.radiation else 0.
    t60 = p.sustain*(220/freq)**.28/(1+.045*(h-1)**1.2)
    t60 = np.broadcast_to(t60, frequencies.shape)/(1+p.radiation_loss_scale*register*(frequencies/1400)**1.2)
    C = np.diag(np.r_[2*modal_mass*np.log(1000)/t60.ravel(), p.bridge_mass*2*np.pi*p.bridge_hz/p.bridge_q]
                if body else 2*modal_mass*np.log(1000)/t60.ravel())
    B, readout = np.zeros((size, strings)), np.zeros(size)
    shape = np.sin(np.pi*h*.135)
    tap = np.sin(np.pi*h*.965)*h**(-1.2*register)
    for j in range(strings):
        sl = slice(j*modes, (j+1)*modes)
        B[sl, j], readout[sl] = shape, tap/strings
        if body:
            cross = 2*modal_mass*(-1.)**(h+1)/(np.pi*h)
            M[sl, -1], M[-1, sl] = cross, cross
            M[-1, -1] += 2*modal_mass/3
            K[-1, -1] += 8*modal_mass*frequencies[j, 0]**2/(1+p.stiffness)
            B[-1, j] = .135
    minimum = float(np.linalg.eigvalsh(M).min())
    if minimum <= 0 or not np.isfinite(M).all() or not np.isfinite(K).all() or not np.isfinite(C).all():
        raise ValueError('unison parameters produce invalid mechanical matrices')
    A = np.block([[np.zeros_like(M), np.eye(size)], [-solve(M, K, assume_a='pos'), -solve(M, C, assume_a='pos')]])
    poles, V = eig(A)
    highest = float(np.max(abs(poles.imag))/(2*np.pi))
    observation_factor = max(1, ceil(2.5*highest/sr))
    if observation_factor > 16 or (body and p.bridge_hz >= p.mechanical_band):
        raise ValueError('coupled modes exceed the supported physical/observation band')
    if np.max(poles.real) >= 0:
        raise ValueError('mechanical poles are not strictly decaying')
    rate = max(sr*observation_factor, 176400)*p.contact_oversample
    dt = 1/rate
    D = M+dt*C/2+dt*dt*K/4
    Lv = solve(D, M-dt*C/2-dt*dt*K/4, assume_a='pos')
    Lq, LF = solve(D, -dt*K, assume_a='pos'), solve(D, dt*B, assume_a='pos')
    result = _Model(M, K, C, B, readout, A, poles, V, Lv, Lq, LF, rate, highest, minimum,
                    float(np.linalg.cond(V)), observation_factor)
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return result


def _force_gradient(a: np.ndarray, b: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete gradient of the compression-only convex felt potential."""
    power = 2.3
    delta, mid = b-a, (a+b)/2
    near = abs(delta) < 1e-4*np.maximum(np.maximum(abs(a), abs(b)), 1e-12)
    force, derivative = np.zeros_like(a), np.zeros_like(a)
    use = ~near
    pa = k*np.maximum(a, 0)**(power+1)/(power+1)
    pb = k*np.maximum(b, 0)**(power+1)/(power+1)
    force[use] = (pb[use]-pa[use])/delta[use]
    derivative[use] = (k*np.maximum(b[use], 0)**power-force[use])/delta[use]
    active = near & (mid > 0)
    force[active] = k*mid[active]**power*(1+power*(power-1)/24*(delta[active]/mid[active])**2)
    derivative[active] = k*power*mid[active]**(power-1)/2
    return force, np.maximum(derivative, 0)


@dataclass(frozen=True)
class _Contact:
    initial: np.ndarray
    head: np.ndarray
    seconds: float
    initial_energy: float
    remaining_energy: float
    dissipated: float
    balance_error: float
    iterations: int
    centroid: float | None


def _contact(model: _Model, freq: float, velocity: float, p: _Parameters,
             centroid_prior: float | None = None, hammer_speed: float | None = None) -> _Contact:
    size, strings = len(model.M), len(p.unison_cents)
    dt, hammer_mass = 1/model.rate, .007*(220/freq)**.15*p.hammer_mass_scale
    felt = 2e9*(.45+p.brightness)*p.felt_stiffness_scale/strings
    q, v = np.zeros(size), np.zeros(size)
    hx, hv = 0., .6+3.4*velocity if hammer_speed is None else hammer_speed
    initial_energy = .5*hammer_mass*hv*hv
    if initial_energy <= np.finfo(float).tiny:
        raise ValueError('hammer_speed produces an energy below the supported numerical range')
    compliance = dt/2*(model.B.T@model.LF)+dt*dt/(2*hammer_mass)*np.ones((strings, strings))
    head: list[float] = []
    dissipated, maximum, touched, iterations = 0., 0., False, 0
    energy = initial_energy
    for _ in range(round(.05*model.rate)):
        head.append(float(model.readout@v))
        a = hx-model.B.T@q
        vf = model.Lv@v+model.Lq@q
        qf = q+dt/2*(v+vf)
        free = hx+dt*hv-model.B.T@qf
        b = free.copy()
        for _iteration in range(16):
            force, derivative = _force_gradient(a, b, felt)
            residual = b-free+compliance@force
            if np.max(abs(residual)) < 1e-15:
                break
            b -= np.linalg.solve(np.eye(strings)+compliance*derivative[None, :], residual)
        else:
            raise ValueError('shared hammer contact failed to converge')
        iterations = max(iterations, _iteration+1)
        force, _ = _force_gradient(a, b, felt)
        vn, qn = vf+model.LF@force, qf+dt/2*(model.LF@force)
        hvn = hv-dt*float(force.sum())/hammer_mass
        hxn = hx+dt/2*(hv+hvn)
        dissipated += dt*float(((v+vn)/2)@model.C@((v+vn)/2))
        q, v, hx, hv = qn, vn, hxn, hvn
        potential = float(np.sum(felt*np.maximum(hx-model.B.T@q, 0)**3.3/3.3))
        energy = .5*float(v@model.M@v+q@model.K@q)+.5*hammer_mass*hv*hv+potential
        maximum = max(maximum, abs(energy+dissipated-initial_energy)/initial_energy)
        touched |= bool(np.max(force) > 0)
        if touched and np.all(hx-model.B.T@q <= 0):
            break
    else:
        raise ValueError('hammer did not separate within the supported contact interval')
    if maximum > 1e-8:
        raise ValueError('shared hammer energy balance failed')
    initial = np.r_[q, v]
    centroid = None
    if model.eigenbasis_condition <= 1e10:
        fundamental = np.zeros(size)
        modes = (size-int(p.bridge_mass is not None))//strings
        fundamental[np.arange(strings)*modes] = 1/strings
        out = np.r_[np.zeros(size), fundamental]
        residues = (out@model.V)*solve(model.V, initial)
        hz = model.poles.imag/(2*np.pi)
        weights = abs(residues)*np.exp(model.poles.real*.1)
        prior = freq if centroid_prior is None else centroid_prior
        near = (hz > .5*prior) & (hz < 1.5*prior) & (weights > max(weights)*1e-8)
        if near.any():
            centroid = float(np.sum(hz[near]*weights[near])/np.sum(weights[near]))
    return _Contact(initial, np.array(head), len(head)/model.rate, initial_energy, energy, dissipated,
                    maximum, iterations, centroid)


def _observe_at_rate(model: _Model, contact: _Contact, n: int, sr: int) -> np.ndarray:
    state = {'A': model.A, 'poles': model.poles, 'V': model.V}
    out = np.r_[np.zeros(len(model.M)), model.readout]
    # Keep the same future support for short and long requested prefixes.
    complete_head = int(np.ceil(contact.seconds*sr))+16
    count = min(n, complete_head)
    support = max(1, int(np.ceil((complete_head/sr-contact.seconds)*model.rate))+int(np.ceil(32*model.rate/sr)))
    extension = _sample(state, contact.initial, out, support, model.rate, None, 'spatial')
    ratio = Fraction(sr, model.rate)
    integrated = resample_poly(np.r_[contact.head, extension], ratio.numerator, ratio.denominator)
    wave = np.empty(n)
    wave[:count] = integrated[:count]
    if count < n:
        shifted = expm(model.A*(count/sr-contact.seconds))@contact.initial
        wave[count:] = _sample(state, shifted, out, n-count, sr, None, 'spatial')
    return wave*.11


def _observe(model: _Model, contact: _Contact, n: int, sr: int) -> np.ndarray:
    factor = model.observation_factor
    guard = 16*factor if factor > 1 else 0
    wave = _observe_at_rate(model, contact, n*factor+guard, sr*factor)
    return resample_poly(wave, 1, factor)[:n] if factor > 1 else wave


def struck_unisons(freq: float, n: int, sr: int, velocity: float, *,
                   unison_cents: tuple[float, ...] = (-1.4, 0., 1.4), modes: int = 64,
                   mechanical_band: float = 17640.,
                   stiffness: float = .00012, sustain: float = 8., brightness: float = .65,
                   hammer_mass_scale: float = 1., felt_stiffness_scale: float = 1.,
                   contact_oversample: int = 1, radiation: bool = True, radiation_loss_scale: float = 1.,
                   bridge_mass: float | None = None, bridge_hz: float = 1500., bridge_q: float = 30.,
                   pitch_compensation: bool = False, hammer_speed: float | None = None,
                   _audit: dict[str, object] | None = None) -> np.ndarray:
    """Separate strings struck by one compression-only hammer, optionally on a moving bridge.

    Offsets describe one to three strings, including coincident ones. Total modal
    mass and felt stiffness are divided across contacts, preserving the one-string
    limit. Positive damping and a discrete-gradient contact retain energy balance.
    The output is a designed velocity tap, not identified acoustic radiation.
    The physical modal band is fixed independently of output rate, bounded by
    modes. Observation is oversampled and decimated when needed; factors above
    16 raise instead of aliasing. Every fundamental must fit below .4*sr.
    Contact refinement retains the mechanical basis.
    The current model requires hammer separation within 50 ms.

    Optional compensation adjusts the bare frequency using generated fundamental-
    coordinate residues weighted after 0.1 s, with a fixed requested-note region.
    It then observes the requested sample clock once. No reference waveform,
    measured phase, IR or fitted output envelope is accepted. None of these controls
    establishes a realistic general piano preset. Output gate/release is external.
    Optional hammer_speed sets initial hammer speed in model m/s, from 0 to 4,
    before nonlinear contact. None retains .6+3.4*velocity exactly. This changes
    contact spectrum and energy, not just output gain; zero velocity stays silent.
    """
    sr, n = _integer(sr, 'sample rate', 4000, 192000), _integer(n, 'sample count', 0, 100_000_000)
    freq, velocity = _number(freq, 'frequency', 20, .4*sr), _number(velocity, 'velocity', 0, 1)
    if hammer_speed is not None:
        hammer_speed = _number(hammer_speed, 'hammer_speed', 0, 4)
    if not isinstance(pitch_compensation, (bool, np.bool_)):
        raise ValueError('pitch_compensation must be boolean')
    p = _Parameters(unison_cents=unison_cents, modes=modes, mechanical_band=mechanical_band,
        stiffness=stiffness, sustain=sustain,
        brightness=brightness, hammer_mass_scale=hammer_mass_scale, felt_stiffness_scale=felt_stiffness_scale,
        contact_oversample=contact_oversample, radiation=radiation, radiation_loss_scale=radiation_loss_scale,
        bridge_mass=bridge_mass, bridge_hz=bridge_hz, bridge_q=bridge_q)
    maximum_fundamental = freq*2**(max(p.unison_cents)/1200)
    if maximum_fundamental >= .4*sr or maximum_fundamental >= p.mechanical_band:
        raise ValueError('fundamentals must fit the physical and observation band')
    if p.bridge_mass is not None and p.bridge_hz >= p.mechanical_band:
        raise ValueError('bridge frequency must fit the mechanical band')
    if n == 0 or velocity == 0 or hammer_speed == 0:
        if _audit is not None:
            _audit.update(silent=True)
        return np.zeros(n)
    tuning = []
    evaluated: dict[float, tuple[_Model, _Contact, float | None]] = {}

    def evaluate(bare: float) -> float | None:
        if bare in evaluated:
            return evaluated[bare][2]
        model = _model(float(bare), sr, p)
        contact = _contact(model, float(bare), velocity, p, freq, hammer_speed)
        actual = contact.centroid
        error = float(1200*np.log2(actual/freq)) if actual is not None else None
        tuning.append({'bare_frequency': bare, 'generated_centroid': actual, 'error_cents': error})
        evaluated[bare] = (model, contact, error)
        return error

    bare = freq
    error = evaluate(bare)
    if pitch_compensation and (error is None or abs(error) >= .05):
        if error is None:
            raise ValueError('no oscillatory fundamental is available for pitch compensation')
        def objective(value: float) -> float:
            result = evaluate(value)
            if result is None:
                raise ValueError('no oscillatory fundamental is available for pitch compensation')
            return result

        low = .8*freq
        high = min(1.25*freq, .399*sr/2**(max(p.unison_cents)/1200))
        left, right = objective(low), objective(high)
        if left*right > 0:
            raise ValueError('could not bracket the generated unison centroid')
        bare = float(brentq(objective, low, high, xtol=freq*1e-8, rtol=1e-10, maxiter=25))
        if abs(objective(bare)) >= .05:
            raise ValueError('generated unison centroid is discontinuous or did not converge')
    model, contact, error = evaluated[bare]
    # Place the actual render's parameters last; a root solver may evaluate other
    # points before returning a previously evaluated root.
    if tuning[-1]['bare_frequency'] != bare:
        tuning.append({'bare_frequency': bare, 'generated_centroid': contact.centroid, 'error_cents': error})
    if _audit is not None:
        _audit.update(strings=len(p.unison_cents),
            modes=(len(model.M)-int(p.bridge_mass is not None))//len(p.unison_cents),
            contact_seconds=contact.seconds, initial_energy=contact.initial_energy,
            remaining_energy=contact.remaining_energy, dissipated_contact_energy=contact.dissipated,
            max_relative_balance_error=contact.balance_error, max_contact_iterations=contact.iterations,
            highest_observed_mode_hz=model.highest, minimum_mass_eigenvalue=model.minimum_mass_eigenvalue,
            largest_free_pole_real=float(np.max(model.poles.real)), tuning=tuning)
        _audit['observation_factor'] = model.observation_factor
    return _observe(model, contact, n, sr)
