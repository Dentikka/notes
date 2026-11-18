"""Shared effective pluck geometry from modal magnitudes, never recorded phases."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from notes.ears.partial_envelopes import PartialEnvelopes
from notes.engine.articulation import _partial_t60s
from notes.engine.dsp import biquad
from notes.engine.instruments.strings import fret_scale
from notes.engine.mechanical_string import _mechanics, _point


@dataclass(frozen=True, kw_only=True)
class ModalPluckObservation:
    """First consecutive string-mode envelopes of one known, steady anchor note."""

    pitch: float
    string: int
    f0: float
    stiffness: float
    envelopes: PartialEnvelopes
    partial_t60s: tuple[float | None, ...] = ()
    valid: np.ndarray | None = None


@dataclass(frozen=True)
class ModalPluckTrial:
    pick_reference: str
    pick_position: float
    pickup_position: float
    pick_cutoff: float
    cost: float
    nfev: int
    converged: bool
    continued: bool = False
    initialization: str = 'local'


@dataclass(frozen=True)
class ModalPluckGeometryFit:
    """Conditional source controls, with convergence and boundary diagnostics."""

    pick_reference: str
    pick_position: float
    pickup_position: float
    pick_cutoff: float
    cost: float
    converged: bool
    observations: int
    supported_cells: int
    active_bounds: tuple[str, ...]
    jacobian_condition: float | None
    trials: tuple[ModalPluckTrial, ...]
    model_sample_rates: tuple[int, ...] = ()

    def note_controls(self, pitch: float, string: int | None = None) -> dict[str, float | int]:
        """Per-note open-length controls for ModalString(fretting=True).

        Set its constructor pickup_position to this fit's pickup_position.
        Vibrating-reference picks must be converted separately for each note.
        """
        if not np.isfinite(pitch):
            raise ValueError('note pitch must be finite')
        if string is not None:
            if isinstance(string, (bool, np.bool_)) or not isinstance(string, (int, np.integer)):
                raise ValueError('string must be an integer1..6')
            string = int(string)
        scale = fret_scale(pitch, string)
        controls: dict[str, float | int] = {
            'pick_position': self.pick_position*(scale if self.pick_reference == 'vibrating' else 1.),
            'pick_cutoff': self.pick_cutoff}
        if string is not None:
            controls['string'] = string
        return controls


@dataclass
class _Data:
    scale: float
    time: np.ndarray
    target: np.ndarray
    support: np.ndarray
    h: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray
    fixed: np.ndarray
    attenuation: np.ndarray


def fit_modal_pluck(
    observations: Sequence[ModalPluckObservation], *, sustain: float = 9., treble_decay: float = 2.5,
    pickup_width: float = .04, electrical_cutoff: float = 4000., electrical_q: float = .7,
    start: float = .2, end: float = .64, max_partials: int = 12, audibility_db: float = 45.,
    pick_reference: str = 'open', pick_bounds: tuple[float, float] = (.03, .3),
    pickup_bounds: tuple[float, float] = (.02, .12), cutoff_bounds: tuple[float, float] = (2000., 12000.),
    max_nfev: int = 100, polish_nfev: int = 200,
    coarse_seeds: bool = True,
    model_sample_rate: int | None = None,
) -> ModalPluckGeometryFit:
    """Fit three shared controls for the uncoupled, spatially rounded ModalString.

    Supply at least two different known pitches, correctly labelled first
    consecutive string modes measured on their own supported frequencies, and
    any already calibrated isolated-coordinate loss times. The model assumes
    bridge/contact zero, standard fret geometry, a point pluck, uniform pickup
    aperture and the stated electrical lowpass. It neither infers those other
    controls nor estimates B, losses, loudness, timing or recorded phases.

    Shape is L2-normalized per frame; only trajectory-valid, coefficient-supported
    and audible magnitudes enter a robust log-amplitude fit. The white coefficient
    floor is not a general noise classifier. Input tracks/support are assumptions.
    "auto" compares open/vibrating pick references by training cost only.
    Three deterministic starts plus coarse spatial/bandwidth seeds reduce nodal
    local minima. The grid is outside the per-local-run evaluation budget; disable
    it explicitly for a quick/local or legacy-reproduction fit. One optional
    continuation refines the best failed start; all trials remain visible.

    Positions are effective and may not be unique: open-length point excitation
    and pickup factors can exchange roles. Convergence/cost/bounds do not certify
    physical identification, transfer or perceptual realism. No waveform, IR or
    phase is returned. Refit/validate residual readout gains separately if needed.
    By default the electrical model uses each observation's analysis rate.
    Set model_sample_rate to the intended renderer rate when analysis audio was
    resampled; the discrete electrical filter otherwise has a different response.
    """
    if pick_reference not in ('open', 'vibrating', 'auto'):
        raise ValueError('pick_reference must be open, vibrating or auto')
    if not isinstance(coarse_seeds, bool):
        raise ValueError('coarse_seeds must be boolean')
    if model_sample_rate is not None and (
            isinstance(model_sample_rate, bool) or not isinstance(model_sample_rate, int) or model_sample_rate < 4000):
        raise ValueError('model_sample_rate must be an integer >=4000 or None')
    if (not np.isfinite([sustain, treble_decay, pickup_width, electrical_cutoff, electrical_q,
                         start, end, audibility_db]).all()
            or min(sustain, treble_decay, electrical_cutoff, electrical_q, audibility_db) <= 0
            or pickup_width < 0 or start < 0 or end <= start):
        raise ValueError('finite positive source controls and an increasing nonnegative window required')
    if (isinstance(max_partials, bool) or not isinstance(max_partials, int) or not 3 <= max_partials <= 80
            or isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1
            or isinstance(polish_nfev, bool) or not isinstance(polish_nfev, int) or polish_nfev < 0):
        raise ValueError('3..80 partials and valid integer evaluation budgets required')
    bounds = np.asarray([pick_bounds, pickup_bounds, cutoff_bounds], dtype=float)
    if (bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] <= 0)
            or np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(bounds[:2, 1] >= 1)
            or bounds[1, 0] < pickup_width/2):
        raise ValueError('increasing positive bounds and an aperture inside the string required')
    try:
        observations = list(observations)
    except TypeError as error:
        raise ValueError('a sequence of observations required') from error
    if len(observations) < 2 or any(not isinstance(o, ModalPluckObservation) for o in observations):
        raise ValueError('at least two observations at different pitches required')
    try:
        pitches = [float(o.pitch) for o in observations]
    except (ValueError, TypeError) as error:
        raise ValueError('finite known pitches required') from error
    if not np.isfinite(pitches).all():
        raise ValueError('finite known pitches required')
    if len(set(pitches)) < 2:
        raise ValueError('at least two observations at different pitches required')
    data = []
    model_rates: set[int] = set()
    for o, pitch in zip(observations, pitches, strict=True):
        try:
            f0, stiffness = float(o.f0), float(o.stiffness)
        except (ValueError, TypeError) as error:
            raise ValueError('finite known f0 and B required') from error
        if (not np.isfinite([f0, stiffness]).all() or stiffness < 0
                or isinstance(o.string, (bool, np.bool_)) or not isinstance(o.string, (int, np.integer))
                or not 1 <= o.string <= 6
                or not isinstance(o.envelopes, PartialEnvelopes)):
            raise ValueError('finite known pitches, B, string1..6 and PartialEnvelopes required')
        p = o.envelopes
        try:
            t, a, floor, ratios = (np.asarray(value, dtype=float) for value in (
                p.time, p.amplitude, p.white_noise_floor, p.ratios))
        except (ValueError, TypeError) as error:
            raise ValueError('finite envelope arrays required') from error
        supported = np.asarray(p.above_white_noise_floor)
        if (not isinstance(p.sample_rate, int) or p.sample_rate < 4000 or not 20 <= f0 < p.sample_rate/4
                or t.ndim != 1 or not len(t) or np.any(np.diff(t) <= 0) or t[0] < 0
                or a.ndim != 2 or a.shape[0] != len(t) or a.shape[1] < 3 or floor.shape != a.shape
                or supported.shape != a.shape or supported.dtype.kind != 'b'
                or ratios.shape != (a.shape[1],) or not np.isfinite(ratios).all()
                or np.any(ratios <= 0) or np.any(np.diff(ratios) <= 0)
                or not np.isfinite(t).all() or not np.isfinite(a).all()
                or not np.isfinite(floor).all() or np.any(a < 0) or np.any(floor < 0) or end > t[-1]):
            raise ValueError('valid finite envelope arrays and a fully measured fit window required')
        valid = np.ones_like(supported, dtype=bool) if o.valid is None else np.asarray(o.valid)
        if valid.shape != a.shape or valid.dtype.kind != 'b':
            raise ValueError('valid must be a boolean mask matching the envelope magnitudes')
        k = min(max_partials, a.shape[1])
        selected = (t >= start) & (t <= end)
        a = a[selected, :k]
        if len(a) < 5:
            raise ValueError('at least five measured frames per observation required')
        mask = supported[selected, :k] & valid[selected, :k] & (a > 4*floor[selected, :k])
        mask &= 20*np.log10(np.maximum(a, 1e-20)) > 20*np.log10(max(float(a.max()), 1e-30))-audibility_db
        norm = np.linalg.norm(a, axis=1)
        active = (mask.sum(axis=1) >= 3) & (norm > 1e-20)
        if active.sum() < 5:
            raise ValueError('at least five supported frames with three modes required')
        mask &= active[:, None]
        scale = fret_scale(pitch, int(o.string))
        losses = _partial_t60s(o.partial_t60s)
        model_sr = p.sample_rate if model_sample_rate is None else model_sample_rate
        model_rates.add(model_sr)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            M, K, C, h = _mechanics(f0, modes=k, stiffness=stiffness, peak=0., length=.65*scale,
                                   sustain=sustain, treble_decay=treble_decay, partial_t60s=losses)
            omega = np.sqrt(np.diag(K)/np.diag(M))
            gamma = np.diag(C)/(2*np.diag(M))
            if (not np.isfinite(omega).all() or not np.isfinite(gamma).all()
                    or np.any(gamma >= omega) or np.any(omega >= 2*np.pi*.45*min(p.sample_rate, model_sr))):
                raise ValueError('model needs resolved underdamped in-band string modes')
            wd = np.sqrt(omega**2-gamma**2)
            b, c = biquad('lowpass', electrical_cutoff, model_sr, electrical_q)
            z = np.exp((-gamma+1j*wd)/model_sr)
            response = sum(v*z**(-j) for j, v in enumerate(b))/sum(v*z**(-j) for j, v in enumerate(c))
            fixed = abs(response)/(np.diag(M)*wd)
        if not np.isfinite(fixed).all():
            raise ValueError('singular modeled electrical response')
        data.append(_Data(scale, t[selected], a/np.maximum(norm[:, None], 1e-30), mask,
                          h, omega, gamma, fixed, np.exp(-t[selected, None]*gamma[None, :])))
    references = ('open', 'vibrating') if pick_reference == 'auto' else (pick_reference,)
    trials: list[ModalPluckTrial] = []
    solutions = []
    for reference in references:
        actual = bounds.copy()
        shortest = min(d.scale for d in data)
        if reference == 'open':
            actual[0, 1] = min(actual[0, 1], shortest*(1-1e-9))
        actual[1, 1] = min(actual[1, 1], shortest-pickup_width/2-shortest*1e-9)
        if np.any(actual[:, 1] <= actual[:, 0]):
            raise ValueError('no feasible positions within the shortest observed vibrating string')
        lower, upper = actual[:, 0].copy(), actual[:, 1].copy()
        lower[2], upper[2] = np.log(lower[2]), np.log(upper[2])

        def errors(x: np.ndarray, ref: str = reference) -> np.ndarray:
            pick, pickup, cutoff = x[0], x[1], np.exp(x[2])
            result = []
            for d in data:
                pos = pick/d.scale if ref == 'open' else pick
                spatial = abs(_point(d.h, pos, body=False)*_point(
                    d.h, pickup/d.scale, body=False, width=pickup_width/d.scale))
                magnitude = d.fixed*spatial/(1+(d.omega/(2*np.pi*cutoff))**2)
                a = magnitude[None, :]*d.attenuation
                a /= np.maximum(np.linalg.norm(a, axis=1)[:, None], 1e-30)
                delta = 20*np.log10(np.maximum(a, 1e-6))-20*np.log10(np.maximum(d.target, 1e-6))
                result.extend(delta[d.support])
            return np.asarray(result)

        def solve(x: np.ndarray, budget: int, continued: bool = False,
                  initialization: str = 'local', *, ref: str = reference,
                  lo: np.ndarray = lower, hi: np.ndarray = upper,
                  objective: Callable[[np.ndarray], np.ndarray] = errors) -> OptimizeResult:
            fit = least_squares(objective, np.clip(x, lo, hi), bounds=(lo, hi), loss='soft_l1',
                                f_scale=2, max_nfev=budget, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            trial = ModalPluckTrial(ref, float(fit.x[0]), float(fit.x[1]), float(np.exp(fit.x[2])),
                                    float(fit.cost), int(fit.nfev), bool(fit.success), continued, initialization)
            trials.append(trial)
            solutions.append((trial, fit, lo.copy(), hi.copy()))
            return fit

        for p0, u0, cut0 in ((.06, .06, 6000), (.12, .08, 8000), (.208, .0646, 7700)):
            solve(np.array([p0, u0, np.log(cut0)]), max_nfev)
        if coarse_seeds:
            seeds = []
            for p0 in np.linspace(lower[0], upper[0], 33):
                for u0 in np.linspace(lower[1], upper[1], 25):
                    for log_cut0 in np.linspace(lower[2], upper[2], 3):
                        x = np.array([p0, u0, log_cut0])
                        residual = errors(x)
                        cost = float(np.sum(4*(np.sqrt(1+(residual/2)**2)-1)))
                        seeds.append((cost, x))
            chosen: list[np.ndarray] = []
            for _, x in sorted(seeds, key=lambda s: s[0]):
                if any(np.linalg.norm((x[:2]-other[:2])/(upper[:2]-lower[:2])) < .05 for other in chosen):
                    continue
                solve(x, max_nfev, initialization='coarse')
                chosen.append(x)
                if len(chosen) == 6:
                    break
        best = min((s for s in solutions if s[0].pick_reference == reference), key=lambda s: s[0].cost)
        if not best[0].converged and polish_nfev:
            solve(best[1].x, polish_nfev, True, best[0].initialization)
    best, fit, lower, upper = min(solutions, key=lambda s: s[0].cost)
    active_bounds = []
    for i, name in enumerate(('pick_position', 'pickup_position', 'pick_cutoff')):
        if (fit.x[i]-lower[i])/(upper[i]-lower[i]) < 1e-4:
            active_bounds.append(name+':lower')
        if (upper[i]-fit.x[i])/(upper[i]-lower[i]) < 1e-4:
            active_bounds.append(name+':upper')
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-15 else None
    return ModalPluckGeometryFit(best.pick_reference, best.pick_position, best.pickup_position, best.pick_cutoff,
                                best.cost, best.converged, len(data), sum(int(d.support.sum()) for d in data),
                                tuple(active_bounds), condition, tuple(trials), tuple(sorted(model_rates)))
