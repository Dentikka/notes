"""Magnitude-only excitation matching against an affine, procedurally generated voice.

The reference supplies band energies, never carrier phases. Losses, frequencies,
timing and the renderer's RNG must already be fixed by the caller.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len
from scipy.optimize import least_squares
from scipy.signal import stft


def _mono(value: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError('real mono audio required')
    x = np.asarray(value, dtype=float)
    if x.ndim != 1 or len(x) < 32 or not np.isfinite(x).all():
        raise ValueError('finite mono audio with at least32 samples required')
    return x


@dataclass(frozen=True)
class ExcitationBasis:
    """Generated affine waves: constant row, then one difference row per control.

    Build with excitation_basis, using unmastered procedural audio. Target audio
    is never inserted into these waves. A successful probe checks the supplied
    renderer, not global linearity of every possible setting or physical realism.
    """

    waves: np.ndarray
    linearity_error: float

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.waves):
            raise ValueError('real generated basis required')
        waves = np.array(self.waves, dtype=float, copy=True)
        if (waves.ndim != 2 or not 2 <= waves.shape[0] <= 81 or waves.shape[1] < 32
                or not np.isfinite(waves).all() or not np.isfinite(self.linearity_error)
                or self.linearity_error < 0):
            raise ValueError('finite generated basis with1..80 controls and nonnegative error required')
        waves.flags.writeable = False
        object.__setattr__(self, 'waves', waves)

    def audio(self, gains: Sequence[float]) -> np.ndarray:
        """Reconstruct generated mono audio, without gain normalization or limiting."""
        g = np.asarray(gains, dtype=float)
        if g.shape != (len(self.waves) - 1,) or not np.isfinite(g).all() or np.any(g < 0):
            raise ValueError('one finite nonnegative gain per generated control required')
        return np.r_[1., g] @ self.waves


def excitation_basis(
    render: Callable[[tuple[float, ...]], np.ndarray], *, controls: int = 12,
    tolerance: float = 1e-8,
) -> ExcitationBasis:
    """Build an affine basis with controls+2 deterministic render calls.

    render(gains) must synthesize the same known note, source losses/frequencies,
    sample count and fresh identically seeded RNG each time. Zero and unit control
    vectors are used; a non-unity probe rejects nonlinear or nondeterministic
    callbacks. Use the dry floating voice before master loudness normalization,
    clipping or an amplifier. Include the linear pickup/insert filters used by
    the actual renderer. For ElectricGuitar this fits excitation_gains, not
    independent output partial amplitudes. Existing defaults are untouched.
    """
    if (isinstance(controls, bool) or not isinstance(controls, int) or not 1 <= controls <= 80
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('1..80 controls and positive finite tolerance required')
    zero = np.zeros(controls)
    constant = _mono(render(tuple(zero))).copy()
    components = []
    for i in range(controls):
        g = zero.copy()
        g[i] = 1.
        x = _mono(render(tuple(g)))
        if x.shape != constant.shape:
            raise ValueError('generated voice lengths must match')
        components.append(x - constant)
    waves = np.vstack([constant, *components])
    if not np.any(waves[1:]):
        raise ValueError('renderer does not respond to excitation controls')
    probe = .2 + 1.5 * (np.arange(controls) + .5) / controls
    actual = _mono(render(tuple(probe)))
    if actual.shape != constant.shape:
        raise ValueError('generated voice lengths must match')
    error = float(np.linalg.norm(np.r_[1., probe] @ waves - actual) /
                  max(float(np.linalg.norm(actual)), 1e-30))
    if error > tolerance:
        raise ValueError(f'renderer failed affine excitation probe: relative error={error:.3g}')
    return ExcitationBasis(waves, error)


def _quadratic_bands(z: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Exact real band-power forms for real affine coefficients; no target phases."""
    out = np.empty((len(masks), z.shape[-1], z.shape[0], z.shape[0]))
    for i, mask in enumerate(masks):
        block = z[:, mask, :].transpose(2, 0, 1)
        out[i] = (block.conj() @ block.swapaxes(1, 2)).real
    return out


@dataclass(frozen=True)
class ExcitationGainFit:
    """Conditional numeric controls; convergence is not a realism certificate."""

    gains: tuple[float, ...]
    cost: float
    initial_cost: float
    optimizer_converged: bool
    optimizer_message: str
    nfev: int
    evaluations: int
    supported_cells: int
    active_bounds: tuple[int, ...]
    linearity_error: float


def fit_excitation_gains(
    reference: np.ndarray, basis: ExcitationBasis, sr: int, *,
    reference_hz: Sequence[float], model_hz: Sequence[float],
    start: float = .14, end: float = .65,
    normalize_start: float = .14, normalize_end: float = .3,
    window_seconds: float = .12, step: float = .02, fft_size: int | None = None,
    initial: Sequence[float] | None = None, bounds: tuple[float, float] = (.001, 16.),
    max_nfev: int = 40, regularization: float = .05, log_weighting: str = 'uniform',
) -> ExcitationGainFit:
    """Fit dry excitation shape with fixed generated phase, poles and one early level.

    reference_hz and model_hz must label corresponding, increasing supported
    components on each signal's own frequencies. No pitch, onset, losses, velocity
    or physical geometry is identified. Audio lengths and clocks must match.
    Frame centres are on the original audio clock; neither alignment nor later
    per-frame level matching is performed. Reject silent references.

    The objective combines log band powers(.25), linear band amplitudes(30), and
    log errors(3) on the first controlled components where the reference exceeds
    -35dB relative to early20..8000Hz power. Six coarse bands have weight.3; other
    components weight1. The floor is -45dB. These are calibration settings, not a
    validated perceptual metric. Supplied frequency/support assumptions need
    independent validation, including held time windows and new executions.

    Exact band-power quadratic forms are built once from the generated basis.
    Only reference magnitudes enter fitting. Returns the lowest observed physical
    MAE, including the initial proposal; regularization guides bounded log-gain
    LS, while its convergence flag/message refer to the optimizer itself.

    ``log_weighting='amplitude'`` weights log residuals by the reference band
    amplitude relative to early total power. Linear-amplitude residuals retain
    their original weights. This reduces the influence of many weak components
    when their errors conflict with dominant components. It is an optional
    calibration criterion, not a validated perceptual metric; costs from the two
    criteria are not directly comparable. ``'uniform'`` preserves the old result.
    """
    ref = _mono(reference)
    if log_weighting not in ('uniform', 'amplitude'):
        raise ValueError("log_weighting must be 'uniform' or 'amplitude'")
    if (not isinstance(basis, ExcitationBasis) or len(ref) != basis.waves.shape[1]
            or isinstance(sr, bool) or not isinstance(sr, int) or sr < 1000):
        raise ValueError('matching reference/basis length and integer sample rate>=1000 required')
    values = [start, end, normalize_start, normalize_end, window_seconds, step, regularization]
    if (not np.isfinite(values).all() or min(start, normalize_start, regularization) < 0
            or end <= start or normalize_end <= normalize_start or not 0 < step <= window_seconds):
        raise ValueError('increasing fit/normalization windows, valid step and nonnegative regularization required')
    rh, mh = np.asarray(reference_hz, dtype=float), np.asarray(model_hz, dtype=float)
    if (rh.ndim != 1 or not 1 <= len(rh) <= 80 or mh.shape != rh.shape
            or not np.isfinite(np.r_[rh, mh]).all() or min(rh.min(), mh.min()) <= 0
            or max(rh.max(), mh.max()) >= .45 * sr or np.any(np.diff(rh) <= 0) or np.any(np.diff(mh) <= 0)):
        raise ValueError('matching increasing supported component frequencies inside0..0.45sr required')
    limits = np.asarray(bounds, dtype=float)
    if (limits.shape != (2,) or not np.isfinite(limits).all() or not 0 < limits[0] < limits[1]
            or isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1):
        raise ValueError('positive increasing gain bounds and integer positive evaluation budget required')
    count = len(basis.waves) - 1
    g0 = np.ones(count) if initial is None else np.asarray(initial, dtype=float)
    if g0.shape != (count,) or not np.isfinite(g0).all() or np.any(g0 < limits[0]) or np.any(g0 > limits[1]):
        raise ValueError('initial gains must fit the declared bounds')
    nperseg, hop = round(window_seconds * sr), round(step * sr)
    if nperseg < 16 or hop < 1 or nperseg > len(ref):
        raise ValueError('analysis window and step must fit the audio')
    if fft_size is None:
        fft_size = next_fast_len(6 * nperseg)
    if isinstance(fft_size, bool) or not isinstance(fft_size, int) or fft_size < nperseg:
        raise ValueError('integer FFT size must cover the spectral window')

    def transform(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return stft(x, sr, nperseg=nperseg, noverlap=nperseg - hop, nfft=fft_size, boundary=None, padded=False)

    # Analyze only the prefix needed by the declared windows, not an entire song.
    samples = min(len(ref), round(max(end, normalize_end) * sr) + nperseg // 2 + hop)
    hz, time, target_z = transform(ref[:samples])
    active = (time >= start) & (time <= end)
    early = (time >= normalize_start) & (time <= normalize_end)
    if active.sum() < 3 or early.sum() < 2 or max(end, normalize_end) > time[-1]:
        raise ValueError('fit needs three frames and normalization two frames inside the available audio')
    select = active | early
    active, early = active[select], early[select]
    width = min(.4 * min(rh[0], mh[0]), max(2 / window_seconds, .15 * min(rh[0], mh[0])))
    broad = np.array([(hz >= lo) & (hz < hi) for lo, hi in
                      [(20, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]])
    rm = np.vstack([abs(hz[None, :] - rh[:, None]) <= width, broad])
    mm = np.vstack([abs(hz[None, :] - mh[:, None]) <= width, broad])
    target_power = rm.astype(float) @ abs(target_z[:, select]) ** 2
    target_scale = float(np.median(target_power[-6:, early].sum(axis=0)))
    if target_scale <= 1e-30:
        raise ValueError('reference has no usable early band energy')
    target_power = np.maximum(target_power[:, active] / target_scale, 10 ** (-45 / 10))
    target_log, target_amp = 10 * np.log10(target_power), np.sqrt(target_power)
    controlled = min(count, len(rh))
    support = target_log[:controlled] >= -35.
    if not support.any():
        raise ValueError('reference has no supported controlled components')
    bins = mm.any(axis=0)
    z = np.empty((len(basis.waves), int(bins.sum()), int(select.sum())), dtype=complex)
    for i, wave in enumerate(basis.waves):
        z[i] = transform(wave[:samples])[2][bins][:, select]
    gram = _quadratic_bands(z, mm[:, bins])
    del z
    weights = np.r_[np.ones(len(rh)), np.full(6, .3)][:, None]
    best_cost, best_gains, initial_cost = float('inf'), g0.copy(), 0.
    evaluations = 0

    def residual(log_gains: np.ndarray) -> np.ndarray:
        nonlocal best_cost, best_gains, initial_cost, evaluations
        gains = np.exp(log_gains)
        coefficients = np.r_[1., gains]
        power = np.maximum(np.einsum('i,btij,j->bt', coefficients, gram, coefficients, optimize=True), 0.)
        scale = max(float(np.median(power[-6:, early].sum(axis=0))), 1e-30)
        power = np.maximum(power[:, active] / scale, 10 ** (-45 / 10))
        log = (10 * np.log10(power) - target_log) * weights
        if log_weighting == 'amplitude':
            log *= target_amp
        amp = (np.sqrt(power) - target_amp) * weights
        physical = np.r_[.25 * log.ravel(), 30. * amp.ravel(), (3. * log[:controlled] * support).ravel()]
        cost = float(np.mean(abs(physical)))
        if evaluations == 0:
            initial_cost = cost
        evaluations += 1
        if cost < best_cost:
            best_cost, best_gains = cost, gains.copy()
        return np.r_[physical, regularization * (log_gains - np.log(g0))]

    result = least_squares(residual, np.log(g0), bounds=(np.log(limits[0]), np.log(limits[1])),
                           max_nfev=max_nfev, diff_step=.001, loss='soft_l1', f_scale=3.)
    active_bounds = tuple(int(i) for i in np.flatnonzero(
        np.isclose(best_gains, limits[0], rtol=1e-4) | np.isclose(best_gains, limits[1], rtol=1e-4)))
    return ExcitationGainFit(tuple(map(float, best_gains)), best_cost, initial_cost, bool(result.success),
                             str(result.message), int(result.nfev), evaluations, int(support.sum()), active_bounds,
                             basis.linearity_error)
