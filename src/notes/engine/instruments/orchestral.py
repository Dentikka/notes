"""Procedural piano and bowed strings; no recorded samples required."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from notes.engine.articulation import _partial_gain_points, _unison_strings, pitch_curve
from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.dsp import adsr, biquad, filt, vel_gain
from notes.engine.piano_impact import _impact_controls, impact_duration, piano_impact
from notes.engine.registry import register_instrument


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _hammer_speed_points(value: object) -> tuple[tuple[float, float], ...]:
    """Explicit monotone MIDI-velocity to initial-speed calibration; empty is legacy."""
    if not isinstance(value, (tuple, list)) or len(value) > 64:
        raise ValueError('hammer_speed_points needs at most 64 velocity/speed rows')
    result = []
    for row in value:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError('hammer_speed_points rows need velocity and speed')
        try:
            velocity, speed = (float(x) for x in row)
        except (TypeError, ValueError) as error:
            raise ValueError('hammer_speed_points needs finite velocity/speed values') from error
        if not np.isfinite((velocity, speed)).all() or not 0 <= velocity <= 1 or not 0 <= speed <= 4:
            raise ValueError('hammer_speed_points needs velocity in [0,1] and speed in [0,4]')
        if result and (velocity <= result[-1][0] or speed < result[-1][1]):
            raise ValueError('hammer_speed_points velocities must increase and speeds must not decrease')
        result.append((velocity, speed))
    if result and (len(result) < 2 or result[0][0] != 0 or result[-1][0] != 1):
        raise ValueError('hammer_speed_points must cover velocity 0 through 1')
    return tuple(result)


def _body_eq_bands(value: object) -> tuple[tuple[float, float, float], ...]:
    """Validate small, explicit (Hz, Q, dB) equalizer recipes, including JSON lists."""
    if not isinstance(value, (list, tuple)) or len(value) > 8:
        raise ValueError('body_eq needs at most eight (frequency, Q, gain_db) bands')
    result = []
    for band in value:
        if not isinstance(band, (list, tuple)) or len(band) != 3:
            raise ValueError('body_eq bands need (frequency, Q, gain_db)')
        try:
            frequency, q, gain = (float(v) for v in band)
        except (TypeError, ValueError) as error:
            raise ValueError('body_eq bands must contain finite numbers') from error
        if (not np.isfinite((frequency, q, gain)).all() or frequency < 5
                or not .05 <= q <= 50 or not -48 <= gain <= 24):
            raise ValueError('body_eq needs Hz >= 5, .05 <= Q <= 50 and -48 <= gain_db <= 24')
        if gain:
            result.append((frequency, q, gain))
    return tuple(result)


def _body_equalize(wave: np.ndarray, sr: int, bands: tuple[tuple[float, float, float], ...],
                   release: float) -> np.ndarray:
    """Causal per-voice EQ; require its estimated decay to fit the declared release."""
    coefficients = []
    decay = 0.
    for frequency, q, gain in bands:
        if frequency >= sr*.49:
            raise ValueError('body_eq frequency must be below .49 of the sample rate')
        b, a = biquad('peak', frequency, sr, q, gain)
        if np.array_equal(b, a):
            continue
        radius = float(np.max(abs(np.roots(a))))
        if radius >= 1:
            raise ValueError('body_eq filter is not stable')
        decay += np.log(1e-3)/np.log(radius)/sr if radius else 0.
        coefficients.append((b, a))
    if decay > release:
        raise ValueError('body_eq decay exceeds release; increase release or reduce filter Q')
    for b, a in coefficients:
        wave = lfilter(b, a, wave)
    if coefficients and len(wave):
        count = min(len(wave), max(2, round(.002*sr)))
        wave[-count:] *= np.linspace(1., 0., count)
    return wave


@register_instrument("piano")
@dataclass(frozen=True, kw_only=True)
class Piano(Instrument):
    """Hammered inharmonic strings, unison beating and velocity-dependent felt.

    sustain is fundamental T60 at A3. Per-note pedal=True lets the string ring
    after key release; it is a fixed per-note pedal, not a continuous MIDI pedal.
    brightness controls hammer hardness, detune is unison spread in cents.
    model="soundboard" enables hammer-position coloration and two-stage decay;
    model="hammer" integrates a nonlinear felt contact exciting string modes.
    model="hammer_damped" additionally softens the modal readout above A2; it is
    an experimental register correction, not a complete soundboard model.
    attack_presence=0..1 adds a small upper-register onset lift in hammer models;
    the default 0 preserves the accepted hammer_damped timbre.
    hammer_mass_scale=0.25..4 multiplies the hammer mass in the contact models;
    individual notes may override it with the same parameter. Default 1 preserves
    the original impact. This control is unused by legacy and soundboard models.
    Per-note stiffness overrides the inharmonicity coefficient in hammer models;
    it must be finite and nonnegative. Other models retain their existing controls.
    Optional partial_gain_points uses the same time/gain rows as BowedString,
    but addresses dispersive mode groups, including their detuned partners.
    It changes only readout, not contact forces, modal damping or frequencies.
    Nonempty curves are supported only by hammer and hammer_damped models.
    Optional unison_strings=1..3 uses equally weighted free-ring components,
    symmetrically detuned around each mode by up to detune cents. None preserves
    the original unequal two-component readout. This is an equivalent-string
    readout, not a new coupled-string hammer simulation. Notes may override it.
    Optional hammer_speed_points maps velocity 0..1 to initial speed 0..4 model
    m/s, with linear interpolation. Empty retains the original affine mapping.
    It changes nonlinear felt contact, not output gain. Per-note hammer_speed
    overrides the curve; None restores the original mapping for that note.
    felt_stiffness_scale changes felt contact hardness, independently of bending
    stiffness. contact_oversample refines integration without changing modes.
    radiation_compensation is a scalar from synthetic impact-state readout power;
    it preserves relative partials and does not identify physical radiation.
    Optional impact_gain adds a designed nonharmonic modal impact after string
    damping. It is not measured action noise or a coupled soundboard model.
    impact_decay is T60 at 200 Hz; color, velocity power and delay are independent.
    All five controls may be overridden on notes; zero gain keeps the old source.
    legacy preserves the original renderer.
    """

    program: int | None = 0
    model: str = "legacy"
    attack_presence: float = 0.0
    hammer_mass_scale: float = 1.0
    sustain: float = 8.0
    brightness: float = 0.65
    detune: float = 1.4
    unison_strings: int | None = None
    stiffness: float = 0.00012
    hammer: float = 0.035
    release: float = 0.18
    level: float = 0.45
    partial_gain_points: tuple[tuple[float, ...], ...] = ()
    hammer_speed_points: tuple[tuple[float, float], ...] = ()
    felt_stiffness_scale: float = 1.0
    contact_oversample: int = 1
    radiation_compensation: float = 0.0
    impact_gain: float = 0.0
    impact_decay: float = .18
    impact_color: float = .5
    impact_velocity_power: float = 1.0
    impact_delay: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'partial_gain_points', _partial_gain_points(self.partial_gain_points))
        object.__setattr__(self, 'unison_strings', _unison_strings(self.unison_strings))
        object.__setattr__(self, 'hammer_speed_points', _hammer_speed_points(self.hammer_speed_points))
        if self.model not in ("legacy", "soundboard", "hammer", "hammer_damped"):
            raise ValueError("model must be legacy, soundboard, hammer or hammer_damped")
        if self.partial_gain_points and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('partial_gain_points requires hammer or hammer_damped piano')
        if self.unison_strings is not None and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('unison_strings requires hammer or hammer_damped piano')
        if self.hammer_speed_points and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('hammer_speed_points requires hammer or hammer_damped piano')
        self._contact_controls(self.felt_stiffness_scale, self.contact_oversample, self.radiation_compensation)
        self._impact_settings(self.impact_gain, self.impact_decay, self.impact_color,
                              self.impact_velocity_power, self.impact_delay)
        if self.impact_gain and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('impact_gain requires contact piano')
        if ((self.felt_stiffness_scale != 1 or self.contact_oversample != 1)
                and self.model not in ('hammer', 'hammer_damped')):
            raise ValueError('felt_stiffness_scale/contact_oversample require contact piano')
        if self.radiation_compensation and self.model != 'hammer_damped':
            raise ValueError('radiation_compensation requires hammer_damped piano')
        if not np.isfinite(self.attack_presence) or not 0 <= self.attack_presence <= 1:
            raise ValueError("attack_presence must be between 0 and 1")
        if not np.isfinite(self.hammer_mass_scale) or not 0.25 <= self.hammer_mass_scale <= 4:
            raise ValueError("hammer_mass_scale must be between 0.25 and 4")
        _positive(sustain=self.sustain, release=self.release)
        for name in ("brightness", "detune", "stiffness", "hammer", "level"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def release_time(self) -> float:
        tail = max(self.release, self.sustain)
        if self.impact_gain:
            tail = max(tail, impact_duration(self.impact_decay, self.impact_delay))
        return tail

    @staticmethod
    def _impact_settings(gain: float, decay: float, color: float, power: float, delay: float) -> None:
        _impact_controls(decay, color, power, delay)
        if not np.isfinite(gain) or not 0 <= gain <= 1:
            raise ValueError('impact_gain must be finite and in [0,1]')

    def _with_impact(self, wave: np.ndarray, v: Voice, ctx: RenderContext, rng: np.random.Generator,
                     controls: dict[str, float]) -> np.ndarray:
        if controls['gain'] == 0 or v.vel == 0:
            return self.level*wave
        n = max(len(wave), ctx.samples(impact_duration(controls['decay'], controls['delay'])))
        if n > ctx.samples(v.dur+self.release_time())+1:
            raise ValueError('impact tail exceeds configured release time; increase Piano sustain/release')
        impact = piano_impact(n, ctx.sr, v.vel, rng, decay=controls['decay'], color=controls['color'],
                              velocity_power=controls['power'], delay=controls['delay'])
        if n > len(wave):
            wave = np.pad(wave, (0, n-len(wave)))
        return self.level*(wave+controls['gain']*impact)

    @staticmethod
    def _contact_controls(felt: float, oversample: int, compensation: float) -> None:
        if not np.isfinite(felt) or not .25 <= felt <= 16:
            raise ValueError('felt_stiffness_scale must be between .25 and 16')
        if isinstance(oversample, bool) or not isinstance(oversample, int) or not 1 <= oversample <= 16:
            raise ValueError('contact_oversample must be an integer from 1 to 16')
        if not np.isfinite(compensation) or not 0 <= compensation <= 1:
            raise ValueError('radiation_compensation must be between 0 and 1')

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        points = _partial_gain_points(v.params.get('partial_gain_points', self.partial_gain_points))
        strings = _unison_strings(v.params.get('unison_strings', self.unison_strings))
        speeds = _hammer_speed_points(v.params.get('hammer_speed_points', self.hammer_speed_points))
        felt = v.params.get('felt_stiffness_scale', self.felt_stiffness_scale)
        oversample = v.params.get('contact_oversample', self.contact_oversample)
        compensation = v.params.get('radiation_compensation', self.radiation_compensation)
        self._contact_controls(felt, oversample, compensation)
        impact = {name: v.params.get(key, getattr(self, key)) for name, key in (
            ('gain', 'impact_gain'), ('decay', 'impact_decay'), ('color', 'impact_color'),
            ('power', 'impact_velocity_power'), ('delay', 'impact_delay'))}
        self._impact_settings(impact['gain'], impact['decay'], impact['color'], impact['power'], impact['delay'])
        if impact['gain'] and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('impact_gain requires contact piano')
        if (felt != 1 or oversample != 1) and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('felt_stiffness_scale/contact_oversample require contact piano')
        if compensation and self.model != 'hammer_damped':
            raise ValueError('radiation_compensation requires hammer_damped piano')
        if points and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('partial_gain_points requires hammer or hammer_damped piano')
        if strings is not None and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('unison_strings requires hammer or hammer_damped piano')
        if (speeds or v.params.get('hammer_speed') is not None) and self.model not in ('hammer', 'hammer_damped'):
            raise ValueError('hammer_speed requires hammer or hammer_damped piano')
        if self.model in ("hammer", "hammer_damped"):
            from notes.engine.hammer import struck_string

            pedal = bool(v.params.get("pedal", False))
            tail = max(self.release, self.sustain) if pedal else self.release
            n = ctx.samples(v.dur + tail)
            stiffness = self.stiffness
            if "stiffness" in v.params:
                try:
                    stiffness = float(v.params["stiffness"])
                except (TypeError, ValueError) as error:
                    raise ValueError("per-note stiffness must be finite and nonnegative") from error
                if not np.isfinite(stiffness) or stiffness < 0:
                    raise ValueError("per-note stiffness must be finite and nonnegative")
            readout_options = {'partial_gain_points': points} if points else {}
            if strings is not None:
                readout_options['unison_strings'] = strings
            if 'hammer_speed' in v.params:
                if v.params['hammer_speed'] is not None:
                    readout_options['hammer_speed'] = v.params['hammer_speed']
            elif speeds:
                readout_options['hammer_speed'] = float(np.interp(v.vel, *np.asarray(speeds).T))
            if felt != 1:
                readout_options['felt_stiffness_scale'] = felt
            if oversample != 1:
                readout_options['contact_oversample'] = oversample
            if compensation:
                readout_options['radiation_compensation'] = compensation
            y = struck_string(v.freq, n, ctx.sr, v.vel, brightness=self.brightness,
                              stiffness=stiffness, sustain=self.sustain, detune=self.detune,
                              radiation=self.model == "hammer_damped", attack_presence=self.attack_presence,
                              hammer_mass_scale=v.params.get("hammer_mass_scale", self.hammer_mass_scale),
                              **readout_options)
            if not pedal:
                y *= adsr(ctx.samples(v.dur), n, ctx.sr, 0.001, 0.01, 1, self.release)
            count = min(n, ctx.samples(0.01))
            if count:
                y[-count:] *= np.linspace(1, 0, count)
            return self._with_impact(y, v, ctx, rng, impact)
        if self.model == "soundboard":
            return self._soundboard(v, ctx, rng)
        pedal = bool(v.params.get("pedal", False))
        tail = self.release_time() if pedal else self.release
        n = ctx.samples(v.dur + tail)
        t = np.arange(n) / ctx.sr
        y = np.zeros(n)
        tau = self.sustain * (220 / v.freq) ** 0.3 / np.log(1000)
        hardness = max(0.1, self.brightness * (0.25 + 0.75 * v.vel))
        strings = 1 if v.pitch < 33 else (2 if v.pitch < 52 else 3)
        for cents in np.linspace(-self.detune, self.detune, strings) if strings > 1 else [0.0]:
            f0 = v.freq * 2 ** (cents / 1200)
            for k in range(1, min(65, int(ctx.sr * 0.47 / f0) + 1)):
                freq = f0 * k * np.sqrt((1 + self.stiffness * k * k) / (1 + self.stiffness))
                if freq >= ctx.sr * 0.47:
                    break
                amp = np.exp(-(k - 1) / (2 + 13 * hardness)) / k ** 0.85
                decay = np.exp(-t * (1 + 0.16 * (k - 1) ** 1.3) / tau)
                y += amp * decay * np.sin(2 * np.pi * freq * t) / strings
        y *= 1 - np.exp(-t / 0.0015)
        noise = filt(rng.normal(size=n), "lowpass", min(3500, ctx.sr * 0.4), ctx.sr, 0.7)
        y += self.hammer * v.vel * noise * np.exp(-t / 0.009) * (1 - np.exp(-t / 0.0005))
        if not pedal:
            y *= adsr(ctx.samples(v.dur), n, ctx.sr, 0.001, 0.01, 1, self.release)
        if n:
            y[-min(n, ctx.samples(0.01)):] *= np.linspace(1, 0, min(n, ctx.samples(0.01)))
        return self.level * vel_gain(v.vel) * y


    def _soundboard(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        """Designed modal variant: hammer-position comb, two-stage partial decay,
        slightly unequal unison strings and a resonant bridge response.
        """
        pedal = bool(v.params.get("pedal", False))
        tail = self.release_time() if pedal else self.release
        n = ctx.samples(v.dur + tail)
        t = np.arange(n) / ctx.sr
        y = np.zeros(n)
        hardness = max(0.08, self.brightness * (0.2 + 0.8 * v.vel))
        strings = 1 if v.pitch < 33 else (2 if v.pitch < 52 else 3)
        offsets = np.linspace(-self.detune, self.detune, strings) if strings > 1 else [0.0]
        stiffness = self.stiffness * (1 + 0.6 * abs(v.pitch - 60) / 24)
        for index, cents in enumerate(offsets):
            f0 = v.freq * 2 ** (cents / 1200)
            # A few samples between strings, rather than phase-identical oscillators.
            lag = index * 0.00013
            age = np.maximum(t - lag, 0)
            for k in range(1, min(97, int(ctx.sr * 0.46 / f0) + 1)):
                hz = f0 * k * np.sqrt((1 + stiffness * k * k) / (1 + stiffness))
                if hz >= ctx.sr * 0.46:
                    break
                # Fixed hammer position creates uneven partial strengths and missing modes.
                strike = np.sin(np.pi * k * 0.137) / np.sin(np.pi * 0.137)
                felt = np.exp(-(hz / (1600 + 6500 * hardness)) ** 1.4)
                amplitude = strike * felt / k ** 0.72
                amplitude *= 1 + 0.06 * np.sin(k * 2.17 + index)
                # Upper partials persist instead of collapsing to the fundamental.
                t60 = self.sustain * (220 / v.freq) ** 0.28 / (1 + 0.055 * (k - 1) ** 1.15)
                prompt = np.exp(-np.log(1000) * age / (0.24 * t60))
                after = np.exp(-np.log(1000) * age / t60)
                envelope = (0.65 * prompt + 0.35 * after) * (1 - np.exp(-age / 0.0018))
                y += amplitude * envelope * np.sin(2 * np.pi * hz * age) / strings
        # Short fixed bridge resonances colour the entire string, including its attack.
        dry = y.copy()
        for hz, q, gain in ((180, 1.0, 1.4), (520, 1.3, -2.0), (2600, 0.8, 2.5)):
            if hz < ctx.sr * 0.45:
                y = filt(y, "peak", hz, ctx.sr, q, gain)
        y = 0.85 * y + 0.15 * dry
        noise = filt(rng.normal(size=n), "lowpass", min(5500, ctx.sr * 0.4), ctx.sr, 0.7)
        y += self.hammer * noise * np.exp(-t / 0.012) * (1 - np.exp(-t / 0.001))
        if not pedal:
            y *= adsr(ctx.samples(v.dur), n, ctx.sr, 0.001, 0.01, 1, self.release)
        count = min(n, ctx.samples(0.01))
        if count:
            y[-count:] *= np.linspace(1, 0, count)
        return self.level * 0.55 * vel_gain(v.vel) * y


@register_instrument("bowed_string")
@dataclass(frozen=True, kw_only=True)
class BowedString(Instrument):
    """Sustained harmonic bow source shaped by fixed body resonances.

    A procedural approximation, not a nonlinear bow/string simulation.
    pressure controls upper harmonics; bow_noise the friction component. Both
    accept finite nonnegative per-note overrides with the same names. These are
    spectral/noise controls, not measured bow force or an inferred dynamic mark.
    Optional bow_noise_seed (also per note) selects only the noise realization,
    preserving pitch, amplitude modulation and the caller's random stream. None
    keeps the original shared random stream, including its exact draw order.
    bow_noise_highpass/lowpass optionally set noise-only biquad corners in Hz,
    also per note. None keeps the body-scaled highpass or disables the lowpass.
    Explicit corners must be >=10 Hz and <=.45 of the rendering sample rate.
    Optional partial_gain_points rows contain seconds followed by linear gains
    for harmonics 1..H (H<=80). Gains in [0,16] interpolate and hold at endpoints,
    before the common envelope and body_eq. Unspecified harmonics keep gain 1;
    the noise, frequencies and random stream are untouched. Empty keeps the old
    source. Notes may override or disable a configured curve with an empty list.
    Per-note legato=True shortens attack; vibrato/bends/pitch_points are supported.
    Optional body_eq is a tuple of (frequency_Hz, Q, gain_dB) peaking filters,
    overridden per note with the same parameter. Frequencies are absolute, not
    scaled by body_size. Empty preserves the original renderer. This is effective
    output coloration, not an identified mechanical body. The release must cover
    the filters' estimated decay; expression gain is applied afterward by the mixer.
    """

    program: int | None = 40
    body_size: float = 1.0
    pressure: float = 0.55
    bow_noise: float = 0.025
    bow_noise_seed: int | None = None
    bow_noise_highpass: float | None = None
    bow_noise_lowpass: float | None = None
    attack: float = 0.12
    release: float = 0.18
    vibrato: float = 0.12
    vibrato_rate: float = 5.3
    level: float = 0.32
    body_eq: tuple[tuple[float, float, float], ...] = ()
    partial_gain_points: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'body_eq', _body_eq_bands(self.body_eq))
        object.__setattr__(self, 'partial_gain_points', _partial_gain_points(self.partial_gain_points))
        self._validate_noise_seed(self.bow_noise_seed)
        for name in ('bow_noise_highpass', 'bow_noise_lowpass'):
            object.__setattr__(self, name, self._noise_corner(name, getattr(self, name)))
        _positive(body_size=self.body_size, attack=self.attack, release=self.release,
                  vibrato_rate=self.vibrato_rate)
        for name in ("pressure", "bow_noise", "vibrato", "level"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @staticmethod
    def _validate_noise_seed(seed: int | None) -> None:
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64):
            raise ValueError('bow_noise_seed must be an unsigned 64-bit integer or None')

    @staticmethod
    def _noise_corner(name: str, value: float | None, sr: int | None = None) -> float | None:
        if value is None:
            return None
        try:
            corner = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name} must be a finite corner >=10 Hz or None') from error
        if not np.isfinite(corner) or corner < 10 or (sr is not None and corner > .45*sr):
            raise ValueError(f'{name} must be >=10 Hz and <=.45 of the sample rate')
        return corner

    def release_time(self) -> float:
        return self.release

    def voice(self, v: Voice, ctx: RenderContext, rng: np.random.Generator) -> np.ndarray:
        body_eq = _body_eq_bands(v.params.get('body_eq', self.body_eq))
        points = _partial_gain_points(v.params.get('partial_gain_points', self.partial_gain_points))
        partial_points = np.asarray(points) if points else None
        noise_seed = v.params.get('bow_noise_seed', self.bow_noise_seed)
        self._validate_noise_seed(noise_seed)
        corners = {name: self._noise_corner(name, v.params.get(name, getattr(self, name)), ctx.sr)
                   for name in ('bow_noise_highpass', 'bow_noise_lowpass')}
        controls = {}
        for name in ('pressure', 'bow_noise'):
            try:
                value = float(v.params[name]) if name in v.params else getattr(self, name)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{name} must be finite and nonnegative') from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            controls[name] = value
        n = ctx.samples(v.dur) + ctx.samples(self.release)
        t = np.arange(n) / ctx.sr
        params = {"vibrato": self.vibrato, "vibrato_rate": self.vibrato_rate, **v.params}
        curve = pitch_curve(params, n, ctx.sr, rng=rng)
        freq = np.full(n, v.freq) if curve is None else v.freq * 2 ** (curve / 12)
        phase = np.concatenate(([0.0], np.cumsum(freq[:-1] / ctx.sr)))[:n]
        y = np.zeros(n)
        highest = max(v.freq, float(np.max(freq, initial=v.freq)))
        for k in range(1, min(81, int(ctx.sr * 0.47 / highest) + 1)):
            hz = v.freq * k
            body = 0.35 + sum(g / (1 + ((hz - f / self.body_size) / (w / self.body_size)) ** 2)
                              for f, w, g in ((460, 100, 0.9), (550, 90, 0.7), (2400, 700, 1.8)))
            amp = body * np.exp(-(k - 1) / (8 + 30 * controls['pressure'] * v.vel)) / k
            partial = amp * np.sin(2 * np.pi * k * phase)
            if partial_points is not None and k < partial_points.shape[1]:
                partial *= np.interp(t, partial_points[:, 0], partial_points[:, k])
            y += partial
        # Consume the legacy draw even with a separate noise seed: the later
        # amplitude phase and the caller's next voice must keep their old stream.
        excitation = rng.normal(size=n)
        if noise_seed is not None:
            excitation = np.random.default_rng(noise_seed).normal(size=n)
        highpass = corners['bow_noise_highpass']
        if highpass is None:
            highpass = min(1000 / self.body_size, ctx.sr * 0.3)
        noise = filt(excitation, "highpass", highpass, ctx.sr, 0.7)
        if corners['bow_noise_lowpass'] is not None:
            noise = filt(noise, 'lowpass', corners['bow_noise_lowpass'], ctx.sr, 0.7)
        y += controls['bow_noise'] * noise
        y *= 1 + 0.025 * np.sin(2 * np.pi * 1.4 * t + rng.uniform(0, 2 * np.pi))
        attack = 0.025 if v.params.get("legato", False) else self.attack
        env = adsr(ctx.samples(v.dur), n, ctx.sr, attack, 0.3, 0.88, self.release)
        if n:
            count = min(n, max(2, ctx.samples(0.002)))
            env[-count:] *= np.linspace(1, 0, count)
        return _body_equalize(self.level * vel_gain(v.vel) * y * env, ctx.sr, body_eq, self.release)


@register_instrument("violin")
@dataclass(frozen=True, kw_only=True)
class Violin(BowedString):
    """Solo violin body and bow preset."""


@register_instrument("cello")
@dataclass(frozen=True, kw_only=True)
class Cello(BowedString):
    """Larger body, slower bow attack and slower vibrato."""

    program: int | None = 42
    body_size: float = 2.2
    attack: float = 0.16
    vibrato_rate: float = 4.7
