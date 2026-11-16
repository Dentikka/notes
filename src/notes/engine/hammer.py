"""Procedural nonlinear felt contact coupled to damped string modes.

A simplified lumped hammer and modal string, not a calibrated concert-piano model.
Contact is integrated at >=176.4 kHz; free ringing uses analytic damped modes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from notes.engine.articulation import _partial_gain_points, _unison_strings


def struck_string(freq: float, n: int, sr: int, velocity: float, *,
                  brightness: float = 0.65, stiffness: float = 0.00012,
                  sustain: float = 8.0, detune: float = 1.4, radiation: bool = False,
                  attack_presence: float = 0.0, hammer_mass_scale: float = 1.0,
                  partial_gain_points: tuple[tuple[float, ...], ...] = (),
                  unison_strings: int | None = None, hammer_speed: float | None = None,
                  felt_stiffness_scale: float = 1.0, contact_oversample: int = 1,
                  radiation_compensation: float = 0.0,
                  _audit: dict[str, float] | None = None) -> np.ndarray:
    """Return bridge-velocity-like output following one felt hammer impact.

    Frequencies and stiffness define the modes, velocity the initial hammer speed.
    The contact has a compression-only power-law force; no prerecorded excitation.
    hammer_mass_scale changes the physical impact, not an output equalizer.
    partial_gain_points changes only modal readout, including both unison
    components and the integrated contact head. Empty preserves the old output.
    unison_strings optionally uses 1..3 equally weighted free-ring components
    from the same impact state. None keeps the original unequal pair. This
    readout approximation does not simulate separate coupled contact states.
    hammer_speed optionally sets initial hammer speed in model m/s, from 0 to 4.
    None retains 0.6 + 3.4*velocity; velocity zero remains a silent note-off.
    felt_stiffness_scale multiplies the compression-force coefficient. Optional
    contact_oversample refines only integration, keeping the output mode band.
    radiation_compensation restores a fraction of synthetic modal readout power
    removed by the register tap: one scalar from the impact state, not a body
    model, recorded response, change of modal ratios or altered contact force.
    """
    if not np.isfinite(hammer_mass_scale) or not 0.25 <= hammer_mass_scale <= 4:
        raise ValueError("hammer_mass_scale must be between 0.25 and 4")
    if not np.isfinite(felt_stiffness_scale) or not .25 <= felt_stiffness_scale <= 16:
        raise ValueError('felt_stiffness_scale must be between .25 and 16')
    if (isinstance(contact_oversample, bool) or not isinstance(contact_oversample, int)
            or not 1 <= contact_oversample <= 16):
        raise ValueError('contact_oversample must be an integer from 1 to 16')
    if not np.isfinite(radiation_compensation) or not 0 <= radiation_compensation <= 1:
        raise ValueError('radiation_compensation must be between 0 and 1')
    if radiation_compensation and not radiation:
        raise ValueError('radiation_compensation requires radiation=True')
    if hammer_speed is not None:
        try:
            hammer_speed = float(hammer_speed)
        except (TypeError, ValueError) as error:
            raise ValueError('hammer_speed must be finite and between 0 and 4') from error
        if not np.isfinite(hammer_speed) or not 0 <= hammer_speed <= 4:
            raise ValueError('hammer_speed must be finite and between 0 and 4')
    points = _partial_gain_points(partial_gain_points)
    controls = np.asarray(points) if points else None
    strings = _unison_strings(unison_strings)
    if strings is not None and (not np.isfinite(detune) or detune < 0):
        raise ValueError('explicit unison detune must be finite and nonnegative')
    if n <= 0:
        return np.zeros(0)
    if velocity <= 0:
        return np.zeros(n)
    factor = max(1, int(np.ceil(176400 / sr)))*contact_oversample
    rate = sr * factor
    dt = 1 / rate
    k = np.arange(1, min(65, int(0.40 * sr / freq) + 1), dtype=float)
    if not len(k):
        return np.zeros(n)
    frequencies = freq * k * np.sqrt((1 + stiffness * k * k) / (1 + stiffness))
    keep = frequencies < 0.4 * sr
    k, frequencies = k[keep], frequencies[keep]
    if (strings is not None and strings > 1 and len(frequencies)
            and detune >= 1200*np.log2(.45*sr/max(frequencies))):
        raise ValueError('explicit unison frequencies must remain below .45 of the sample rate')
    omega = 2 * np.pi * frequencies
    coupling = np.sin(np.pi * k * 0.135)
    bridge = np.sin(np.pi * k * 0.965)
    # The old near-bridge velocity tap boosts overtones relative to the fundamental.
    # Blend a displacement-like modal readout above A2; A2 and below remain identical.
    register = float(np.clip(np.log2(freq / 110) / 2.6, 0, 1)) if radiation else 0.0
    bridge *= k ** (-1.2 * register)
    # A bounded onset-only lift; the accepted long-term spectral balance is unchanged.
    upper = float(np.clip(np.log2(freq / 261.625565) / 1.333333, 0, 1))
    onset_lift = 0.4 * attack_presence * upper * (1 - 1 / k)
    mass = 0.012 * (220 / freq) ** 0.35
    hammer_mass = 0.007 * (220 / freq) ** 0.15
    hammer_mass *= hammer_mass_scale
    # Force constant has units for the chosen power law; hardness sets felt stiffness.
    felt = 2.0e9 * (0.45 + brightness)
    felt *= felt_stiffness_scale
    exponent = 2.3
    hammer_x, hammer_v = 0.0, 0.6 + 3.4 * velocity if hammer_speed is None else hammer_speed
    initial_hammer_speed = hammer_v
    q, speed = np.zeros(len(k)), np.zeros(len(k))
    c, s = np.cos(omega * dt), np.sin(omega * dt)
    t60 = sustain * (220 / freq) ** 0.28 / (1 + 0.045 * (k - 1) ** 1.2)
    # Frequency-dependent loss is additional to the original per-mode decay.
    t60 /= 1 + register * (frequencies / 1400) ** 1.2
    damping = np.exp(-np.log(1000) * dt / t60)
    contact_samples = min(n * factor, int(0.025 * rate))
    contact = np.zeros(contact_samples)
    contact_gains = None
    if controls is not None:
        contact_gains = np.ones((contact_samples, len(k)))
        contact_time = np.arange(contact_samples)/rate
        for index, number in enumerate(k.astype(int)):
            if number < controls.shape[1]:
                contact_gains[:, index] = np.interp(contact_time, controls[:, 0], controls[:, number])
    touched = False
    stop = contact_samples
    for i in range(contact_samples):
        compression = max(0.0, hammer_x - float(coupling @ q))
        force = felt * compression ** exponent
        acceleration = coupling * force / mass
        new_q = c * q + s * speed / omega + (1 - c) * acceleration / omega ** 2
        speed = (c * speed - omega * s * q + s * acceleration / omega) * damping
        q = new_q * damping
        hammer_v -= dt * force / hammer_mass
        hammer_x += dt * hammer_v
        readout = bridge * (1 + onset_lift * np.exp(-i * dt / 0.06))
        if contact_gains is not None:
            readout = readout*contact_gains[i]
        contact[i] = readout @ speed
        touched |= force > 0.001
        if touched and compression <= 0:
            stop = i + 1
            break
    # Decimate only the integrated contact. Pad with free ringing to avoid an edge transient.
    compensation = 1.
    if (radiation_compensation or _audit is not None) and register:
        state_power = speed*speed+(omega*q)**2
        original_power = float(np.sum(np.sin(np.pi*k*.965)**2*state_power))
        filtered_power = float(np.sum(bridge*bridge*state_power))
        if filtered_power > np.finfo(float).tiny:
            unit_compensation = np.sqrt(original_power/filtered_power)
            compensation = unit_compensation**radiation_compensation
        else:
            unit_compensation = 1.
    else:
        unit_compensation = 1.
    if _audit is not None:
        # Diagnostic bookkeeping only, never used to alter the force or state.
        # A decrease of total energy includes both modeled damping and numerical
        # error; this does not identify all missing energy as physical loss.
        _audit.update(contact_seconds=stop/rate, initial_energy=.5*hammer_mass*initial_hammer_speed**2,
            string_energy=.5*mass*float(np.sum(speed*speed+(omega*q)**2)),
            hammer_energy=.5*hammer_mass*hammer_v**2, readout_power_gain=float(unit_compensation),
            separated=float(touched and compression <= 0))
    contact = contact[:stop]
    total = n * factor
    end = min(total, stop + 64 * factor)
    times = np.arange(max(0, end - stop)) / rate
    extension = np.zeros(len(times))
    for number, w, qi, vi, b, decay, lift in zip(k.astype(int), omega, q, speed, bridge, t60, onset_lift, strict=True):
        readout = b * (1 + lift * np.exp(-(stop / rate + times) / 0.06))
        if controls is not None and number < controls.shape[1]:
            readout *= np.interp(stop/rate+times, controls[:, 0], controls[:, number])
        extension += readout * (vi * np.cos(w * times) - w * qi * np.sin(w * times)) * np.exp(-np.log(1000)*times/decay)
    head = resample_poly(np.concatenate((contact, extension)), 1, factor)
    # The rest of the ring is evaluated at the output rate, at the same physical times.
    y = np.zeros(n)
    count = min(n, int(np.ceil(stop / factor)) + 16)
    y[:count] = head[:count]
    times = np.arange(count, n) / sr - stop / rate
    for number, w, qi, vi, b, decay, lift in zip(k.astype(int), omega, q, speed, bridge, t60, onset_lift, strict=True):
        phase = w * times
        primary = vi * np.cos(phase) - w * qi * np.sin(phase)
        if strings is None:
            partner_phase = w * 2 ** (detune / 1200) * times
            partner = vi * np.cos(partner_phase) - w * qi * np.sin(partner_phase)
            # Preserve the original unequal secondary polarization exactly.
            blend = 0.23 * (1 - np.exp(-times / 0.06))
            ringing = (1 - blend) * primary + blend * partner
        elif strings == 1:
            ringing = primary
        else:
            combined = np.zeros_like(primary)
            for cents in np.linspace(-detune, detune, strings):
                if cents == 0:
                    combined += primary
                else:
                    unison_w = w*2**(cents/1200)
                    combined += vi*np.cos(unison_w*times)-unison_w*qi*np.sin(unison_w*times)
            blend = 1-np.exp(-times/.06)
            ringing = (1-blend)*primary+blend*combined/strings
        readout = b * (1 + lift * np.exp(-(stop / rate + times) / 0.06))
        if controls is not None and number < controls.shape[1]:
            readout *= np.interp(stop/rate+times, controls[:, 0], controls[:, number])
        y[count:] += readout * ringing * np.exp(-np.log(1000)*times/decay)
    output = y*0.11
    return output*compensation if compensation != 1. else output
