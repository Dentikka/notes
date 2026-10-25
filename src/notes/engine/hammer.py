"""Procedural nonlinear felt contact coupled to damped string modes.

A simplified lumped hammer and modal string, not a calibrated concert-piano model.
Contact is integrated at >=176.4 kHz; free ringing uses analytic damped modes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def struck_string(freq: float, n: int, sr: int, velocity: float, *,
                  brightness: float = 0.65, stiffness: float = 0.00012,
                  sustain: float = 8.0, detune: float = 1.4, radiation: bool = False) -> np.ndarray:
    """Return bridge-velocity-like output following one felt hammer impact.

    Frequencies and stiffness define the modes, velocity the initial hammer speed.
    The contact has a compression-only power-law force; no prerecorded excitation.
    """
    if n <= 0:
        return np.zeros(0)
    if velocity <= 0:
        return np.zeros(n)
    factor = max(1, int(np.ceil(176400 / sr)))
    rate = sr * factor
    dt = 1 / rate
    k = np.arange(1, min(65, int(0.40 * sr / freq) + 1), dtype=float)
    if not len(k):
        return np.zeros(n)
    frequencies = freq * k * np.sqrt((1 + stiffness * k * k) / (1 + stiffness))
    keep = frequencies < 0.4 * sr
    k, frequencies = k[keep], frequencies[keep]
    omega = 2 * np.pi * frequencies
    coupling = np.sin(np.pi * k * 0.135)
    bridge = np.sin(np.pi * k * 0.965)
    # The old near-bridge velocity tap boosts overtones relative to the fundamental.
    # Blend a displacement-like modal readout above A2; A2 and below remain identical.
    register = float(np.clip(np.log2(freq / 110) / 2.6, 0, 1)) if radiation else 0.0
    bridge *= k ** (-1.2 * register)
    mass = 0.012 * (220 / freq) ** 0.35
    hammer_mass = 0.007 * (220 / freq) ** 0.15
    # Force constant has units for the chosen power law; hardness sets felt stiffness.
    felt = 2.0e9 * (0.45 + brightness)
    exponent = 2.3
    hammer_x, hammer_v = 0.0, 0.6 + 3.4 * velocity
    q, speed = np.zeros(len(k)), np.zeros(len(k))
    c, s = np.cos(omega * dt), np.sin(omega * dt)
    t60 = sustain * (220 / freq) ** 0.28 / (1 + 0.045 * (k - 1) ** 1.2)
    # Frequency-dependent loss is additional to the original per-mode decay.
    t60 /= 1 + register * (frequencies / 1400) ** 1.2
    damping = np.exp(-np.log(1000) * dt / t60)
    contact_samples = min(n * factor, int(0.025 * rate))
    contact = np.zeros(contact_samples)
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
        contact[i] = bridge @ speed
        touched |= force > 0.001
        if touched and compression <= 0:
            stop = i + 1
            break
    # Decimate only the integrated contact. Pad with free ringing to avoid an edge transient.
    contact = contact[:stop]
    total = n * factor
    end = min(total, stop + 64 * factor)
    times = np.arange(max(0, end - stop)) / rate
    extension = np.zeros(len(times))
    for w, qi, vi, b, decay in zip(omega, q, speed, bridge, t60, strict=True):
        extension += b * (vi * np.cos(w * times) - w * qi * np.sin(w * times)) * np.exp(-np.log(1000)*times/decay)
    head = resample_poly(np.concatenate((contact, extension)), 1, factor)
    # The rest of the ring is evaluated at the output rate, at the same physical times.
    y = np.zeros(n)
    count = min(n, int(np.ceil(stop / factor)) + 16)
    y[:count] = head[:count]
    times = np.arange(count, n) / sr - stop / rate
    for w, qi, vi, b, decay in zip(omega, q, speed, bridge, t60, strict=True):
        phase = w * times
        primary = vi * np.cos(phase) - w * qi * np.sin(phase)
        partner_phase = w * 2 ** (detune / 1200) * times
        partner = vi * np.cos(partner_phase) - w * qi * np.sin(partner_phase)
        # Bring in the secondary polarization continuously after contact.
        blend = 0.23 * (1 - np.exp(-times / 0.06))
        y[count:] += b * ((1 - blend) * primary + blend * partner) * np.exp(-np.log(1000)*times/decay)
    return y * 0.11
