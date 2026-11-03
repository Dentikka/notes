"""Analytic pluck force and a designed body with static initial loading.

String losses are prescribed independently of body radiation; this is not the
passively coupled bridge model. No recorded excitation or measured IR is used.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from notes.engine.articulation import pitch_curve, variable_rate_read
from notes.engine.base import RenderContext, Voice
from notes.engine.dsp import delay, vel_gain
from notes.engine.effects.body import Body
from notes.engine.effects.tone import Filter
from notes.engine.instruments.strings import _gate, _pick_click, _t60, fret_scale

if TYPE_CHECKING:
    from notes.engine.instruments.acoustic import AcousticGuitar


def pluck_force(freq: float, n: int, sr: int, rng: np.random.Generator, *,
                pick: float = .18, hardness: float = .9, sustain: float = 7.,
                treble_decay: float = 3., prompt_decay: float = 0.,
                polarization: float = .3, stiffness: float = 4e-5,
                roughness: float = .03) -> tuple[np.ndarray, float]:
    """Bridge force and its initial static load for a released deflected string.

    Each damped mode starts at its displacement maximum with zero velocity.
    The small initial-shape roughness is generated, not a sampled pick attack.
    """
    if (not np.isfinite(freq) or not np.isfinite(sr) or sr < 1000 or int(sr) != sr
            or not 0 < freq < sr*.44 or not isinstance(n, int) or n < 0):
        raise ValueError("invalid modal string frequency, sample rate or length")
    if not np.isfinite(pick) or not 0 < pick < 1:
        raise ValueError("modal pick position must lie inside the vibrating string")
    for name, value in (("sustain", sustain), ("treble_decay", treble_decay)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("hardness", hardness), ("prompt_decay", prompt_decay),
                        ("polarization", polarization), ("stiffness", stiffness), ("roughness", roughness)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if n == 0:
        return np.zeros(0), 0.
    time = np.arange(n)/sr
    result = np.zeros(n)
    static = 0.
    t60 = _t60(sustain, freq)
    prompt = min(t60, t60*prompt_decay) if prompt_decay > 0 else t60
    cutoff = min(1500*2**(3*np.clip(hardness, 0, 1)), sr*.45)
    branches = ((1., freq, prompt, min(prompt, treble_decay)),
                (polarization, freq*2**(.7/1200), 1.4*t60 if prompt_decay == 0 else t60, treble_decay))
    for weight, fundamental, decay, upper in branches:
        if weight <= 0:
            continue
        h = np.arange(1, min(385, int(sr*.44/fundamental)+1), dtype=float)
        hz = fundamental*h*np.sqrt((1+stiffness*h*h)/(1+stiffness))
        use = hz < sr*.44
        h, hz = h[use], hz[use]
        amplitude = 2/np.pi*(np.sin(np.pi*h*pick)+roughness*rng.uniform(-1, 1, len(h)))/h
        amplitude /= 1+(hz/cutoff)**2
        loss = 1/decay+max(0, 1/upper-1/decay)*np.maximum(
            0, (hz*hz-fundamental*fundamental)/max(1, 4000**2-fundamental*fundamental))
        static += weight*float(amplitude.sum())
        for frequency, amp, rate in zip(hz, amplitude, loss, strict=True):
            omega = 2*np.pi*frequency
            gamma = np.log(1000)*rate
            result += weight*amp*np.exp(-gamma*time)*(np.cos(omega*time)+gamma/omega*np.sin(omega*time))
    return result, static


def modal_acoustic_voice(instrument: AcousticGuitar, voice: Voice, ctx: RenderContext,
                         rng: np.random.Generator) -> np.ndarray:
    """Render string force changes, then the fixed body, including note-off.

    Pitch curves operate on force change before body filtering. Zero padding of
    that change therefore represents the preloaded state, not an unloaded body.
    """
    if ctx.sr < 4000:
        raise ValueError("modal acoustic body needs a sample rate of at least 4000 Hz")
    gate = ctx.samples(voice.dur)
    count = ctx.samples(voice.dur+instrument.release_time())
    if gate == 0 or voice.vel <= 0 or instrument.level == 0:
        return np.zeros(count)
    scale = fret_scale(voice.pitch, voice.params.get("string")) if instrument.fretting else 1.
    pick = float(voice.params.get("pick_position", instrument.pick_position))/scale
    legato = bool(voice.params.get("legato", False))
    hardness = float(voice.params.get("pick_hardness", instrument.brightness))
    if "pick_hardness" in voice.params and (not np.isfinite(hardness) or not 0 <= hardness <= 1):
        raise ValueError("per-note pick_hardness must be finite and between 0 and 1")
    hardness *= .6 if legato else 1.
    hardness *= 1-instrument.velocity_brightness*(1-voice.vel)
    try:
        stiffness = float(voice.params.get("stiffness", instrument.stiffness))
    except (TypeError, ValueError) as error:
        raise ValueError("per-note stiffness must be finite and nonnegative") from error
    curve = pitch_curve(voice.params, count, ctx.sr, vibrato_shape="push", rng=rng)

    def string(freq: float) -> np.ndarray:
        static = 0.

        def render(n: int) -> np.ndarray:
            nonlocal static
            raw, static = pluck_force(freq, n, ctx.sr, rng, pick=pick, hardness=hardness,
                sustain=instrument.sustain, treble_decay=instrument.treble_decay,
                prompt_decay=instrument.prompt_decay, polarization=instrument.polarization,
                stiffness=stiffness)
            return raw-static

        change = (render(count) if curve is None else variable_rate_read(
            render, curve, interpolation=voice.params.get("pitch_interpolation", "linear")))
        envelope = _gate(gate, count, ctx.sr, instrument.release)
        envelope[:gate] = 1.  # held static load has no zero-state attack ramp
        signal = (change+static)*envelope-static
        if not legato:
            signal += instrument.pick_noise*abs(static)*_pick_click(count, ctx.sr, rng)
        wave = signal[None, :]
        for effect in (Filter(kind="highpass", cutoff=60.),
                       Body(size=instrument.body_size, diffuse=instrument.diffuse,
                            mix=instrument.body, phase=instrument.body_phase)):
            wave = effect.process(wave, ctx)
        return wave[0]

    result = string(voice.freq)
    if instrument.twelve:
        selected = voice.params.get("string")
        octave = selected in (3, 4, 5, 6) if selected is not None else voice.pitch < 59
        partner = string(voice.freq*(2. if octave else 1.)*2**(instrument.course_detune/1200))
        lag = ctx.samples(instrument.course_lag)
        result = (delay(result, lag)+instrument.octave_level*partner if octave else
                  result+instrument.octave_level*delay(partner, lag))
    result *= instrument.level*vel_gain(voice.vel)
    fade = min(len(result), ctx.samples(.01))
    if fade:
        result[-fade:] *= np.linspace(1, 0, fade)
    return result
