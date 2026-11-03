from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from notes import AcousticGuitar, NylonGuitar
from notes.engine import RenderContext, Voice
from notes.engine.instruments.strings import fret_scale
from notes.engine.modal_acoustic import pluck_force
from notes.engine.registry import build_instrument


def test_force_satisfies_damped_oscillator_with_static_displacement_and_zero_velocity():
    sr, freq, sustain = 8000, 1500., 7.
    # At this rate only two modes fit; a centre pluck suppresses mode 2.
    signal, static = pluck_force(freq, 120, sr, np.random.default_rng(17), pick=.5,
                                 sustain=sustain, stiffness=0, polarization=0, roughness=0)
    gamma = np.log(1000)/(sustain*(220/freq)**.3)
    omega = 2*np.pi*freq
    reference = solve_ivp(lambda _, q: [q[1], -2*gamma*q[1]-(omega**2+gamma**2)*q[0]],
                          [0, 120/sr], [1., 0.], t_eval=np.arange(120)/sr, rtol=1e-10, atol=1e-12)
    assert abs(signal[0]-static) < 1e-12
    assert np.max(abs(signal/static-reference.y[0])) < 1e-7


def test_static_body_does_not_sound_until_load_changes_and_bends_do_not_retune_it(monkeypatch):
    def constant_force(freq, n, sr, rng, **kwargs):
        return np.full(n, 1.25), 1.25
    monkeypatch.setattr("notes.engine.modal_acoustic.pluck_force", constant_force)
    inst = AcousticGuitar(model="modal", pick_noise=0)
    ctx = RenderContext(8000, 60)
    voice = Voice(220., 57, .25, .7, {})
    original = inst.voice(voice, ctx, np.random.default_rng(17))
    bent = inst.voice(replace(voice, params={"bend": 7}), ctx, np.random.default_rng(17))
    # FFT convolution roundoff can leak sub-1e-18 values from the release.
    assert np.max(abs(original[:2000])) < 1e-14
    assert np.linalg.norm(original[2000:]) > .001
    assert np.array_equal(original, bent)


@pytest.mark.parametrize("instrument", [AcousticGuitar, NylonGuitar])
@pytest.mark.parametrize("sr", [8000, 44100])
def test_complete_modal_voice_is_reproducible_serializable_and_has_body_tail(instrument, sr):
    inst = instrument(model="modal", pick_position=.08)
    assert build_instrument(inst.to_spec()) == inst
    voice = Voice(329.6276, 64, .3, .7, {"string": 1, "pitch_points": [[0, 0], [.2, .5]],
                                      "pitch_interpolation": "sinc"})
    ctx = RenderContext(sr, 60)
    a = inst.voice(voice, ctx, np.random.default_rng(17))
    b = inst.voice(voice, ctx, np.random.default_rng(17))
    assert np.array_equal(a, b) and np.isfinite(a).all()
    assert len(a) == round((.3+inst.release_time())*sr)
    assert inst.release_time() == pytest.approx(inst.release+.25)
    assert inst.inserts() == () and abs(a[-1]) < 1e-10


def test_modal_fret_geometry_matches_the_equivalent_active_string_pick_position():
    ctx = RenderContext(8000, 60)
    scale = fret_scale(71, 1)
    fixed = AcousticGuitar(model="modal", fretting=True, pick_position=.08)
    active = AcousticGuitar(model="modal", pick_position=.08/scale)
    a = fixed.voice(Voice(493.8833, 71, .4, .7, {"string": 1}), ctx, np.random.default_rng(17))
    b = active.voice(Voice(493.8833, 71, .4, .7, {}), ctx, np.random.default_rng(17))
    assert np.array_equal(a, b)


def test_per_note_modal_pick_and_stiffness_controls_match_constructor_settings():
    ctx = RenderContext(8000, 60)
    inst = AcousticGuitar(model="modal")
    params = {"pick_position":.1, "pick_hardness":.5, "stiffness":.0002}
    a = inst.voice(Voice(220, 57, .3, .7, params), ctx, np.random.default_rng(17))
    selected = AcousticGuitar(model="modal", pick_position=.1, brightness=.5, stiffness=.0002)
    b = selected.voice(Voice(220, 57, .3, .7, {}), ctx, np.random.default_rng(17))
    assert np.array_equal(a, b)


def test_twelve_string_modal_course_adds_the_octave():
    ctx = RenderContext(8000, 60)
    inst = AcousticGuitar(model="modal", pick_position=.2, body=0, polarization=0,
                          course_detune=0, course_lag=0, pick_noise=0)
    voice = Voice(110., 45, 1., .7, {"string": 5})
    six = inst.voice(voice, ctx, np.random.default_rng(17))
    twelve = replace(inst, twelve=True).voice(voice, ctx, np.random.default_rng(17))
    def ratio(wave):
        wave = wave[800:7200]
        spectrum = abs(np.fft.rfft(wave*np.hanning(len(wave))))
        frequency = np.fft.rfftfreq(len(wave), 1/ctx.sr)
        return spectrum[abs(frequency-220)<3].max()/spectrum[abs(frequency-110)<3].max()
    assert ratio(twelve) > ratio(six)*1.3


def test_absent_note_never_excites_a_preloaded_body():
    inst = AcousticGuitar(model="modal")
    ctx = RenderContext(8000, 60)
    for duration, velocity in ((0., .7), (.2, 0.)):
        result = inst.voice(Voice(220, 57, duration, velocity, {}), ctx, np.random.default_rng(17))
        assert not np.any(result)


@pytest.mark.parametrize("settings", [{"sustain":0}, {"treble_decay":-1}, {"pick_noise":float("nan")},
                                       {"velocity_brightness":2}])
def test_modal_configuration_validation(settings):
    with pytest.raises(ValueError):
        AcousticGuitar(model="modal", **settings)


def test_modal_pick_must_be_inside_the_fretted_string():
    inst = AcousticGuitar(model="modal", fretting=True, pick_position=.4)
    with pytest.raises(ValueError, match="pick position"):
        inst.voice(Voice(1318.5, 88, .2, .7, {"string": 1}), RenderContext(8000, 60), np.random.default_rng(17))
