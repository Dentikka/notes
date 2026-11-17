"""Prescribed modal forces, preload and an output clock independent of pitch."""
from dataclasses import replace

import numpy as np
import pytest

from notes import AcousticGuitar, Filter, Note, NylonGuitar, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.articulation import variable_rate_read
from notes.engine.dsp import vel_gain
from notes.engine.instruments.strings import _gate, _t60
from notes.engine.modal_acoustic import pluck_force
from notes.ir import Score


def test_one_modulated_force_matches_an_independent_damped_mode_and_initial_load():
    freq, sr, n = 1500., 8000, 800
    points = ((.01, .3), (.04, 1.2))
    wave, static = pluck_force(freq, n, sr, np.random.default_rng(9), stiffness=0, roughness=0,
        polarization=0, force_gains=(1.,)+(0.,)*79, force_gain_points=points)
    t = np.arange(n)/sr
    amplitude = 2/np.pi*np.sin(np.pi*.18)/(1+(freq/(sr*.45))**2)
    gamma, omega = np.log(1000)/_t60(7., freq), 2*np.pi*freq
    carrier = amplitude*np.exp(-gamma*t)*(np.cos(omega*t)+gamma/omega*np.sin(omega*t))
    expected = carrier*np.interp(t, [.01, .04], [.3, 1.2])
    np.testing.assert_allclose(wave, expected, atol=1e-15)
    assert static == pytest.approx(amplitude*.3, abs=1e-15)
    assert wave[0] == pytest.approx(static, abs=1e-15)


@pytest.mark.parametrize('interpolation', ['linear', 'sinc'])
def test_uniform_force_curve_stays_on_output_clock_through_an_octave_pitch_change(interpolation):
    sr, n, freq = 8000, 1600, 800.
    curve = np.full(n, 12.)
    options = dict(polarization=.3, force_gains=(.4, 1.3))
    _, held = pluck_force(freq, n, sr, np.random.default_rng(9), **options)
    def source(m):
        raw, load = pluck_force(freq, m, sr, np.random.default_rng(9), **options)
        return raw-load
    def reader(render):
        return variable_rate_read(render, curve, interpolation=interpolation)
    raw = reader(source)+held
    time = np.arange(n)/sr
    gains = np.interp(time, [0., .08, .16], [.5, 0., 1.])
    points = ((0., *([.5]*80)), (.08, *([0.]*80)), (.16, *([1.]*80)))
    actual, static = pluck_force(freq, n, sr, np.random.default_rng(9), force_gain_points=points,
                                 _read_mode=reader, **options)
    np.testing.assert_allclose(actual, raw*gains, atol=2e-14, rtol=2e-13)
    assert static == pytest.approx(.5*held, abs=1e-14)
    # A source-clock envelope would put this zero at .04 instead of .08 seconds.
    assert abs(actual[round(.08*sr)]) < 1e-15


def test_identity_is_bit_exact_and_omitted_modes_keep_unit_gain():
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    raw, static = pluck_force(220., 1000, 8000, a)
    unchanged, held = pluck_force(220., 1000, 8000, b, force_gain_points=((0., *([1.]*80)),))
    assert np.array_equal(raw, unchanged) and static == held and a.random() == b.random()
    first = ((0., .4), (.1, 1.2))
    full = tuple((t, value, *([1.]*79)) for t, value in first)
    x, sx = pluck_force(220., 1000, 8000, np.random.default_rng(9), force_gain_points=first)
    y, sy = pluck_force(220., 1000, 8000, np.random.default_rng(9), force_gain_points=full)
    assert np.array_equal(x, y) and sx == sy


@pytest.mark.parametrize('interpolation', ['linear', 'sinc'])
def test_voice_applies_output_clock_force_before_body_and_noteoff(interpolation):
    sr = 8000
    ctx = RenderContext(sr, 60)
    points = ((0., *([.5]*80)), (.08, *([0.]*80)), (.16, *([1.]*80)))
    inst = AcousticGuitar(model='modal', body=0, polarization=0, pick_noise=0, force_gain_points=points)
    v = Voice(800., 67., .2, .7, {'pitch_points': [[0., 12.]], 'pitch_interpolation': interpolation})
    actual = inst.voice(v, ctx, np.random.default_rng(9))
    n = ctx.samples(v.dur+inst.release_time())
    _, static = pluck_force(v.freq, n, sr, np.random.default_rng(9), polarization=0)
    def source(m):
        raw, held = pluck_force(v.freq, m, sr, np.random.default_rng(9), polarization=0)
        return raw-held
    carrier = variable_rate_read(source, np.full(n, 12.), interpolation=interpolation)+static
    gain = np.interp(np.arange(n)/sr, [0., .08, .16], [.5, 0., 1.])
    envelope = _gate(ctx.samples(v.dur), n, sr, inst.release)
    envelope[:ctx.samples(v.dur)] = 1.
    expected = Filter(kind='highpass', cutoff=60.).process(
        (carrier*gain*envelope-.5*static)[None, :], ctx)[0]*inst.level*vel_gain(v.vel)
    fade = ctx.samples(.01)
    expected[-fade:] *= np.linspace(1, 0, fade)
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize('phase', ['designed', 'minimum'])
@pytest.mark.parametrize('twelve', [False, True])
def test_constant_curves_multiply_static_forces_including_body_attack_and_noteoff(phase, twelve):
    ctx = RenderContext(8000, 60)
    v = Voice(220., 57., .08, .7, {'pitch_points': [[0., 0.], [.1, .5]]})
    inst = AcousticGuitar(model='modal', body_phase=phase, twelve=twelve,
                         force_gains=(.8, 1.3), force_gain_points=((0., .4, .6),))
    expected = replace(inst, force_gain_points=(), force_gains=(.8*.4, 1.3*.6))
    a = inst.voice(v, ctx, np.random.default_rng(9))
    b = expected.voice(v, ctx, np.random.default_rng(9))
    np.testing.assert_allclose(a, b, atol=3e-13, rtol=3e-13)
    assert np.linalg.norm(a[ctx.samples(v.dur):]) > 1e-5


@pytest.mark.parametrize('interpolation', ['linear', 'sinc'])
def test_identity_curves_preserve_complete_expressive_voice(interpolation):
    inst = AcousticGuitar(model='modal', body_sampling='fixed', twelve=True)
    v = Voice(220., 57., .1, .7, {'bend': .7, 'pitch_interpolation': interpolation})
    a = inst.voice(v, RenderContext(8000, 60), np.random.default_rng(9))
    b = replace(inst, force_gain_points=((0., 1., 1.), (.1, 1., 1.))).voice(
        v, RenderContext(8000, 60), np.random.default_rng(9))
    assert np.array_equal(a, b)


@pytest.mark.parametrize('bad', [None, [[0.]], [[-1., 1.]], [[.1, 1.], [.1, 2.]],
                                [[0., -1.]], [[0., 17.]], [[0., np.nan]], [[0., 1.], [.1, 1., 2.]],
                                [[float(t), 1.] for t in range(513)], [[0., *([1.]*81)]]])
def test_invalid_curves_rejected_before_empty_output(bad):
    with pytest.raises(ValueError, match='force_gain_points'):
        pluck_force(220., 0, 8000, np.random.default_rng(9), force_gain_points=bad)
    with pytest.raises(ValueError, match='force_gain_points'):
        AcousticGuitar(model='modal', force_gain_points=bad)
    with pytest.raises(ValueError, match='force_gain_points'):
        AcousticGuitar(model='modal').voice(Voice(220., 57., 0., .7, {'force_gain_points': bad}),
            RenderContext(8000, 60), np.random.default_rng(9))


@pytest.mark.parametrize('model', ['waveguide', 'coupled'])
def test_nonmodal_models_reject_instead_of_ignoring_curves(model):
    with pytest.raises(ValueError, match='requires'):
        AcousticGuitar(model=model, force_gain_points=((0., 1.),))
    with pytest.raises(ValueError, match='requires'):
        AcousticGuitar(model=model).voice(Voice(220., 57., .1, .7, {'force_gain_points': [[0., 1.]]}),
            RenderContext(8000, 60), np.random.default_rng(9))


@pytest.mark.parametrize('cls', [AcousticGuitar, NylonGuitar])
def test_registry_per_note_override_and_json_pcm_with_curves(cls, tmp_path):
    points = ((0., .4, 1.2), (.1, 1.3, .6))
    inst = cls(model='modal', body_sampling='fixed', force_gain_points=[list(r) for r in points])
    assert inst.force_gain_points == points and build_instrument(inst.to_spec()) == inst
    voice = Voice(220., 57., .1, .7, {'force_gain_points': [[0., .2], [.1, 1.]]})
    a = inst.voice(voice, RenderContext(8000, 60), np.random.default_rng(9))
    b = replace(inst, force_gain_points=((0., .2), (.1, 1.))).voice(
        replace(voice, params={}), RenderContext(8000, 60), np.random.default_rng(9))
    assert np.array_equal(a, b)
    score = Song(Track('g', inst)(Note('A3', .1).param(pitch_points=[[0., 0.], [.1, .5]],
        pitch_interpolation='sinc')), bpm=60, sample_rate=8000, seed=117, loudness=None).compile()
    path = tmp_path/'score.json'
    score.to_json(path)
    a, b = render_score(score), render_score(Score.from_json(path))
    assert np.array_equal(a.audio, b.audio) and np.isfinite(a.audio).all()
