"""Note controls must reach the source, remain independent and survive IR."""
from dataclasses import replace

import numpy as np
import pytest

from notes import Cello, Note, Song, Track, Violin, cat
from notes.engine import RenderContext, Voice, build_instrument
from notes.engine.mixer import render_score
from notes.ir.score import Score


def voice(instrument, params, sr=22050, velocity=.65):
    return instrument.voice(Voice(220., 57, .3, velocity, params), RenderContext(sr, 60),
                            np.random.default_rng(17))


@pytest.mark.parametrize('cls', [Violin, Cello])
@pytest.mark.parametrize('sr', [8000, 22050, 44100])
@pytest.mark.parametrize('velocity', [.35, .9])
@pytest.mark.parametrize('controls', [{'pressure': .22}, {'bow_noise': .011}, {'pressure': 2., 'bow_noise': 0.}])
def test_note_excitation_equals_independently_configured_instrument(cls, sr, velocity, controls):
    instrument = cls(vibrato=0)
    actual = voice(instrument, controls, sr, velocity)
    expected = voice(replace(instrument, **controls), {}, sr, velocity)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, voice(instrument, {}, sr, velocity))


@pytest.mark.parametrize('cls', [Violin, Cello])
def test_default_controls_can_override_a_configured_source(cls):
    baseline = cls(vibrato=0)
    changed = replace(baseline, pressure=2., bow_noise=0.)
    assert np.array_equal(voice(baseline, {}), voice(changed, {'pressure': .55, 'bow_noise': .025}))
    assert np.array_equal(voice(baseline, {}), voice(baseline, {'pressure': .55, 'bow_noise': .025}))


@pytest.mark.parametrize('control', ['pressure', 'bow_noise'])
@pytest.mark.parametrize('value', [-.01, float('nan'), float('inf'), None, 'bad'])
def test_invalid_note_excitation_is_rejected(control, value):
    with pytest.raises(ValueError, match=control):
        voice(Violin(), {control: value})


def test_note_excitation_survives_dsl_ir_and_composes_with_expression_and_body_eq():
    phrase = cat(Note(55, .25).param(pressure=.22, bow_noise=.018, body_eq=((202., 4., -12.),)),
                 Note(69, .25).param(pressure=2., bow_noise=.011,
                                    expression_points=((0., .4), (.2, 1.), (.4, .5))))
    score = Song(Track('violin', Violin(vibrato=0))(phrase), bpm=60, sample_rate=8000,
                 seed=17, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param('pressure') == .22
    assert restored.events[1].param('bow_noise') == .011
    actual = render_score(score).audio
    assert np.array_equal(actual, render_score(restored).audio)
    plain = replace(score, events=tuple(replace(e, params=()) for e in score.events))
    other = render_score(plain).audio
    common = min(actual.shape[1], other.shape[1])
    assert np.linalg.norm(actual[:, :common]-other[:, :common]) > .01


@pytest.mark.parametrize('cls', [Violin, Cello])
@pytest.mark.parametrize('sr', [8000, 22050, 44100])
def test_noise_seed_isolates_a_linear_noise_component_and_preserves_caller_stream(cls, sr):
    # Nonzero vibrato, EQ, expression-independent envelope and legato stay fixed.
    instrument = cls(vibrato=.2, body_eq=((150., 1., -3.),))
    params = {'legato': True, 'pitch_points': ((0., 0.), (.2, .15))}
    base = voice(instrument, {**params, 'bow_noise': 0.}, sr)
    assert np.array_equal(base, voice(instrument, {**params, 'bow_noise': 0., 'bow_noise_seed': 41}, sr))
    a = voice(instrument, {**params, 'bow_noise': .01, 'bow_noise_seed': 41}, sr)
    b = voice(instrument, {**params, 'bow_noise': .02, 'bow_noise_seed': 41}, sr)
    # Recursive EQ accumulates round-off before subtracting the common carrier.
    np.testing.assert_allclose(b-base, 2*(a-base), atol=2e-12, rtol=1e-9)
    assert np.linalg.norm(a-voice(instrument, {**params, 'bow_noise': .01, 'bow_noise_seed': 29}, sr)) > .01
    states = []
    for seed in (None, 0, 2**64-1):
        rng = np.random.default_rng(17)
        instrument.voice(Voice(220., 57, .3, .65, {**params, 'bow_noise_seed': seed}),
                         RenderContext(sr, 60), rng)
        states.append(rng.random(10))
    assert all(np.array_equal(states[0], s) for s in states[1:])


@pytest.mark.parametrize('cls', [Violin, Cello])
def test_noise_seed_note_override_legacy_restore_and_ir_roundtrip(cls):
    seeded = cls(bow_noise_seed=41)
    assert build_instrument(seeded.to_spec()) == seeded
    assert np.array_equal(voice(seeded, {}), voice(cls(), {'bow_noise_seed': 41}))
    assert np.array_equal(voice(cls(), {}), voice(seeded, {'bow_noise_seed': None}))
    # With the current harmonic source, noise is the first caller RNG draw.
    assert np.array_equal(voice(cls(), {}), voice(cls(), {'bow_noise_seed': 17}))
    score = Song(Track('strings', seeded)(Note(57, .3).param(bow_noise_seed=2**64-1)),
                 bpm=60, sample_rate=8000, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param('bow_noise_seed') == 2**64-1
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize('seed', [-1, 2**64, True, 41., '41', float('nan'), [], {}])
def test_noise_seed_rejects_invalid_constructor_and_note_values(seed):
    with pytest.raises(ValueError, match='bow_noise_seed'):
        Violin(bow_noise_seed=seed)
    with pytest.raises(ValueError, match='bow_noise_seed'):
        voice(Violin(), {'bow_noise_seed': seed})


def test_scalar_texture_fitter_recovers_known_bowed_noise_without_changing_pitch_controls():
    from notes.ears import fit_roughness

    ctx = RenderContext(22050, 60)
    v = Voice(220., 57, 3., .65, {})
    def source(amount, seed):
        return Violin(vibrato=0, bow_noise=amount, bow_noise_seed=seed).voice(
            v, ctx, np.random.default_rng(17))[:3*ctx.sr]
    target = source(.018, 97)
    result = fit_roughness(target, source, ctx.sr, v.freq)
    assert result.verified_within_tolerance and not result.verification_regressed
    assert result.roughness == pytest.approx(.018, rel=.1)
    assert len(result.history) == 4
