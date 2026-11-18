"""Damping poles change independently of the existing register readout."""

import numpy as np
import pytest
from notes import Note, Piano, Song, Track
from notes.ears import partial_envelopes
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.hammer import struck_string
from notes.ir import Score


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0, 4.0])
def test_freely_ringing_fundamental_has_expected_positive_t60(scale):
    f = 1760.0
    sr = 16000
    y = struck_string(
        f, round(1.4 * sr), sr, 0.6, radiation=True, radiation_loss_scale=scale, unison_strings=1, detune=0.0
    )
    profile = partial_envelopes(y, sr, f, partials=1)
    use = (profile.time >= 0.3) & (profile.time <= 1.1)
    slope = np.polyfit(profile.time[use], 20 * np.log10(profile.amplitude[use, 0]), 1)[0]
    expected = 8 * (220 / f) ** 0.28 / (1 + scale * (f / 1400) ** 1.2)
    assert slope < 0 and -60 / slope == pytest.approx(expected, rel=2e-4)


def test_low_register_and_explicit_default_are_bitexact():
    opts = dict(freq=110.0, n=4000, sr=16000, velocity=0.6, radiation=True)
    base = struck_string(**opts)
    for scale in [0.0, 0.25, 1.0, 4.0]:
        np.testing.assert_array_equal(base, struck_string(**opts, radiation_loss_scale=scale))
    opts["freq"] = 1760.0
    np.testing.assert_array_equal(struck_string(**opts), struck_string(**opts, radiation_loss_scale=1.0))


def test_loss_control_keeps_initial_hammer_energy_and_contact_is_dissipative():
    states = []
    for scale in [0.0, 0.25, 1.0, 4.0]:
        audit = {}
        y = struck_string(1760.0, 4000, 16000, 0.6, radiation=True, radiation_loss_scale=scale, _audit=audit)
        assert (
            np.isfinite(y).all()
            and 0 < audit["string_energy"] + audit["hammer_energy"] <= 1.02 * audit["initial_energy"]
        )
        states.append(audit["initial_energy"])
    assert len(set(states)) == 1


@pytest.mark.parametrize("value", [True, False, None, -0.01, 4.01, np.nan, np.inf, [1.0]])
def test_invalid_loss_control_rejected_at_both_entrypoints(value):
    with pytest.raises(ValueError, match="radiation_loss_scale"):
        Piano(model="hammer_damped", radiation_loss_scale=value)
    with pytest.raises(ValueError, match="radiation_loss_scale"):
        struck_string(440.0, 100, 8000, 0.6, radiation=True, radiation_loss_scale=value)


@pytest.mark.parametrize("model", ["legacy", "soundboard", "hammer"])
def test_nondefault_control_cannot_be_silently_ignored(model):
    with pytest.raises(ValueError, match="requires hammer_damped"):
        Piano(model=model, radiation_loss_scale=0.5)
    inst = Piano(model=model)
    with pytest.raises(ValueError, match="requires hammer_damped"):
        inst.voice(
            Voice(440.0, 69.0, 0.1, 0.6, {"radiation_loss_scale": 0.5}),
            RenderContext(8000, 60),
            np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="radiation=True"):
        struck_string(440.0, 100, 8000, 0.6, radiation_loss_scale=0.5)


def test_constructor_note_override_reset_and_json_roundtrip(tmp_path):
    p = Piano(model="hammer_damped", radiation_loss_scale=0.25)
    plain = Piano(model="hammer_damped")
    assert build_instrument(p.to_spec()) == p
    ctx = RenderContext(8000, 60)
    v = Voice(1760.0, 93.0, 0.15, 0.6, {})
    a = p.voice(v, ctx, np.random.default_rng(1))
    b = plain.voice(Voice(v.freq, v.pitch, v.dur, v.vel, {"radiation_loss_scale": 0.25}), ctx, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    reset = p.voice(Voice(v.freq, v.pitch, v.dur, v.vel, {"radiation_loss_scale": 1.0}), ctx, np.random.default_rng(1))
    np.testing.assert_array_equal(reset, plain.voice(v, ctx, np.random.default_rng(1)))
    score = Song(Track("p", plain)(Note(93, 0.15).param(radiation_loss_scale=0.25))).compile()
    path = tmp_path / "score.json"
    score.to_json(path)
    restored = Score.from_json(path)
    assert restored.events[0].param("radiation_loss_scale") == 0.25


@pytest.mark.parametrize("model", ["legacy", "soundboard", "hammer", "hammer_damped"])
def test_explicit_undamped_note_rings_without_setting_pedal(model):
    inst = Piano(model=model, sustain=0.5)
    ctx = RenderContext(8000, 60)
    v = Voice(1760.0, 93.0, 0.05, 0.6, {})
    a = inst.voice(Voice(v.freq, v.pitch, v.dur, v.vel, {"undamped": True}), ctx, np.random.default_rng(2))
    b = inst.voice(Voice(v.freq, v.pitch, v.dur, v.vel, {"pedal": True}), ctx, np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    plain = inst.voice(v, ctx, np.random.default_rng(2))
    off = inst.voice(Voice(v.freq, v.pitch, v.dur, v.vel, {"undamped": False}), ctx, np.random.default_rng(2))
    np.testing.assert_array_equal(plain, off)
    assert len(a) > len(plain)
    with pytest.raises(ValueError, match="undamped"):
        inst.voice(Voice(v.freq, v.pitch, v.dur, v.vel, {"undamped": "false"}), ctx, np.random.default_rng(2))


def test_undamped_note_survives_the_normal_renderer_after_key_release():
    inst = Piano(model="hammer_damped", sustain=2.0, release=0.02, radiation_loss_scale=0.0)
    outputs = []
    for flag in [False, True]:
        score = Song(
            Track("p", inst)(Note("A4", 0.05, 0.9).param(undamped=flag)), bpm=60, sample_rate=8000, loudness=None
        ).compile()
        outputs.append(render_score(score, workers=1))
    assert outputs[1].duration > outputs[0].duration + 0.2
    assert np.max(abs(outputs[1].audio[:, 1600:2400])) > 1e-4
