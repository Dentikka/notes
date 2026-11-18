"""Open-length B, per-note precedence, expression and public IR behavior."""

from dataclasses import replace

import numpy as np
import pytest
from notes import ElectricGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.instruments.strings import fret_scale
from notes.ir import Score


@pytest.mark.parametrize("string,pitch", [(6, 40), (6, 45), (4, 55), (1, 64), (1, 67), (1, 72), (1, 77), (1, 88)])
@pytest.mark.parametrize("sr", [16000, 44100])
def test_open_reference_matches_absolute_per_note_B(string, pitch, sr):
    inst = ElectricGuitar(
        amplified=False, fretting=True, stiffness=2.3e-5, stiffness_reference="open", dispersion_fit="bracketed"
    )
    old = replace(inst, stiffness_reference="vibrating")
    ctx = RenderContext(sr, 60)
    f0 = 440 * 2 ** ((pitch - 69) / 12)
    params = {"string": string, "pitch_interpolation": "sinc"}
    actual = inst.voice(Voice(f0, float(pitch), 0.12, 0.7, params), ctx, np.random.default_rng(17))
    expected = old.voice(
        Voice(f0, float(pitch), 0.12, 0.7, {**params, "stiffness": inst.stiffness / fret_scale(pitch, string) ** 2}),
        ctx,
        np.random.default_rng(17),
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("B", [0.0, 0.00013])
def test_absolute_override_is_not_scaled_a_second_time(B):
    inst = ElectricGuitar(fretting=True, stiffness_reference="open", stiffness=0.00002)
    v = Voice(698.46, 77.0, 0.15, 0.7, {"string": 1, "stiffness": B})
    a = inst.voice(v, RenderContext(16000, 60), np.random.default_rng(5))
    b = replace(inst, stiffness_reference="vibrating").voice(v, RenderContext(16000, 60), np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


def test_implicit_string_and_pitch_expression_keep_the_same_precedence():
    inst = ElectricGuitar(fretting=True, stiffness_reference="open", stiffness=2.2e-5, dispersion_fit="bracketed")
    params = {"bend": 1.0, "bend_time": 0.1, "pitch_interpolation": "sinc", "vibrato": 0.1}
    voice = Voice(523.25, 72.0, 0.2, 0.7, params)
    ctx = RenderContext(16000, 60)
    a = inst.voice(voice, ctx, np.random.default_rng(9))
    other = replace(inst, stiffness_reference="vibrating")
    b = other.voice(
        replace(voice, params={**params, "stiffness": inst.stiffness / fret_scale(72.0) ** 2}),
        ctx,
        np.random.default_rng(9),
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("value", ["unknown", None, False])
def test_bad_reference_is_rejected(value):
    with pytest.raises(ValueError, match="stiffness_reference"):
        ElectricGuitar(stiffness_reference=value)


@pytest.mark.parametrize("value", [None, -1.0, np.nan, np.inf])
def test_open_B_validation_and_required_geometry(value):
    with pytest.raises(ValueError, match="stiffness"):
        ElectricGuitar(fretting=True, stiffness_reference="open", stiffness=value)
    with pytest.raises(ValueError, match="fretting"):
        ElectricGuitar(stiffness_reference="open")


def test_old_specs_and_new_IR_roundtrip_and_native_render(tmp_path):
    legacy = ElectricGuitar(fretting=True).to_spec()
    legacy.pop("stiffness_reference")
    assert build_instrument(legacy).stiffness_reference == "vibrating"
    inst = ElectricGuitar(fretting=True, stiffness_reference="open", stiffness=2.3e-5, dispersion_fit="bracketed")
    song = Song(Track("g", inst)(Note("G4", 0.15).param(string=1)), sample_rate=16000, loudness=None)
    path = tmp_path / "score.json"
    song.compile().to_json(path)
    score = Score.from_json(path)
    restored = build_instrument(score.track("g").instrument)
    assert restored.stiffness_reference == "open" and restored.stiffness == inst.stiffness
    result = render_score(score, workers=1)
    assert np.isfinite(result.audio).all() and result.stats["limiter_db"] == 0
