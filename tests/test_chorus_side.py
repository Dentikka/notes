import json

import numpy as np
import pytest
from notes import Chorus, Note, Song, Synth, Track
from notes.engine import RenderContext, render_score
from notes.engine.registry import build_effect
from notes.ir import Score


@pytest.mark.parametrize("sr", [8000, 44100])
@pytest.mark.parametrize("channels", [1, 2])
def test_side_return_matches_modulated_difference_and_preserves_mid(sr, channels):
    rng = np.random.default_rng(25)
    x = rng.normal(0, .1, (channels, sr // 3))
    ctx = RenderContext(sr, 120, offset=137)
    params = dict(rate=1.5, depth=.0003, base_delay=.012, mix=.48)
    # Independent route through the unchanged chorus produces the two delay returns.
    wet = Chorus(**params, dry=0).process(x, ctx) * (1 + .5 * params["mix"])
    direct = np.repeat(x, 2, axis=0) if channels == 1 else x
    expected = direct + wet - wet.mean(0, keepdims=True)
    y = Chorus(**params, side_only=True).process(x, ctx)
    np.testing.assert_allclose(y, expected, atol=1e-15, rtol=1e-14)
    np.testing.assert_allclose(y.mean(0), direct.mean(0), atol=1e-15, rtol=1e-14)
    assert np.linalg.norm(y - direct) > .01
    pure = Chorus(**params, side_only=True, dry=0).process(x, ctx)
    np.testing.assert_array_equal(pure[0], -pure[1])


def test_legacy_chorus_matches_explicit_delay_formula():
    sr = 8000
    ctx = RenderContext(sr, 120, offset=191)
    x = np.random.default_rng(8).normal(size=(2, sr))
    chorus = Chorus()
    idx = np.arange(sr, dtype=float)
    expected = []
    for ch in range(2):
        t = (idx + ctx.offset) / sr
        lfo = .5 * (1 + np.sin(2 * np.pi * chorus.rate * t + ch * np.pi / 2))
        lag = (chorus.base_delay + chorus.depth * lfo) * sr
        wet = np.interp(idx - lag, idx, x[ch], left=0.)
        expected.append((chorus.dry * x[ch] + chorus.mix * wet) / (1 + .5 * chorus.mix))
    np.testing.assert_array_equal(chorus.process(x, ctx), np.array(expected))


def test_side_mode_serializes_and_reaches_native_renderer():
    effect = Chorus(side_only=True, rate=1.5, depth=.0003, mix=.48)
    assert build_effect(json.loads(json.dumps(effect.to_spec()))) == effect
    assert not build_effect({"type": "chorus"}).side_only
    dry = Song(Track("probe", Synth())(Note(60, dur=.5)), bpm=60, sample_rate=8000, loudness=None).compile()
    wet = Song(Track("probe", Synth(), fx=(effect,))(Note(60, dur=.5)),
               bpm=60, sample_rate=8000, loudness=None).compile()
    decoded = Score.from_dict(json.loads(json.dumps(wet.to_dict())))
    a, b = render_score(dry, workers=1), render_score(decoded, workers=1)
    assert a.stats["limiter_db"] == b.stats["limiter_db"] == 0
    x, y = a.audio, b.audio
    n = min(x.shape[-1], y.shape[-1])
    # The renderer stores float32 PCM; averaging the two rounded channels adds one ulp.
    np.testing.assert_allclose(y.mean(0)[:n], x.mean(0)[:n], atol=3e-8, rtol=0)
    assert np.linalg.norm(y[0] - y[1]) > .001
