import json
import os

import numpy as np
import pytest

from notes.ears.mert import MERT_RATE, MertEar, _trim_event, to_mert_rate


def test_music_audio_resampling_and_channel_layout():
    sr = 44100
    t = np.arange(sr)/sr
    x = np.sin(2*np.pi*220*t)
    converted = to_mert_rate(np.stack((x, x)), sr)
    assert converted.shape == (MERT_RATE,) and converted.dtype == np.float32
    target = np.sin(2*np.pi*220*np.arange(MERT_RATE)/MERT_RATE)
    assert np.max(abs(converted[100:-100]-target[100:-100])) < .003


@pytest.mark.parametrize("x", [np.zeros(24000), np.ones(24000), np.full(24000, np.nan),
                             np.zeros((0, 24000)), np.zeros((2, 2, 24000)), np.zeros(100)])
def test_invalid_or_silent_music_audio_is_rejected(x):
    with pytest.raises(ValueError):
        to_mert_rate(x, MERT_RATE)


@pytest.mark.parametrize("trim", [False, True])
def test_cli_adds_optional_music_similarity_without_loading_a_model(tmp_path, monkeypatch, trim):
    from notes.cli import main
    from notes.engine.audio import write_wav

    class FixedEar:
        model_name = "test-mert"
        revision = "fixed-test"

        def __init__(self, local_path=None):
            assert local_path == "offline-cache"

        def embed_audio(self, x, sr):
            assert x.ndim == 2 and sr == 44100
            return np.array([1., 0.])

        def similarity(self, comp, *, trim_edges=False):
            assert trim_edges == trim
            assert comp.reference_context is not None and comp.render_context is not None
            return float(self.embed_audio(comp.reference_context, 44100)@self.embed_audio(comp.render_context, 44100))

    monkeypatch.setattr("notes.ears.MertEar", FixedEar)
    t = np.arange(44100)/44100
    x = np.sin(2*np.pi*220*t)*np.exp(-t*3)
    reference = write_wav(tmp_path/"reference.wav", x, 44100)
    candidate = write_wav(tmp_path/"candidate.wav", x*.5, 44100)
    args = ["compare", str(reference), str(candidate), "--mert", "--mert-path", "offline-cache", "--no-ab"]
    assert main(args + (["--mert-trim"] if trim else [])) == 0
    report = json.loads((tmp_path/"candidate.vs.reference.json").read_text())
    assert report["mert"]["cosine"] == 1
    assert report["mert"]["status"] == "diagnostic-only"
    assert report["mert"]["trim_quiet_edges"] == trim


def test_single_event_trimming_retains_the_event_and_is_gain_invariant():
    x = np.zeros(24000*2)
    t = np.arange(2400)/24000
    x[24000:26400] = np.sin(2*np.pi*220*t)*np.exp(-t*15)
    original = x.copy()
    trimmed = _trim_event(x)
    assert 2400 <= len(trimmed) < 5000
    assert np.max(abs(trimmed)) == np.max(abs(x))
    assert np.array_equal(x, original)
    assert np.allclose(trimmed, _trim_event(x*.25)*4)


@pytest.mark.skipif(not os.environ.get("NOTES_MERT_MODEL_PATH"), reason="explicit local MERT checkpoint not supplied")
def test_cached_native_checkpoint_embeddings_are_deterministic_and_gain_invariant():
    import torch

    torch.set_num_threads(2)
    ear = MertEar(os.environ["NOTES_MERT_MODEL_PATH"])
    t = np.arange(24000)/24000
    x = np.sin(2*np.pi*220*t)+.3*np.sin(2*np.pi*440*t)
    a = ear.embed_audio(x, 24000)
    b = ear.embed_audio(x, 24000)
    scaled = ear.embed_audio(x*.25, 24000)
    assert a.shape == (768,) and np.isfinite(a).all()
    assert np.linalg.norm(a) == pytest.approx(1, abs=1e-5)
    assert np.allclose(a, b, atol=1e-6) and float(a@scaled) > .999
    event = ear.embed_audio(np.pad(x, (24000, 24000)), 24000, trim_edges=True)
    assert np.isfinite(event).all() and np.linalg.norm(event) == pytest.approx(1, abs=1e-5)
    from notes.ears.compare import compare

    comparison = compare(x, x*.5, sr=24000, max_lag=0, preserve_context=True)
    assert ear.similarity(comparison, trim_edges=True) > .999
