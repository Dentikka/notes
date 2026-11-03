import urllib.request
from pathlib import Path

import numpy as np
import pytest

from notes.ears.aesthetics import AestheticsEar, to_aesthetics_rate


def test_audio_frontend_preserves_channels_frequency_and_input():
    sr = 44100
    wave = np.sin(2*np.pi*220*np.arange(sr)/sr)
    original = wave.copy()
    converted = to_aesthetics_rate(np.stack((wave, wave)), sr, normalize_lufs=None)
    expected = np.sin(2*np.pi*220*np.arange(16000)/16000)
    assert converted.dtype == np.float32 and converted.shape == (16000,)
    assert max(abs(converted[100:-100]-expected[100:-100])) < .003
    assert np.array_equal(wave, original)
    quiet = to_aesthetics_rate(wave*.25, sr)
    assert np.allclose(quiet, to_aesthetics_rate(wave, sr), atol=1e-7)


def test_short_duration_and_high_crest_are_preserved_without_pcm_clipping():
    sr = 44100
    wave = .001*np.sin(2*np.pi*440*np.arange(sr//5)/sr)
    wave[100] = .4
    converted = to_aesthetics_rate(wave, sr)
    assert len(converted) == 3200
    assert max(abs(converted)) > 1


@pytest.mark.parametrize("wave", [np.zeros(16000), np.ones(16000), np.full(16000, np.nan),
                                   np.zeros((0, 16000)), np.zeros((2, 2, 16000)), np.zeros(100)])
def test_invalid_or_silent_quality_inputs_are_rejected(wave):
    with pytest.raises(ValueError):
        to_aesthetics_rate(wave, 16000)


def test_model_loading_never_implicitly_downloads(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected network request")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    cache = tmp_path/"absent-cache"
    with pytest.raises(FileNotFoundError, match="fetch_aesthetics"):
        AestheticsEar(cache)
    assert not cache.exists()


def test_modified_backend_is_rejected_before_execution(tmp_path):
    cache = tmp_path/"cache"
    (cache/"source").mkdir(parents=True)
    (cache/"source/LICENSE").write_text("unverified content", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned revision"):
        AestheticsEar(cache)


@pytest.mark.skipif(not (Path.home()/".cache/notes-audiobox-aesthetics/model/model.safetensors").exists(),
                    reason="optional pinned Audiobox checkpoint is not cached")
def test_actual_model_matches_official_calibration_and_preserves_known_piano_preference():
    torch = pytest.importorskip("torch")
    from notes.engine.hammer import struck_string

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        ear = AestheticsEar()
        assert ear.weight_keys == 324 and ear.parameter_count == 103859112
        rejected = struck_string(261.6255653005986, 88200, 44100, .8, radiation=False)
        accepted = struck_string(261.6255653005986, 88200, 44100, .8, radiation=True)
        before = accepted.copy()
        a = ear.score_audio(rejected, 44100)
        b = ear.score_audio(accepted, 44100)
        assert set(b) == {"CE", "CU", "PC", "PQ"}
        assert b["PQ"] == pytest.approx(7.36947799, abs=1e-4)
        assert b["PQ"] > a["PQ"]+.5
        assert np.array_equal(accepted, before)
        with pytest.raises(ValueError, match="silent"):
            ear.score_audio(np.zeros(16000), 16000)
    finally:
        torch.set_num_threads(previous_threads)
