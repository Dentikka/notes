"""Ears: pure helpers always; the CLAP model only when it is installed and already cached."""

import numpy as np
import pytest

from notes.ears.clap import CLAP_SAMPLE_RATE, DEFAULT_MODEL, to_clap_rate, windows

SR = CLAP_SAMPLE_RATE


def test_windows_are_full_length_and_cover_the_tail():
    x = np.arange(int(28 * SR), dtype=float)
    ws = windows(x)
    assert all(len(w) == 10 * SR for w in ws)
    assert [int(w[0]) // SR for w in ws] == [0, 5, 10, 15, 18]  # the last one is aligned to the end
    assert int(ws[-1][-1]) == len(x) - 1


def test_short_input_is_one_window_and_a_short_tail_is_dropped():
    assert len(windows(np.zeros(3 * SR))) == 1
    assert [int(w[0]) // SR for w in windows(np.arange(int(21 * SR), dtype=float))] == [0, 5, 10]


def test_resampling_to_48k_mixes_to_mono():
    stereo = np.stack([np.ones(44100), np.zeros(44100)])
    y = to_clap_rate(stereo, 44100)
    assert len(y) == 48000
    assert np.allclose(y[1000:-1000], 0.5, atol=1e-3)


def _cached(model: str) -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model, "config.json"), str)


needs_model = pytest.mark.skipif(not _cached(DEFAULT_MODEL), reason="CLAP model or torch/transformers not available")


@pytest.fixture(scope="module")
def ear():
    from notes.ears import Ear

    return Ear()


@needs_model
def test_text_embeddings_are_not_collapsed(ear):
    """Guards against checkpoints like laion/larger_clap_music whose prompts all embed alike."""
    e = ear.embed_texts(["a rock song with distorted guitar", "a calm piano piece", "a dog barking", "techno"])
    off_diagonal = (e @ e.T)[~np.eye(4, dtype=bool)]
    assert off_diagonal.max() < 0.9


@needs_model
def test_listening_is_deterministic_and_sectioned(ear):
    from notes import DrumKit, Song, Track, steps

    beat = Track("d", DrumKit())(steps(kick="x...x...", snare="....x...", hat="x.x.x.x."))
    song = Song((beat * 4).named("a"), (beat * 4).named("b"), bpm=120)
    result = song.render()
    first = ear.listen(result.audio, result.sample_rate, song.compile(), prompts=["drums"])
    second = ear.listen(result.audio, result.sample_rate, song.compile(), prompts=["drums"])
    assert first == second
    assert [s["name"] for s in first["sections"]] == ["whole", "a", "b"]
    assert set(first["sections"][0]["tags"]) == {"genre", "instrument", "mood", "production"}
