import numpy as np
import pytest

from notes import Song, SynthGuitar, Track, melody
from notes.engine import RenderContext, Voice

SR = 44100


def _note(g, pitch, seconds=1.0, **params):
    ctx = RenderContext(sr=SR, bpm=60)
    return g.voice(Voice(440.0 * 2 ** ((pitch - 69) / 12), pitch, seconds, 0.8, params), ctx, np.random.default_rng(0))


def _centroid(y):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    return float(np.sum(freqs * spec) / np.sum(spec))


def _peak(y, low, high):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    band = (freqs > low) & (freqs < high)
    return float(freqs[band][np.argmax(spec[band])])


def test_synth_guitar_keeps_the_first_groove_sound():
    # Rendered with the engine of First Groove (cbd8640): the restored guitar matched that
    # render to -130 dB, so these numbers pin the sound against changes to shared helpers.
    lead = Track("lead", SynthGuitar(drive=0.65, tone=0.6, pickup="neck"))
    audio = Song(lead(melody("A4 C5 E5 _ D5 C5 D5 E5")), bpm=112, loudness=None).render().audio.mean(axis=0)
    assert 20 * np.log10(np.sqrt(np.mean(audio**2))) == pytest.approx(-22.08, abs=0.02)
    assert _centroid(audio) == pytest.approx(2134.2, rel=0.002)


def test_palm_mute_chugs_are_short_and_dark():
    g = SynthGuitar(drive=0.85, tone=0.45)
    open_, muted = _note(g, 45), _note(g, 45, palm_mute=True)
    late = slice(int(0.4 * SR), int(0.9 * SR))
    assert np.sqrt(np.mean(muted[late] ** 2)) < 0.1 * np.sqrt(np.mean(open_[late] ** 2))
    attack = slice(0, int(0.1 * SR))
    assert _centroid(muted[attack]) < _centroid(open_[attack])


def test_bend_raises_the_pitch():
    g = SynthGuitar(pickup="neck")
    y = _note(g, 69, seconds=1.0, bend=2.0, bend_time=0.1, bend_start=0.3)
    assert _peak(y[: int(0.25 * SR)], 400, 560) == pytest.approx(440.0, rel=0.01)
    assert _peak(y[int(0.5 * SR) : int(0.95 * SR)], 400, 560) == pytest.approx(493.9, rel=0.01)


def test_unknown_pickup_is_rejected():
    with pytest.raises(ValueError, match="pickup"):
        SynthGuitar(pickup="humbucker")
