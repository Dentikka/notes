import numpy as np
import pytest
from notes import Cello, Piano, Song, Track, Violin, melody
from notes.engine import RenderContext, Voice, build_instrument


def render(instrument, vel=0.8, params=None, sr=22050):
    return instrument.voice(Voice(220, 57, 0.7, vel, params or {}),
                            RenderContext(sr, 60), np.random.default_rng(7))


@pytest.mark.parametrize("instrument", [Piano(), Violin(vibrato=0), Cello(vibrato=0)])
def test_tuning_roundtrip_and_dynamics(instrument):
    assert build_instrument(instrument.to_spec()) == instrument
    y = render(instrument)
    assert np.isfinite(y).all()
    spectrum = abs(np.fft.rfft(y[:11025] * np.hanning(11025)))
    freq = np.fft.rfftfreq(11025, 1 / 22050)
    mask = (freq > 200) & (freq < 240)
    assert freq[mask][spectrum[mask].argmax()] == pytest.approx(220, abs=2)
    assert np.sqrt(np.mean(render(instrument, 0.3) ** 2)) < np.sqrt(np.mean(y ** 2)) * 0.5
    assert np.array_equal(y, render(instrument))
    assert abs(y[-1]) < 1e-8


def test_pedal_preserves_string_after_key_release():
    piano = Piano(sustain=2)
    dry = render(piano)
    pedal = render(piano, params={"pedal": True})
    assert len(pedal) > len(dry)
    assert np.sqrt(np.mean(pedal[22050:33075] ** 2)) > 1e-4


def test_bowed_sustain_and_explicit_vibrato_override():
    y = render(Violin(vibrato=0))
    assert np.sqrt(np.mean(y[11025:13230] ** 2)) > 0.03
    assert not np.array_equal(y, render(Violin(vibrato=0), params={"vibrato": 0.4}))


@pytest.mark.parametrize("cls", [Piano, Violin, Cello])
def test_dsl_render(cls):
    result = Song(Track("solo", cls())(melody("A3 C4 E4", step=0.25)),
                  sample_rate=22050, loudness=None).render()
    assert np.isfinite(result.audio).all()


@pytest.mark.parametrize("kwargs", [{"release": 0}, {"level": float("nan")}])
def test_invalid_parameters(kwargs):
    for cls in (Piano, Violin, Cello):
        with pytest.raises(ValueError):
            cls(**kwargs)
