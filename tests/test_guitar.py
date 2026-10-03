import numpy as np
import pytest

from notes.engine import RenderContext
from notes.engine.effects.speaker import speaker_ir
from notes.engine.effects.tone import Cabinet
from notes.engine.string_model import steel_string

SR = 48000


def _peak_hz(y: np.ndarray, near: float, width: float) -> float:
    seg = y * np.hanning(y.size)
    spec = np.abs(np.fft.rfft(seg, 8 * seg.size))
    freqs = np.fft.rfftfreq(8 * seg.size, 1 / SR)
    band = (freqs > near - width) & (freqs < near + width)
    return float(freqs[band][np.argmax(spec[band])])


@pytest.mark.parametrize("freq", [82.41, 246.94, 830.6])
def test_steel_string_is_in_tune(freq):
    y = steel_string(freq, 2 * SR, SR, np.random.default_rng(0), second=0.0)
    assert abs(1200 * np.log2(_peak_hz(y[SR // 2 :], freq, 0.1 * freq) / freq)) < 2


def test_steel_string_decays_at_the_set_rate():
    y = steel_string(220.0, 3 * SR, SR, np.random.default_rng(0), t60=3.0, second=0.0)
    rms = [np.sqrt(np.mean(y[int(t * SR) : int(t * SR) + SR // 10] ** 2)) for t in (0.5, 1.5)]
    assert 20 * np.log10(rms[0] / rms[1]) == pytest.approx(20.0, abs=4.0)  # 60 dB in 3 s


def test_stiff_string_stretches_its_partials():
    y = steel_string(110.0, 2 * SR, SR, np.random.default_rng(0), stiffness=4e-5, second=0.0)
    h = 12  # near 1.5 kHz, where the dispersion is fitted: 5 cents sharp
    cents = 1200 * np.log2(_peak_hz(y[SR // 4 :], h * 110.0 * np.sqrt(1 + 4e-5 * h * h), 30.0) / (h * 110.0))
    assert cents == pytest.approx(1200 * np.log2(np.sqrt(1 + 4e-5 * h * h)), abs=1.0)


def test_greenback_cabinet_rolls_off_above_the_presence():
    mag = np.abs(np.fft.rfft(speaker_ir(SR), 8192))
    at = lambda f: 20 * np.log10(mag[int(round(f * 8192 / SR))])  # noqa: E731
    assert at(1000) == pytest.approx(0.0, abs=1e-6)
    assert at(2600) > at(1000) and at(8000) < at(1000) - 25


def test_cabinet_models():
    x = np.random.default_rng(0).normal(size=(2, SR // 4))
    ctx = RenderContext(sr=SR, bpm=120)
    for model in ("greenback", "filters"):
        y = Cabinet(model=model).process(x, ctx)
        assert y.shape == x.shape and np.isfinite(y).all()
    with pytest.raises(ValueError, match="cabinet model"):
        Cabinet(model="v30")
