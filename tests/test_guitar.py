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


def test_fretted_notes_move_the_pickup_along_the_string():
    from notes.engine.instruments.strings import _along, fret_scale

    assert fret_scale(64) == 1.0 and fret_scale(76) == pytest.approx(0.5)  # open high E, 12th fret
    assert _along(0.27, fret_scale(76)) == pytest.approx(0.46)  # the neck pickup near mid-string


def test_prompt_sound_gives_a_two_stage_decay():
    def level(y, t):
        return 20 * np.log10(np.sqrt(np.mean(y[int(t * SR) : int(t * SR) + SR // 20] ** 2)))

    rng = np.random.default_rng(0)
    one = steel_string(330.0, 2 * SR, SR, rng, t60=6.0)
    two = steel_string(330.0, 2 * SR, SR, np.random.default_rng(0), t60=6.0, second=0.4, prompt_t60=0.9)
    early = lambda y: level(y, 0.02) - level(y, 0.5)  # noqa: E731
    late = lambda y: level(y, 1.0) - level(y, 1.5)  # noqa: E731
    assert early(two) > early(one) + 3 and late(two) == pytest.approx(late(one), abs=1.5)


def test_legato_note_has_no_pick_and_starts_softer():
    from notes import ElectricGuitar
    from notes.engine import Voice

    ctx = RenderContext(sr=SR, bpm=120)
    g = ElectricGuitar(drive=0.0)
    picked = g.voice(Voice(330.0, 64.0, 0.5, 0.8, {}), ctx, np.random.default_rng(0))
    legato = g.voice(Voice(330.0, 64.0, 0.5, 0.8, {"legato": True}), ctx, np.random.default_rng(0))
    first = slice(0, int(0.01 * SR))
    assert np.sqrt(np.mean(legato[first] ** 2)) < 0.7 * np.sqrt(np.mean(picked[first] ** 2))


def test_softer_picking_sounds_darker_only_when_asked():
    from notes import ElectricGuitar
    from notes.engine import Voice

    def centroid(y):
        mag = np.abs(np.fft.rfft(y[: SR // 4]))
        return float((np.fft.rfftfreq(SR // 4, 1 / SR) * mag).sum() / mag.sum())

    ctx = RenderContext(sr=SR, bpm=120)

    def note(vel, coupling):
        g = ElectricGuitar(drive=0.0, velocity_brightness=coupling)
        return g.voice(Voice(330.0, 64.0, 0.5, vel, {}), ctx, np.random.default_rng(0))

    assert np.allclose(note(1.0, 1.0), note(1.0, 0.0))  # full velocity: unchanged
    assert np.allclose(note(0.3, 0.0), note(0.3, 0.0))
    assert centroid(note(0.3, 1.0)) < 0.9 * centroid(note(0.3, 0.0))


def test_wound_strings_can_be_darker():
    from notes import ElectricGuitar
    from notes.engine import Voice

    def treble(y):
        f, mag = np.fft.rfftfreq(SR // 2, 1 / SR), np.abs(np.fft.rfft(y[: SR // 2]))
        return float(mag[f > 2000].sum() / mag.sum())

    ctx = RenderContext(sr=SR, bpm=120)
    dark = ElectricGuitar(drive=0.0, wound_treble=0.2, wound_brightness=0.5)
    plain = ElectricGuitar(drive=0.0)

    def note(g, pitch, string=None):
        params = {} if string is None else {"string": string}
        return g.voice(Voice(440.0 * 2 ** ((pitch - 69) / 12), pitch, 0.5, 0.8, params), ctx, np.random.default_rng(0))

    assert treble(note(dark, 50.0)) < 0.7 * treble(note(plain, 50.0))  # D3: only the D string or lower
    assert np.allclose(note(dark, 64.0), note(plain, 64.0))  # E4 with no string: taken as plain
    assert treble(note(dark, 59.0, string=4)) < 0.7 * treble(note(plain, 59.0, string=4))  # B3 on the D string
    assert np.allclose(note(dark, 59.0, string=2), note(plain, 59.0, string=2))  # open B: plain
