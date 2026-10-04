import numpy as np
import pytest

from notes import AcousticGuitar, ElectricGuitar, Song, Track, melody
from notes.engine import RenderContext, Voice
from notes.engine.effects.body import body_ir

SR = 48000


def _note(g, pitch, string=None, seconds=2.0):
    ctx = RenderContext(sr=SR, bpm=60)
    params = {} if string is None else {"string": string}
    return g.voice(Voice(440.0 * 2 ** ((pitch - 69) / 12), pitch, seconds, 0.8, params), ctx, np.random.default_rng(0))


def _level_at(y, f):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    return 20 * np.log10(spec[np.abs(freqs - f) < 3].max())


def test_acoustic_guitar_is_in_tune():
    y = _note(AcousticGuitar(), 57)[: SR]  # A3
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    low = (freqs > 150) & (freqs < 300)
    assert freqs[low][np.argmax(spec[low])] == pytest.approx(220.0, rel=0.01)


def test_twelve_strings_add_octaves_low_and_unisons_high():
    six, twelve = AcousticGuitar(), AcousticGuitar(twelve=True)

    def octave_over_fundamental(y, f):
        return _level_at(y, 2 * f) - _level_at(y, f)

    # A2 is on an octave course: the octave string lifts the second harmonic
    assert octave_over_fundamental(_note(twelve, 45), 110.0) > octave_over_fundamental(_note(six, 45), 110.0) + 2.0
    # E4 is on a unison course: no octave added
    assert octave_over_fundamental(_note(twelve, 64), 329.6) == pytest.approx(
        octave_over_fundamental(_note(six, 64), 329.6), abs=3.0)
    # B3 played on the G string is on an octave course after all
    on_g, as_if_plain = _note(twelve, 59, string=3), _note(twelve, 59)
    assert octave_over_fundamental(on_g, 246.9) > octave_over_fundamental(as_if_plain, 246.9) + 2.0


def test_body_rings_at_the_sound_hole():
    ir = body_ir(SR)
    mag = np.abs(np.fft.rfft(ir, 1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1 / SR)
    at = lambda f: 20 * np.log10(mag[np.argmin(np.abs(freqs - f))])  # noqa: E731
    assert at(98.0) > at(70.0) + 5.0 and at(98.0) > at(140.0) + 5.0  # the Helmholtz resonance
    around_1k = 20 * np.log10(np.sqrt(np.mean(mag[(freqs > 890) & (freqs < 1120)] ** 2)))
    assert around_1k == pytest.approx(0.0, abs=0.5)


def test_acoustic_sits_near_the_electric_in_level():
    def loudness(instrument):
        return Song(Track("g", instrument)(melody("[E3,B3,E4,G#4]@4", step=1)), bpm=60).render().stats["tracks"]["g"][
            "lufs_in_mix"]

    assert abs(loudness(AcousticGuitar()) - loudness(ElectricGuitar(drive=0.06, tone=0.5))) < 8.0
