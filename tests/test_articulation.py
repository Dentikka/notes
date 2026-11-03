import numpy as np
import pytest
from _audio import estimate_f0, inharmonic_ratio

from notes import (
    AcousticGuitar,
    BassGuitar,
    DrumKit,
    E,
    ElectricGuitar,
    Note,
    NylonGuitar,
    Pluck,
    Q,
    Song,
    SynthGuitar,
    Track,
    melody,
)
from notes.engine import RenderContext, Voice
from notes.engine.articulation import pitch_curve, variable_rate_read
from notes.engine.dsp import karplus_strong
from notes.engine.effects.tone import saturate
from notes.lang.perform import perform

SR = 44100


def test_plain_notes_have_no_curve():
    assert pitch_curve({}, 100, SR) is None
    assert pitch_curve({"palm_mute": True}, 100, SR) is None


def test_bend_slide_and_vibrato_shapes():
    bend = pitch_curve({"bend": 2.0, "bend_time": 0.1}, SR, SR)
    assert bend[0] == 0.0 and bend[-1] == pytest.approx(2.0)
    slide = pitch_curve({"slide": -3.0, "slide_time": 0.05}, SR, SR)
    assert slide[0] == pytest.approx(-3.0) and slide[-1] == 0.0
    vib = pitch_curve({"vibrato": 0.3, "vibrato_delay": 0.2}, SR, SR)
    assert np.all(vib[: int(0.2 * SR)] == 0.0)
    assert np.max(np.abs(vib)) <= 0.3 + 1e-12
    assert abs(vib[int(0.5 * SR) :].mean()) < 0.02


def test_bent_string_lands_on_the_target_pitch():
    rng = np.random.default_rng(0)
    curve = pitch_curve({"bend": 2.0, "bend_time": 0.1}, SR, SR)
    y = variable_rate_read(lambda m: karplus_strong(220.0, m, SR, rng, t60=4.0), curve)
    target = 220.0 * 2 ** (2 / 12)
    cents = 1200 * np.log2(estimate_f0(y[int(0.3 * SR) : int(0.8 * SR)], SR) / target)
    assert len(y) == SR and abs(cents) < 5


def test_expressive_guitar_voice_is_finite():
    ctx = RenderContext(sr=SR, bpm=120)
    v = Voice(freq=330.0, pitch=64.0, dur=0.8, vel=0.8, params={"vibrato": 0.3, "bend": 1.0, "slide": -2.0})
    y = ElectricGuitar().voice(v, ctx, np.random.default_rng(0))
    assert len(y) == int(round(0.8 * SR)) + int(round(0.06 * SR))
    assert np.isfinite(y).all()


def test_melody_notation_marks_bends_and_vibrato():
    evs = perform(melody("D5^2~@4 E5^ F5^-1 G5~", step=E)).events
    assert [e.params for e in evs] == [{"bend": 2.0, "vibrato": 0.25}, {"bend": 2.0}, {"bend": -1.0}, {"vibrato": 0.25}]
    assert [e.pitch for e in evs] == [74, 76, 77, 79]
    assert perform(melody("~ E5")).events[0].time == 0.5  # a bare '~' is still a rest


def test_articulation_methods_set_params():
    e = perform(Note("E5", Q).vibrato(0.4, rate=6).bend(1, time=0.2).slide(-2)).events[0]
    assert e.params["vibrato"] == 0.4 and e.params["vibrato_rate"] == 6.0
    assert e.params["bend"] == 1.0 and e.params["bend_time"] == 0.2
    assert e.params["slide"] == -2.0


def test_doubled_part_is_two_panned_takes():
    rhythm = Track("rhythm", ElectricGuitar(drive=0.8))
    part = melody("E2 E2 G2 A2", step=E)
    score = Song(rhythm.doubled(part, spread=0.7) | Track("d", DrumKit())(Note(36, Q))).compile()
    left, right = score.track("rhythm_L"), score.track("rhythm_R")
    assert (left.pan, right.pan) == (-0.7, 0.7)
    times_l = [e.time for e in score.events_of("rhythm_L")]
    times_r = [e.time for e in score.events_of("rhythm_R")]
    assert len(times_l) == len(times_r) == 4 and times_l != times_r


def test_oversampled_clipping_folds_far_less_back():
    t = np.arange(SR) / SR
    x = 0.8 * np.sin(2 * np.pi * 5000.0 * t)
    naive = inharmonic_ratio(saturate(x, 30.0, 0.1, oversample=1), 5000.0, SR)
    oversampled = inharmonic_ratio(saturate(x, 30.0, 0.1, oversample=4), 5000.0, SR)
    assert oversampled < naive / 100  # at least 20 dB less aliasing


def test_pushed_vibrato_only_raises_the_pitch():
    rng = np.random.default_rng(1)
    vib = pitch_curve({"vibrato": 0.3, "vibrato_delay": 0.1}, SR, SR, vibrato_shape="push", rng=rng)
    assert vib.min() >= 0.0 and vib.max() == pytest.approx(0.6, abs=0.01)
    with pytest.raises(ValueError, match="vibrato_shape"):
        pitch_curve({"vibrato": 0.3}, SR, SR, vibrato_shape="square")


def test_sinc_pitch_read_tracks_analytic_tone_without_linear_interpolation_coloration():
    sr, freq, n = 22050, 4000., 8192
    curve = np.full(n, .35)
    ratio = 2**(curve/12)
    positions = np.r_[0., np.cumsum(ratio[:-1])]
    source = lambda m: np.sin(2*np.pi*freq*np.arange(m)/sr)  # noqa: E731
    expected = np.sin(2*np.pi*freq*positions/sr)[128:-128]
    errors = {}
    for mode in ("linear", "sinc"):
        actual = variable_rate_read(source, curve, interpolation=mode)[128:-128]
        errors[mode] = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    assert errors["sinc"] < 1e-3
    assert errors["sinc"] < errors["linear"]/100


def test_sinc_pitch_up_filters_components_that_would_alias():
    sr, n = 24000, 8192
    def source(m):
        t = np.arange(m)/sr
        return np.sin(2*np.pi*1000*t) + np.sin(2*np.pi*10000*t)
    curve = np.full(n, 12.)
    expected = np.sin(2*np.pi*2000*np.arange(n)/sr)[128:-128]
    old = variable_rate_read(source, curve)[128:-128]
    new = variable_rate_read(source, curve, interpolation="sinc")[128:-128]
    assert np.linalg.norm(new-expected) < np.linalg.norm(old-expected)/1000


def test_sinc_identity_and_impulse_timing():
    source = lambda m: np.sin(np.arange(m)*.73)  # noqa: E731
    assert np.array_equal(variable_rate_read(source, np.zeros(1000)),
                          variable_rate_read(source, np.zeros(1000), interpolation="sinc"))
    def pulse(m):
        y = np.zeros(m)
        y[1000] = 1
        return y
    y = variable_rate_read(pulse, np.full(2000, 12.), interpolation="sinc")
    assert abs(y).argmax() == 500


@pytest.mark.parametrize("instrument", [
    Pluck(), ElectricGuitar(amplified=False), BassGuitar(), AcousticGuitar(), NylonGuitar(),
    AcousticGuitar(twelve=True), AcousticGuitar(model="coupled"), SynthGuitar(),
])
def test_string_instruments_forward_sinc_per_note_control(instrument):
    ctx = RenderContext(sr=22050, bpm=60)
    params = {"pitch_points": [[0, .2], [.3, .35]], "pitch_interpolation": "sinc"}
    voice = Voice(330., 64., .4, .7, params)
    y = instrument.voice(voice, ctx, np.random.default_rng(17))
    assert len(y) == round((voice.dur+instrument.release)*ctx.sr)
    assert np.isfinite(y).all() and abs(y).max() > 1e-6
    with pytest.raises(ValueError, match="interpolation"):
        instrument.voice(Voice(330., 64., .4, .7, {**params, "pitch_interpolation": "invalid"}),
                         ctx, np.random.default_rng(17))


def test_pitch_reader_rejects_invalid_curves_and_bounds_sinc_support():
    def source(m):
        return np.ones(m)
    for curve in (np.array([np.nan]), np.zeros((2, 2))):
        with pytest.raises(ValueError, match="pitch curve"):
            variable_rate_read(source, curve)
    with pytest.raises(ValueError, match="four octaves"):
        variable_rate_read(source, np.full(10, 49.), interpolation="sinc")
    assert variable_rate_read(source, np.array([]), interpolation="sinc").size == 0
