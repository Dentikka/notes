import numpy as np
import pytest
from _audio import estimate_f0, inharmonic_ratio

from notes import DrumKit, E, ElectricGuitar, Note, Q, Song, Track, melody
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
