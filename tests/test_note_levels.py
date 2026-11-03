import numpy as np
import pytest

from notes.ears.transcribe import TrackedNote, measure_note_levels, tracked_music
from notes.lang.perform import perform


def _note(start, end, pitch):
    return TrackedNote(start, end, pitch, -1, np.array([start, end]), np.array([pitch, pitch]))


def test_note_dynamics_follow_recorded_amplitude_not_identical_model_confidence():
    sr = 16000
    t = np.arange(sr * 2) / sr
    x = np.zeros(len(t))
    notes = [_note(.2, .6, 57), _note(1, 1.4, 57)]
    for note, amplitude in zip(notes, (1, .25), strict=True):
        mask = (t >= note.start) & (t < note.end)
        x[mask] += amplitude * np.sin(2 * np.pi * 220 * t[mask])
    measured = measure_note_levels(x, sr, notes)
    assert measured[0].level_db - measured[1].level_db == pytest.approx(12.04, abs=.5)
    assert notes[0].level_db == notes[1].level_db == -1
    assert [(n.start, n.end, n.pitch) for n in measured] == [(n.start, n.end, n.pitch) for n in notes]
    events = perform(tracked_music(measured, peak_normalize=True)).events
    assert events[0].vel == 1
    assert events[1].vel == pytest.approx(.25 ** (2 / 3), abs=.02)


def test_loud_octave_does_not_inflate_the_lower_notes_level():
    sr = 16000
    t = np.arange(sr) / sr
    mask = (t >= .2) & (t < .7)
    x = mask * (.05 * np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 440 * t))
    low, high = measure_note_levels(x, sr, [_note(.2, .7, 57), _note(.2, .7, 69)])
    assert high.level_db - low.level_db > 22


def test_note_at_start_uses_silence_before_audio_and_has_finite_level():
    sr = 16000
    t = np.arange(sr) / sr
    measured = measure_note_levels(np.sin(2 * np.pi * 220 * t), sr, [_note(0, .8, 57)])
    assert np.isfinite(measured[0].level_db) and measured[0].level_db > -100


def test_silence_does_not_turn_model_candidates_into_loud_notes():
    notes = measure_note_levels(np.zeros(16000), 16000, [_note(.2, .7, 57)])
    assert perform(tracked_music(notes, peak_normalize=True)).events[0].vel == 0


def test_isolated_quieter_repeat_keeps_its_sounding_level():
    sr = 16000
    t = np.arange(round(.3 * sr)) / sr
    x = np.r_[np.sin(2 * np.pi * 200 * t), .7 * np.sin(2 * np.pi * 200 * t)]
    original = x.copy()
    notes = [_note(0, .3, 55), _note(.3, .6, 55)]
    measured = measure_note_levels(x, sr, notes, method="rms")
    assert measured[1].level_db - measured[0].level_db == pytest.approx(20 * np.log10(.7), abs=.01)
    assert measured[1].level_db > -10
    assert [(n.start, n.end, n.pitch) for n in measured] == [(n.start, n.end, n.pitch) for n in notes]
    assert notes[0].level_db == notes[1].level_db == -1
    np.testing.assert_array_equal(x, original)
    legacy = measure_note_levels(x, sr, notes)
    explicit = measure_note_levels(x, sr, notes, method="growth")
    assert [n.level_db for n in legacy] == [n.level_db for n in explicit]
    assert legacy[1].level_db == -200  # Growth cannot represent a quieter sustained repeat.


@pytest.mark.parametrize("duration", [.005, .02, .08])
def test_isolated_short_note_does_not_measure_the_loud_note_that_follows(duration):
    sr = 16000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 200 * t) * np.where(t < duration, .1, 1.)
    measured = measure_note_levels(x, sr, [_note(0, duration, 55)], method="rms")
    assert measured[0].level_db == pytest.approx(10 * np.log10(.1**2 / 2), abs=.02)


@pytest.mark.parametrize("dc", [0., 2.])
def test_isolated_silence_or_dc_is_silent_and_out_of_range_note_is_silent(dc):
    notes = [_note(.2, .5, 55), _note(2, 3, 55)]
    measured = measure_note_levels(np.full(16000, dc), 16000, notes, method="rms")
    assert [n.level_db for n in measured] == [-200, -200]


@pytest.mark.parametrize("start,end", [(-1, .1), (.1, .1), (.2, .1), (np.nan, .1), (0, np.inf)])
def test_isolated_levels_reject_invalid_note_spans(start, end):
    with pytest.raises(ValueError, match="starts and positive durations"):
        measure_note_levels(np.zeros(16000), 16000, [_note(start, end, 55)], method="rms")


def test_unknown_level_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        measure_note_levels(np.zeros(16000), 16000, [], method="unknown")
