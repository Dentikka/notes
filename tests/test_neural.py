import numpy as np
import pytest

from notes.ears.neural import NeuralTrack, notes_from_probabilities


def _prediction():
    t = np.arange(100) * 0.01
    note = np.zeros((100, 88))
    note[:, 64 - 21] = 0.8
    onset = np.zeros_like(note)
    onset[[25, 50, 75], 72 - 21] = 0.9  # accompaniment, not the held E4
    contour = np.zeros((100, 264))
    contour[:, (64 - 21) * 3] = 0.9
    return NeuralTrack(t, note, onset, contour, 1.0)


def test_pitch_specific_predictions_do_not_split_lead_at_other_voices_attacks():
    notes = notes_from_probabilities(_prediction(), low=64, high=64)
    assert len(notes) == 1
    assert notes[0].start == 0 and notes[0].end == pytest.approx(1)
    assert np.all(notes[0].contour == 64)


def test_pitch_specific_repick_is_preserved():
    track = _prediction()
    track.onset[50, 64 - 21] = 0.9
    notes = notes_from_probabilities(track, low=64, high=64)
    assert len(notes) == 2
    assert notes[1].start == pytest.approx(0.5)


def test_silent_frames_do_not_become_notes_even_with_false_onsets():
    track = _prediction()
    track.note[:] = 0
    assert notes_from_probabilities(track) == []
