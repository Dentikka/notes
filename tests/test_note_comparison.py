import numpy as np

from notes.ears.note_comparison import compare_notes
from notes.ears.transcribe import TrackedNote


def _note(start, end, pitch):
    return TrackedNote(start, end, pitch, -20, np.array([start, end]), np.array([pitch, pitch]))


def test_same_note_histogram_with_wrong_rhythm_does_not_pass():
    reference = [_note(0, 0.4, 60), _note(0.5, 0.9, 64), _note(1, 1.4, 67)]
    wrong = [_note(0.2, 0.6, 60), _note(0.7, 1.1, 64), _note(1.2, 1.6, 67)]
    assert compare_notes(reference, wrong)["onset_pitch"]["f1"] == 0
    assert compare_notes(reference, reference)["onset_pitch_offset"]["f1"] == 1


def test_held_note_split_into_four_plucks_is_penalised():
    reference = [_note(0, 1, 60)]
    repeated = [_note(t, t + 0.25, 60) for t in (0, 0.25, 0.5, 0.75)]
    report = compare_notes(reference, repeated)
    assert report["onset_pitch"]["precision"] == 0.25
    assert report["onset_pitch_offset"]["f1"] == 0


def test_octaves_and_empty_results_are_not_accepted_as_the_same_notes():
    reference = [_note(0, 1, 60)]
    assert compare_notes(reference, [_note(0, 1, 72)])["onset_pitch"]["f1"] == 0
    assert compare_notes(reference, [])["onset_pitch"]["recall"] == 0
