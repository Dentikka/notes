import numpy as np
import pytest

from notes.ears.neural import NeuralTrack, notes_from_onsets, notes_from_probabilities


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


def test_confidence_dropout_cannot_restrike_an_onset_decoded_note():
    track = _prediction()
    track.onset[5, 43] = 0.9
    track.note[40:46, 43] = 0.1
    notes = notes_from_onsets(track, low=64, high=64)
    assert len(notes) == 1
    assert notes[0].start == pytest.approx(0.05)
    assert notes[0].end == pytest.approx(1)


def test_secondary_attack_peak_does_not_erase_the_initial_note():
    track = _prediction()
    track.onset[5, 43] = 0.95
    track.onset[11, 43] = 0.75
    notes = notes_from_onsets(track, low=64, high=64, min_dur=0.1)
    assert len(notes) == 1 and notes[0].start == pytest.approx(0.05)


def test_real_repick_remains_and_frame_only_notes_are_not_invented():
    track = _prediction()
    assert notes_from_onsets(track, low=64, high=64) == []
    track.onset[[5, 50], 43] = 0.95
    notes = notes_from_onsets(track, low=64, high=64)
    assert len(notes) == 2
    assert [n.start for n in notes] == pytest.approx([0.05, 0.5])


def test_quiet_real_octave_is_preserved_unless_pruning_explicitly_requested():
    track = _prediction()
    track.onset[5, 43] = track.onset[5, 55] = 0.95
    track.note[:, 55] = 0.5
    assert len(notes_from_onsets(track, low=64, high=76)) == 2
    assert len(notes_from_onsets(track, low=64, high=76, harmonic_ratio=0.85)) == 1


def test_lower_release_threshold_extends_a_note_without_accepting_weak_false_attacks():
    track = _prediction()
    track.onset[5, 43] = 0.95
    track.note[40:, 43] = 0.2
    track.onset[5, 55] = 0.95
    track.note[:, 55] = 0.2
    base = notes_from_onsets(track, low=64, high=76)
    extended = notes_from_onsets(track, low=64, high=76, release_threshold=0.15)
    assert len(base) == len(extended) == 1
    assert base[0].end == pytest.approx(0.4)
    assert extended[0].end == pytest.approx(1)
    assert extended[0].level_db == base[0].level_db


def test_cli_onset_decoder_records_settings_and_can_disable_contours(tmp_path, monkeypatch):
    import json

    from notes.cli import main
    from notes.engine.audio import write_wav

    track = _prediction()
    track.onset[5, 43] = 0.95
    monkeypatch.setattr("notes.ears.neural.basic_pitch_track", lambda *args, **kwargs: track)
    t = np.arange(44100) / 44100
    audio = 0.2 * np.sin(2 * np.pi * 330 * t)
    write_wav(tmp_path / "ref.wav", np.stack((audio, audio)), 44100)
    (tmp_path / "model.onnx").write_bytes(b"mock model")
    out = tmp_path / "result"
    assert main(["transcribe", str(tmp_path / "ref.wav"), "-o", str(out), "--method", "basic-pitch",
                 "--model-path", str(tmp_path / "model.onnx"), "--instrument", "electric",
                 "--low", "64", "--high", "64", "--note-decoder", "onsets", "--no-contours",
                 "--velocity-source", "audio"]) == 0
    data = json.loads((out / "notes.json").read_text())
    assert len(data["notes"]) == 1 and data["notes"][0]["start"] == pytest.approx(0.05)
    assert data["note_decoder"] == "onsets" and data["onset_threshold"] == 0.7
    assert not data["contours_enabled"] and data["status"] == "unverified"
    assert data["velocity_source"] == "audio" and data["velocity_normalization"] == "peak"
    assert data["notes"][0]["level_db"] != data["notes"][0]["tracker_level_db"]
