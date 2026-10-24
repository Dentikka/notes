import numpy as np
import pytest

from notes.ears.polyphonic import notes_from_templates, template_track
from notes.ears.transcribe import PitchTrack, notes_from_track


def test_monophonic_sustain_bridges_dropouts_but_stops_at_silence_and_restarts_at_onsets():
    t = np.arange(100) * 0.01
    f0 = np.full(100, np.nan)
    f0[3:20] = 110
    f0[53:70] = 110
    level = np.full(100, -20.0)
    level[40:50] = level[90:] = -100
    track = PitchTrack(t, f0, np.ones(100), level)
    legacy = notes_from_track(track, [0.0, 0.5])
    assert legacy[0].end < 0.3
    notes = notes_from_track(track, [0.0, 0.5], sustain_unvoiced=True)
    assert len(notes) == 2
    assert [n.start for n in notes] == pytest.approx([0, 0.5])
    assert [n.end for n in notes] == pytest.approx([0.4, 0.9])
    assert [n.pitch for n in notes] == pytest.approx([45, 45])
    assert len(notes[0].contour) == 17


def test_unvoiced_onset_does_not_inherit_previous_notes_pitch():
    t = np.arange(100) * 0.01
    f0 = np.full(100, np.nan)
    f0[:20] = 110
    track = PitchTrack(t, f0, np.ones(100), np.full(100, -20.0))
    notes = notes_from_track(track, [0.0, 0.5], sustain_unvoiced=True)
    assert len(notes) == 1 and notes[0].end == pytest.approx(0.5)


def _tone(pitch, duration=0.4, sr=16000):
    t = np.arange(round(duration * sr)) / sr
    f = 440 * 2 ** ((pitch - 69) / 12)
    envelope = np.minimum(t / 0.01, 1) * np.exp(-t * 4)
    envelope *= np.minimum((duration - t) / 0.01, 1)
    return envelope * sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 7))


def test_joint_templates_recover_overlapping_notes_without_labelling_overtones():
    sr = 16000
    templates = {p: _tone(p) for p in (57, 60, 64, 69, 72, 76)}
    x = np.zeros(sr * 2)
    truth = [(0.1, 57), (0.3, 60), (0.5, 64)]
    for onset, p in truth:
        start = round(onset * sr)
        x[start:start + len(templates[p])] += templates[p]
    track = template_track(x, sr, templates, n_fft=2048, hop=128)
    notes = notes_from_templates(track, relative_db=15, floor_db=25, min_dur=0.15)
    assert [n.pitch for n in notes] == [57, 60, 64]
    assert [n.start for n in notes] == pytest.approx([n[0] for n in truth], abs=0.06)
    assert [n.end for n in notes] == pytest.approx([n[0] + 0.4 for n in truth], abs=0.07)
    assert np.median(track.residual[track.activation.sum(axis=0) > 0.5]) < 0.4


def test_repeated_template_note_can_split_at_its_own_attack():
    sr = 16000
    sample = _tone(57, duration=0.7)
    x = np.zeros(sr * 2)
    x[:len(sample)] += sample
    x[sr // 2:sr // 2 + len(sample)] += sample
    track = template_track(x, sr, {57: sample, 69: _tone(69)}, n_fft=2048, hop=128)
    notes = notes_from_templates(track, [0, 0.5], min_dur=0.2)
    assert [n.pitch for n in notes] == [57, 57]
    assert notes[1].start == pytest.approx(0.5, abs=0.03)


def test_template_silence_returns_no_notes_and_silent_dictionary_is_rejected():
    track = template_track(np.zeros(16000), 16000, {57: _tone(57)}, n_fft=2048)
    assert notes_from_templates(track) == []
    with pytest.raises(ValueError, match="silent template"):
        template_track(np.ones(16000), 16000, {57: np.zeros(16000)})


def test_measured_contour_roundtrips_through_ir_and_changes_rendered_pitch():
    from notes import Note, Song, SynthGuitar, Track
    from notes.ears.transcribe import TrackedNote, tracked_music
    from notes.engine.articulation import pitch_curve
    from notes.ir import Score

    note = TrackedNote(0.1, 0.6, 57, -20, np.array([0.1, 0.3, 0.6]), np.array([55, 57, 57]))
    music = tracked_music([note], duration=0.8, contours=True)
    score = Song(Track("guitar", SynthGuitar())(music), bpm=60, loudness=None).compile()
    loaded = Score.from_json(score.to_json())
    params = dict(loaded.events[0].params)
    curve = pitch_curve(params, 8000, 16000)
    assert curve[0] == pytest.approx(-2) and curve[-1] == pytest.approx(0)
    assert loaded.events[0].pitch == 57
    assert np.isfinite(Song(Track("guitar", SynthGuitar())(music), bpm=60).render().audio).all()
    assert pitch_curve(dict(Note(57, 1).params), 100, 16000) is None


def test_transcribe_cli_writes_notes_midi_reconstruction_and_comparison(tmp_path):
    import json

    from notes import BassGuitar, Song, Track, melody
    from notes.cli import main

    reference = tmp_path / "reference.wav"
    Song(Track("bass", BassGuitar())(melody("A1 D2", step=0.5)), bpm=60,
         loudness=None).render(reference)
    out = tmp_path / "transcription"
    assert main(["transcribe", str(reference), "-o", str(out), "--low", "28", "--high", "55"]) == 0
    data = json.loads((out / "notes.json").read_text(encoding="utf-8"))
    assert [round(n["pitch"]) for n in data["notes"]] == [33, 38]
    for name in ("transcribed.ir.json", "transcribed.mid", "transcribed.wav", "transcribed.png",
                 "transcribed.spec.png", "comparison.json", "comparison.png", "transcribed.ab.wav"):
        assert (out / name).exists()


@pytest.mark.parametrize("points", [[], [(0.1, 0), (0, 1)], [(0, float("nan"))]])
def test_invalid_measured_pitch_curve_is_rejected(points):
    from notes.engine.articulation import pitch_curve

    with pytest.raises(ValueError, match="pitch_points"):
        pitch_curve({"pitch_points": points}, 100, 16000)
