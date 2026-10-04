import numpy as np
import pytest

from notes import BassGuitar, E, RockKit, Song, Track, hit, melody, stack
from notes.ears.compare import _onset_strength, _onsets
from notes.ears.transcribe import best_chord, chroma, drum_hits, drum_profile, notes_from_track, pitch_track

SR = 44100


def _render(music, bpm=120):
    return Song(music, bpm=bpm, loudness=None).render().audio.mean(axis=0)


def test_pitch_track_follows_a_bass_line():
    line = melody("A1@2 D2@2 E2@2 G1@2", step=E)
    x = _render(Track("bass", BassGuitar())(line))
    track = pitch_track(x, SR, 35.0, 400.0)
    truth = {0.05: 33, 0.55: 38, 1.05: 40, 1.55: 31}  # A1, D2, E2, G1 a quarter note apart
    for t, midi in truth.items():
        sel = (track.t > t) & (track.t < t + 0.35)
        got = 69 + 12 * np.log2(np.nanmedian(track.f0[sel]) / 440.0)
        assert got == pytest.approx(midi, abs=0.15)


def test_notes_come_back_with_their_pitches_and_starts():
    line = melody("A1@2 D2@2 E2@2 G1@2", step=E)
    x = _render(Track("bass", BassGuitar())(line))
    onsets = _onsets(*_onset_strength(x))
    notes = notes_from_track(pitch_track(x, SR, 35.0, 400.0), onsets)
    assert [round(n.pitch) for n in notes] == [33, 38, 40, 31]
    assert [n.start for n in notes] == pytest.approx([0.0, 0.5, 1.0, 1.5], abs=0.05)


def test_drum_hits_are_named_after_the_kit_sounds():
    kit = RockKit()
    templates = {}
    for sound in ("kick", "snare", "hat"):
        y = _render(Track("d", kit)(hit(sound, 1)))
        templates[sound] = drum_profile(y, SR, 0.0)
    pattern = stack(hit("kick", 1), hit("hat", 1).shift(0.5), hit("snare", 1).shift(1),
                    hit("hat", 1).shift(1.5), hit("kick", 1).shift(2))
    x = _render(Track("d", kit)(pattern))
    hits = drum_hits(x, SR, templates, [0.0, 0.25, 0.5, 0.75, 1.0])
    named = {round(t, 2): {n for tt, n, _ in hits if tt == t} for t, _, _ in hits}
    assert "kick" in named[0.0] and "hat" in named[0.25] and "snare" in named[0.5]
    assert "hat" in named[0.75] and "kick" in named[1.0]
    assert "snare" not in named[0.0] and "kick" not in named[0.25]


def test_a_minor_chord_is_found_on_its_tuning():
    t = np.arange(SR) / SR
    a4 = 455.0
    x = sum(np.sin(2 * np.pi * a4 * 2 ** ((m - 69) / 12) * t) / (1 + i) for i, m in enumerate((57, 60, 64, 69)))
    _, c = chroma(x, SR, a4)
    name, score = best_chord(c.mean(axis=1))
    assert name == "Am" and score > 0.8
