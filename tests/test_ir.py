from fractions import Fraction

import pytest

from notes import DrumKit, ElectricGuitar, Reverb, Song, Track, melody, steps
from notes.ir import Score, TempoMap


def small_song() -> Song:
    drums = Track("drums", DrumKit())
    gtr = Track("gtr", ElectricGuitar(drive=0.6), fx=[Reverb(mix=0.2)], gain=-3, pan=0.2)
    beat = drums(steps(kick="x...x...", hat="x.x.x.x."))
    riff = gtr(melody("A4 C5 E5 _", step=Fraction(1, 3)))
    return Song((beat | riff).named("a"), (beat * 2).named("b"), bpm=100, key="A minor", title="t")


def test_compile_collects_tracks_markers_paths():
    score = small_song().compile()
    assert [t.name for t in score.tracks] == ["drums", "gtr"]
    assert [(m.name, m.start, m.end) for m in score.markers] == [("a", 0, 2), ("b", 2, 6)]
    assert all(e.path in ("a", "b") for e in score.events)
    assert score.track("gtr").fx[0]["type"] == "reverb"


def test_json_round_trip_is_exact(tmp_path):
    score = small_song().compile()
    path = tmp_path / "s.ir.json"
    score.to_json(path)
    again = Score.from_json(path)
    assert again == score
    assert Score.from_json(score.to_json()) == score
    assert any(e.dur == Fraction(1, 3) for e in again.events)


def test_tempo_map_is_piecewise_constant():
    tm = TempoMap([(0, 120), (8, 60)])
    assert tm.seconds(8) == pytest.approx(4.0)
    assert tm.seconds(10) == pytest.approx(6.0)
    assert tm.bpm_at(9) == 60


def test_unbound_notes_go_to_the_default_track():
    score = Song(melody("C4 E4 G4")).compile()
    assert [t.name for t in score.tracks] == ["main"]
    assert score.tracks[0].instrument["type"] == "epiano"


def test_empty_song_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        Song().compile()


def test_summary_mentions_every_track():
    text = small_song().compile().summary()
    assert "drums" in text and "gtr" in text and "a 1-1.5" in text
