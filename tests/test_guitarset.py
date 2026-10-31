import numpy as np
import pytest

from notes import AcousticGuitar, Song, Track
from notes.ears.guitarset import guitarset_music


def annotation():
    return {"file_metadata": {"duration": 1.5, "title": "test"}, "annotations": [
        {"namespace": "note_midi", "annotation_metadata": {"data_source": "0"},
         "data": [{"time": .123456789, "duration": .5, "value": 45.2, "confidence": .1}]},
        {"namespace": "pitch_contour", "annotation_metadata": {"data_source": "0"},
         "data": {"time": [.13, .3, .5], "duration": [0, 0, 0],
                  "value": [{"voiced": True, "frequency": 111}, {"voiced": False, "frequency": 0},
                            {"voiced": True, "frequency": 110}], "confidence": [None, None, None]}}]}


def test_annotation_preserves_timing_frets_and_pitch_contours():
    music = guitarset_music(annotation(), velocity=.7)
    song = Song(Track("guitar", AcousticGuitar(fretting=True))(music), bpm=60, sample_rate=8000, loudness=None)
    score = song.compile()
    event = score.events[0]
    assert str(event.time) == "123456789/1000000000"
    assert float(event.dur) == .5 and event.pitch == 45 and event.vel == .7
    assert event.param("string") == 6
    points = event.param("pitch_points")
    assert points[0][0] == 0 and points[-1][0] == .5
    assert points[0][1] == pytest.approx(69+12*np.log2(111/440)-45)
    assert points[-1][1] == pytest.approx(0)
    assert np.isfinite(song.render().audio).all()


def test_disabled_contours_keep_fractional_tuning_and_ignore_confidence():
    event = Song(guitarset_music(annotation(), contours=False)).compile().events[0]
    assert event.vel == .8 and event.param("pitch_points")[0][1] == pytest.approx(.2)


def test_voiced_zero_frequency_is_a_reported_hole_not_a_pitch(caplog):
    data = annotation()
    data["annotations"][1]["data"]["value"][1]["voiced"] = True
    event = Song(guitarset_music(data)).compile().events[0]
    assert "1 missing voiced frequencies" in caplog.text
    assert np.isfinite(np.asarray(event.param("pitch_points"))).all()


def test_string_filter_retains_silence_and_full_timeline():
    low = guitarset_music(annotation(), string=6)
    high = guitarset_music(annotation(), string=1)
    assert low.dur == high.dur
    assert len(Song(low).compile().events) == 1
    assert not Song(high).compile().events
    with pytest.raises(ValueError):
        guitarset_music(annotation(), string=0)


def test_contour_indices_prevent_a_neighbouring_note_from_leaking_into_the_pitch():
    data = annotation()
    data["annotations"][0]["data"] = [
        {"time": .1, "duration": .5, "value": 45}, {"time": .4, "duration": .4, "value": 47}]
    data["annotations"][1]["data"] = [
        {"time": .13, "value": {"voiced": True, "index": 0, "frequency": 110}},
        {"time": .42, "value": {"voiced": True, "index": 1, "frequency": 123.470825}},
        {"time": .49, "value": {"voiced": True, "index": 0, "frequency": 110}},
        {"time": .7, "value": {"voiced": True, "index": 1, "frequency": 123.470825}}]
    events = Song(guitarset_music(data)).compile().events
    for event in events:
        assert max(abs(value) for _, value in event.param("pitch_points")) < .0001


@pytest.mark.parametrize("title", ["04_BN3-154-E_comp", "04_Jazz1-200-B_comp", "02_Funk2-119-G_comp"])
def test_known_annotation_errors_are_rejected(title):
    data = annotation()
    data["file_metadata"]["title"] = title
    with pytest.raises(ValueError, match="known GuitarSet"):
        guitarset_music(data)


def test_malformed_columns_and_impossible_strings_are_rejected():
    data = annotation()
    data["annotations"][1]["data"]["time"].pop()
    with pytest.raises(ValueError, match="different lengths"):
        guitarset_music(data)
    data = annotation()
    data["annotations"][0]["annotation_metadata"]["data_source"] = "5"
    with pytest.raises(ValueError, match="cannot be played"):
        guitarset_music(data)
