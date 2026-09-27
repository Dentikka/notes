import pytest

from notes.lang.theory import Key, chord_pitches, midi_to_hz, parse_chord, parse_pitch, pitch_name, voice_lead


@pytest.mark.parametrize(
    ("name", "midi"),
    [("C4", 60), ("A4", 69), ("Bb2", 46), ("F#3", 54), ("c#-1", 1), ("B#3", 60), ("bb3", 58), ("E", 64)],
)
def test_parse_pitch(name, midi):
    assert parse_pitch(name) == midi


def test_parse_pitch_rejects_garbage():
    with pytest.raises(ValueError, match="Not a pitch"):
        parse_pitch("H2")


def test_pitch_name_and_microtones():
    assert pitch_name(61) == "C#4"
    assert pitch_name(60.5) == "C4+50c"


def test_equal_temperament():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert midi_to_hz(81) == pytest.approx(880.0)
    assert midi_to_hz(69, a4=432.0) == pytest.approx(432.0)


def test_chord_symbols():
    assert parse_chord("Am7") == (9, (0, 3, 7, 10), None)
    assert chord_pitches("C", 4) == [60, 64, 67]
    assert chord_pitches("E5", 2) == [40, 47, 52]
    assert chord_pitches("G7/B", 3) == [47, 55, 59, 62, 65]
    with pytest.raises(ValueError, match="Unknown chord quality"):
        parse_chord("Cxyz")


def test_key_degrees_cross_octaves():
    am = Key.parse("A minor")
    assert [am.degree(d) for d in (1, 3, 5, 8)] == [69, 72, 76, 81]
    assert am.degree(0) == 67  # the step below the tonic: G4
    assert am.degree(3, alter=1) == 73
    assert str(Key.parse("F# dorian")) == "F# dorian"


def test_roman_numerals_follow_case():
    am = Key.parse("A minor")
    assert am.roman("i") == [57, 60, 64]
    assert am.roman("V") == [64, 68, 71]  # major V: G#
    assert am.roman("VII7") == [67, 71, 74, 77]
    assert Key.parse("C").roman("bVII") == [58, 62, 65]


def test_voice_leading_moves_little():
    prog = [chord_pitches(s, 3) for s in ("Am", "F", "C", "G")]
    voiced = voice_lead(prog)
    for a, b in zip(voiced, voiced[1:], strict=False):
        assert sum(abs(x - y) for x, y in zip(sorted(a), sorted(b), strict=True)) <= 5
