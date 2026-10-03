from fractions import Fraction

import pytest

from notes import E, Key, S, chords, euclid, hit, melody, steps
from notes.lang.notation import parse_hits
from notes.lang.perform import perform


def events(m):
    return perform(m).events


def test_melody_holds_rests_and_lengths():
    evs = events(melody("C4 _ . E4@2", step=E))
    assert [(e.pitch, e.time, e.dur) for e in evs] == [(60, 0, 1), (64, Fraction(3, 2), 1)]


def test_melody_chords_accents_repeats():
    evs = events(melody("[C4,E4,G4]@4 A4> B4!2", step=E))
    assert [e.pitch for e in evs] == [60, 64, 67, 69, 71, 71]
    assert evs[3].vel == pytest.approx(1.0)
    assert melody("A B", octave=3).dur == 1 and events(melody("A", octave=3))[0].pitch == 57


def test_melody_rejects_leading_hold():
    with pytest.raises(ValueError, match="must follow"):
        melody("_ C4")


def test_steps_grid():
    m = steps(kick="x...x...", hat="x.x.x.x.")
    assert m.dur == 2
    kicks = [e.time for e in events(m) if e.params["sound"] == "kick"]
    hats = [e.time for e in events(m) if e.params["sound"] == "hat"]
    assert kicks == [0, 1] and hats == [0, Fraction(1, 2), 1, Fraction(3, 2)]


def test_steps_velocities_holds_and_aliases():
    hits, cells = parse_hits("X.o_9 |x")
    assert cells == 6
    assert hits == [(0, 1, 1.0), (2, 2, 0.5), (4, 1, 1.0), (5, 1, 0.8)]
    assert events(steps(bd="x"))[0].params["sound"] == "kick"
    assert events(steps({"C2": "x"}))[0].pitch == 36 and "sound" not in events(steps({"C2": "x"}))[0].params


def test_steps_polyrhythm_via_length():
    m = steps(kick="x..", hat="x...", length=3, step=S)
    assert m.dur == 3
    assert len([e for e in events(m) if e.params["sound"] == "kick"]) == 4
    assert len([e for e in events(m) if e.params["sound"] == "hat"]) == 3


def test_unknown_lane_lists_the_drums():
    with pytest.raises(ValueError, match="kick"):
        steps(bongo="x...")


def test_euclid():
    assert euclid(3, 8) == "x..x..x."
    assert euclid(4, 16) == "x...x...x...x..."
    assert euclid(0, 4) == "...." and euclid(4, 4) == "xxxx"
    assert euclid(3, 8, rotate=3) == "x..x.x.."
    assert all(euclid(k, 16).count("x") == k for k in range(17))


def test_chords_and_roman_numerals():
    prog = chords("Am F C G")
    assert prog.dur == 16
    first = sorted(e.pitch for e in events(prog) if e.time == 0)
    assert first == [57, 60, 64]
    roman = Key.parse("A minor").chords("i VI III VII", voicing="smooth")
    assert roman.dur == 16
    held = chords("Am _ F@2")
    assert held.dur == 16


def test_key_melody_uses_degrees():
    m = Key.parse("C major").melody("1 3 5 8 b3", step=E)
    assert [e.pitch for e in events(m)] == [60, 64, 67, 72, 63]


def test_hit():
    assert events(hit("crash", 4))[0].params["sound"] == "crash"


def test_legato_and_slide_prefixes():
    evs = perform(melody("/B5~@4 A5 &G5 &/E5", step=E)).events
    assert [e.pitch for e in evs] == [83, 81, 79, 76]
    assert [e.params for e in evs] == [{"slide": -2.0, "vibrato": 0.25}, {}, {"legato": True},
                                       {"legato": True, "slide": -2.0}]
