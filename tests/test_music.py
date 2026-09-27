from fractions import Fraction

import pytest

from notes import DrumKit, E, H, Note, Q, Rest, S, Track, W, cat, chord, melody, stack, steps
from notes.lang.music import Par, Repeat, Seq
from notes.lang.perform import perform


def pitches(m):
    return [e.pitch for e in perform(m).events]


def times(m):
    return [e.time for e in perform(m).events]


def test_algebra_durations():
    a, b = Note("C4", Q), Note("D4", H)
    assert (a + b).dur == 3
    assert (a | b).dur == 2
    assert (a * 3).dur == 3
    assert (3 * a).dur == 3
    assert Rest(0) + a == cat(Rest(0), a)


def test_cat_and_stack_flatten():
    a, b = Note("C4", Q), Note("E4", Q)
    assert isinstance(cat(a, cat(b, a)), Seq) and len(cat(a, cat(b, a)).items) == 3
    assert isinstance(stack(a, stack(b, a)), Par) and len(stack(a, stack(b, a)).items) == 3


def test_exact_time_with_triplets():
    trip = Note("C4", Fraction(1, 3)) * 3
    assert trip.dur == 1
    assert times(trip + Note("D4", Q)) == [0, Fraction(1, 3), Fraction(2, 3), 1]


def test_repeat_keeps_structure_and_every_varies():
    phrase = Note("C4", Q)
    looped = (phrase * 4).every(2, lambda m: m.transpose(12))
    assert isinstance(looped, Repeat)
    assert pitches(looped) == [60, 72, 60, 72]
    replaced = (phrase * 4).every(4, Note("G4", Q))
    assert pitches(replaced) == [60, 60, 60, 67]


def test_mute_bar_silences_only_that_bar():
    beat = melody("C4 C4 C4 C4", step=Q) * 4
    muted = beat.mute(bars=2)
    assert muted.dur == 16
    ts = times(muted)
    assert len(ts) == 12 and not any(4 <= t < 8 for t in ts)
    assert len(perform(beat.mute(window=(0, 2))).events) == 14


def test_reverse_is_an_involution():
    m = melody("C4 D4@2 . E4 [G4,B4]", step=E)
    assert perform(m.reverse().reverse()).events == perform(m).events
    assert pitches(melody("C4 D4 E4", step=Q).reverse()) == [64, 62, 60]


def test_stretch_transpose_invert():
    m = melody("C4 E4", step=Q)
    assert m.stretch(2).dur == 4 and times(m.stretch(2)) == [0, 2]
    assert pitches(m.transpose(2)) == [62, 66]
    assert pitches(m.invert("E4")) == [68, 64]


def test_pitch_operations_leave_drums_alone():
    beat = steps(kick="x...")
    assert pitches(beat.transpose(5)) == [36]


def test_swing_delays_offbeats():
    swung = melody("C4 C4 C4 C4", step=E).swing(Fraction(2, 3), grid=E)
    assert times(swung) == [0, Fraction(2, 3), 1, Fraction(5, 3)]


def test_slice_and_loop():
    m = melody("C4 D4 E4 F4", step=Q)
    assert pitches(m.slice(1, 3)) == [62, 64]
    looped = m.loop(6)
    assert looped.dur == 6 and pitches(looped) == [60, 62, 64, 65, 60, 62]


def test_named_sections_become_markers():
    a, b = Note("C4", Q).named("intro"), Note("D4", H).named("verse")
    perf = perform(a + b * 2)
    assert perf.markers == [("intro", 0, 1), ("verse", 1, 3), ("verse", 3, 5)]
    assert perf.events[0].path == ("intro",)


def test_track_binding_inner_wins_and_conflicts_raise():
    one, two = Track("one", DrumKit()), Track("two", DrumKit())
    m = one(two(Note("C4", Q)) + Note("D4", Q))
    assert [e.track for e in perform(m).events] == ["two", "one"]
    impostor = Track("one", DrumKit(tune=3))
    with pytest.raises(ValueError, match="Two different tracks"):
        perform(one(Note("C4", Q)) + impostor(Note("D4", Q)))


def test_rhythm_restrikes_held_harmony():
    riff = chord("C", W).rhythm("x.x.", step=Q)
    perf = perform(riff)
    assert riff.dur == 4
    assert sorted({e.time for e in perf.events}) == [0, 2]
    assert len(perf.events) == 6


def test_arp_walks_the_chord():
    arp = chord("C", H, octave=4).arp("up", step=E)
    assert pitches(arp) == [60, 64, 67, 60]
    assert pitches(chord("C", H, octave=4).arp("updown", step=S)) == [60, 64, 67, 64, 60, 64, 67, 64]


def test_humanize_is_seeded():
    m = melody("C4 D4 E4 F4", step=E)
    first, second = perform(m.humanize(seed=1)).events, perform(m.humanize(seed=1)).events
    assert first == second
    assert first != perform(m.humanize(seed=2)).events


def test_arp_follows_a_picking_pattern():
    from fractions import Fraction

    picked = chord("C", H, octave=4).arp([0, 2, 3, 4, 3, 2], step=Fraction(1, 3), octaves=2)
    assert pitches(picked) == [60, 67, 72, 76, 72, 67]
    with pytest.raises(ValueError, match="tone indices"):
        chord("C", H).arp("sideways")
