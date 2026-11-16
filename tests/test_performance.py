import struct
from fractions import Fraction

import pytest

from notes.export.midi import _vlq, write_midi
from notes.ir.score import NoteEvent, Score, TrackSpec
from notes.performance import read_midi_performance


def midi(tmp_path, rows, *, ppq=480, format_=0):
    tracks = []
    for row in rows:
        body = b''.join(_vlq(delta)+message for delta, message in row)
        body += b'\x00\xff\x2f\x00'
        tracks.append(b'MTrk'+struct.pack('>I', len(body))+body)
    path = tmp_path/'test.mid'
    path.write_bytes(b'MThd'+struct.pack('>IHHH', 6, format_, len(rows), ppq)+b''.join(tracks))
    return path


def test_export_roundtrip_with_tempo_change(tmp_path):
    score = Score((NoteEvent(Fraction(0), Fraction(2), 60, 80/127, 'p'),
                   NoteEvent(Fraction(2), Fraction(1, 2), 64, 40/127, 'p')),
                  (TrackSpec('p', {'type': 'piano'}),), Fraction(3),
                  tempo=((Fraction(0), 120.), (Fraction(1), 60.)))
    p = read_midi_performance(write_midi(score, tmp_path/'export.mid'))
    assert [(n.onset, n.key_offset, n.offset, n.pitch, n.velocity) for n in p.notes] == [
        (0., 1.5, 1.5, 60, 80), (1.5, 2., 2., 64, 40)]


def test_sustain_is_channel_local_and_keeps_raw_key_offset(tmp_path):
    p = read_midi_performance(midi(tmp_path, [[(0, b'\xb0\x40\x7f'), (0, b'\x90\x3c\x50'),
        (0, b'\x91\x3c\x40'), (480, b'\x80\x3c\x20'), (0, b'\x81\x3c\x10'),
        (480, b'\xb0\x40\x00')]]))
    a, b = p.notes
    assert (a.key_offset, a.offset, a.release_velocity) == (.5, 1., 32)
    assert (b.key_offset, b.offset, b.channel) == (.5, .5, 1)


def test_sostenuto_captures_only_existing_down_keys_and_sustain_can_take_over(tmp_path):
    p = read_midi_performance(midi(tmp_path, [[(0, b'\x90\x3c\x50'), (120, b'\xb0\x42\x7f'),
        (0, b'\x90\x40\x50'), (120, b'\x80\x3c\x00'), (0, b'\x80\x40\x00'),
        (120, b'\xb0\x40\x7f'), (120, b'\xb0\x42\x00'), (480, b'\xb0\x40\x00')]]))
    assert [(n.key_offset, n.offset) for n in p.notes] == [(.25, 1.), (.25, .25)]


def test_running_status_zero_velocity_and_fifo_overlap(tmp_path):
    p = read_midi_performance(midi(tmp_path, [[(0, b'\x90\x3c\x50'), (120, b'\x3c\x40'),
        (120, b'\x3c\x00'), (120, b'\x3c\x00')]]))
    assert [(n.onset, n.offset) for n in p.notes] == [(0., .25), (.125, .375)]


def test_repedal_does_not_resurrect_closed_note_and_eof_is_flagged(tmp_path):
    p = read_midi_performance(midi(tmp_path, [[(0, b'\x90\x3c\x50'), (240, b'\x80\x3c\x00'),
        (120, b'\xb0\x40\x7f'), (0, b'\x90\x40\x50'), (120, b'\x80\x40\x00')]]))
    assert p.notes[0].offset == .25 and not p.notes[0].damper_truncated
    assert p.notes[1].offset == p.duration and p.notes[1].damper_truncated


@pytest.mark.parametrize('rows,ppq,format_', [
    ([[(0, b'\x90\x3c\x50')]], 480, 0),
    ([[(0, b'\x80\x3c\x00')]], 480, 0),
    ([[(0, b'\x3c\x50')]], 480, 0),
    ([[(0, b'\xff\x51\x03\x00\x00\x00')]], 480, 0),
    ([[(0, b'\xb0\x7b\x00')]], 480, 0),
    ([[(0, b'\xe0\x01\x40')]], 480, 0),
    ([[]], 0xE728, 0), ([[]], 480, 2), ([[], []], 480, 0),
])
def test_unsupported_or_unmatched_events_fail_explicitly(tmp_path, rows, ppq, format_):
    with pytest.raises(ValueError):
        read_midi_performance(midi(tmp_path, rows, ppq=ppq, format_=format_))


def test_skips_meta_and_sysex_preserves_soft_pedal_and_binary_threshold(tmp_path):
    p = read_midi_performance(midi(tmp_path, [[(0, b'\xff\x01\x03abc'), (0, b'\xf0\x02\x01\xf7'),
        (0, b'\xb0\x43\x22'), (0, b'\xb0\x40\x3f'), (0, b'\x90\x3c\x50'),
        (240, b'\x80\x3c\x00'), (240, b'\xb0\x40\x40'), (0, b'\x90\x40\x50'),
        (240, b'\x80\x40\x00'), (240, b'\xb0\x40\x00')]]))
    assert [(n.key_offset, n.offset) for n in p.notes] == [(.25, .25), (.75, 1.)]
    assert (p.controls[0].number, p.controls[0].value) == (67, 34)


@pytest.mark.parametrize('content', [b'', b'MThd', b'MThd\x00\x00\x00\x06',
    b'MThd'+struct.pack('>IHHH', 6, 0, 1, 480)+b'MTrk\x00\x00\x00\x05\x00',
    b'MThd'+struct.pack('>IHHH', 6, 0, 1, 480)+b'MTrk\x00\x00\x00\x02\x00\x90'])
def test_malformed_files_are_rejected(tmp_path, content):
    path = tmp_path/'bad.mid'
    path.write_bytes(content)
    with pytest.raises(ValueError):
        read_midi_performance(path)


def test_conductor_track_tempo_and_channel_track_events_are_merged(tmp_path):
    path = midi(tmp_path, [[(0, b'\xff\x51\x03\x07\xa1\x20'),
        (480, b'\xff\x51\x03\x0f\x42\x40')],
        [(240, b'\x90\x3c\x50'), (720, b'\x80\x3c\x00')]], format_=1)
    p = read_midi_performance(path)
    assert p.notes[0].onset == .25 and p.notes[0].offset == 1.5


def test_running_status_is_not_inherited_across_tracks_or_sysex(tmp_path):
    for rows, fmt in [([[(0, b'\x90\x3c\x50')], [(0, b'\x3c\x00')]], 1),
                      ([[(0, b'\x90\x3c\x50'), (0, b'\xf0\x01\xf7'), (480, b'\x3c\x00')]], 0)]:
        with pytest.raises(ValueError, match='Running status'):
            read_midi_performance(midi(tmp_path, rows, format_=fmt))
