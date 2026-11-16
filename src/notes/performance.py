"""Known MIDI performances for reference evaluation, without audio transcription.

SMF 0/1 with PPQN timing; tempo changes, running status and channel-local binary
sustain/sostenuto are supported. Key release and effective damper time stay
separate. This is not half-pedaling, sympathetic resonance or acoustic decay.
"""
from __future__ import annotations

import struct
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

__all__ = ['MidiControl', 'MidiNote', 'MidiPerformance', 'read_midi_performance']


@dataclass(frozen=True)
class MidiNote:
    onset: float
    key_offset: float
    offset: float
    pitch: int
    velocity: int
    release_velocity: int
    channel: int
    damper_truncated: bool


@dataclass(frozen=True)
class MidiControl:
    time: float
    channel: int
    number: int
    value: int


@dataclass(frozen=True)
class MidiPerformance:
    notes: tuple[MidiNote, ...]
    controls: tuple[MidiControl, ...]
    duration: float
    ppq: int
    format: int


def _vlq(data: bytes, at: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if at >= len(data):
            raise ValueError('Truncated MIDI variable-length value')
        byte = data[at]
        at += 1
        value = (value << 7) | (byte & 127)
        if byte < 128:
            return value, at
    raise ValueError('MIDI variable-length value exceeds four bytes')


def _track(data: bytes) -> list[tuple[int, str, tuple[int, ...]]]:
    events = []
    at, tick, running = 0, 0, None
    while at < len(data):
        delta, at = _vlq(data, at)
        tick += delta
        if at == len(data):
            raise ValueError('Missing MIDI event after delta')
        status = data[at]
        if status >= 128:
            at += 1
            running = status if status < 240 else None
        elif running is None:
            raise ValueError('Running status without channel status')
        else:
            status = running
        if status == 255:
            if at == len(data):
                raise ValueError('Missing MIDI meta type')
            kind = data[at]
            size, at = _vlq(data, at+1)
            payload = data[at:at+size]
            if len(payload) != size:
                raise ValueError('Truncated MIDI meta event')
            at += size
            if kind == 81:
                if size != 3 or not int.from_bytes(payload, 'big'):
                    raise ValueError('Invalid MIDI tempo')
                events.append((tick, 'tempo', (int.from_bytes(payload, 'big'),)))
            elif kind == 47:
                if size or at != len(data):
                    raise ValueError('Invalid end-of-track event')
                events.append((tick, 'end', ()))
                return events
        elif status in (240, 247):
            size, at = _vlq(data, at)
            at += size
            if at > len(data):
                raise ValueError('Truncated MIDI system-exclusive event')
        elif 128 <= status < 240:
            kind, channel = status >> 4, status & 15
            size = 1 if kind in (12, 13) else 2
            payload = data[at:at+size]
            if len(payload) != size or any(byte >= 128 for byte in payload):
                raise ValueError('Invalid MIDI channel payload')
            at += size
            if kind in (8, 9):
                name = 'on' if kind == 9 and payload[1] else 'off'
                events.append((tick, name, (channel, *payload)))
            elif kind == 11:
                if payload[0] in (120, 121, 123):
                    raise ValueError('Channel-mode note termination/reset is not supported')
                events.append((tick, 'control', (channel, *payload)))
            elif kind == 14 and payload != b'\x00\x40':
                raise ValueError('Pitch bends require a pitch trajectory, not fixed MIDI notes')
        else:
            raise ValueError('Unsupported MIDI system status')
    raise ValueError('Missing MIDI end-of-track event')


def read_midi_performance(path: str | Path) -> MidiPerformance:
    """Read an annotated performance with key gates and binary pedal-held gates.

    Repeated overlapping keys pair FIFO within their channel. CC64 holds released
    keys. CC66 captures only keys down at its rising edge. Controls are retained
    verbatim, including CC67, which has no modeled effect here. Pedals use the
    MIDI threshold 64. A held note at file end is explicitly marked truncated;
    an unmatched key on/off is rejected. Cross-track equal-tick ties follow file
    track order. Later re-pedaling cannot resurrect a previously damped note.
    """
    data = Path(path).read_bytes()
    if len(data) < 14 or data[:4] != b'MThd':
        raise ValueError('Not a Standard MIDI File')
    header = int.from_bytes(data[4:8], 'big')
    if header < 6 or 8+header > len(data):
        raise ValueError('Truncated MIDI header')
    format_, count, ppq = struct.unpack('>HHH', data[8:14])
    if format_ not in (0, 1) or not count or (format_ == 0 and count != 1):
        raise ValueError('Only MIDI formats 0 and 1 are supported')
    if not ppq or ppq & 0x8000:
        raise ValueError('Only positive PPQN time division is supported')
    at, merged = 8+header, []
    for track in range(count):
        if data[at:at+4] != b'MTrk' or at+8 > len(data):
            raise ValueError('Missing MIDI track chunk')
        length = int.from_bytes(data[at+4:at+8], 'big')
        body = data[at+8:at+8+length]
        if len(body) != length:
            raise ValueError('Truncated MIDI track')
        for order, event in enumerate(_track(body)):
            merged.append((event[0], track, order, event[1], event[2]))
        at += 8+length
    if at != len(data):
        raise ValueError('Unexpected data after MIDI tracks')
    active: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    down: set[int] = set()
    ringing: set[int] = set()
    captured: list[set[int]] = [set() for _ in range(16)]
    sustain, sostenuto = [False]*16, [False]*16
    records: list[dict] = []
    controls = []
    now, last_tick, tempo = 0., 0, 500000

    def close_dampers(channel: int) -> None:
        for index in tuple(ringing):
            note = records[index]
            if (note['channel'] == channel and index not in down and not sustain[channel]
                    and index not in captured[channel]):
                note['offset'] = now
                ringing.remove(index)

    for tick, _, _, kind, payload in sorted(merged):
        now += (tick-last_tick)*tempo/(ppq*1_000_000)
        last_tick = tick
        if kind == 'tempo':
            tempo = payload[0]
        elif kind == 'on':
            channel, pitch, velocity = payload
            index = len(records)
            records.append({'onset': now, 'pitch': pitch, 'velocity': velocity, 'channel': channel,
                            'damper_truncated': False})
            active[channel, pitch].append(index)
            down.add(index)
            ringing.add(index)
        elif kind == 'off':
            channel, pitch, velocity = payload
            if not active[channel, pitch]:
                raise ValueError('MIDI key-off has no matching key-on')
            index = active[channel, pitch].popleft()
            records[index].update(key_offset=now, release_velocity=velocity)
            down.remove(index)
            close_dampers(channel)
        elif kind == 'control':
            channel, number, value = payload
            controls.append(MidiControl(now, channel, number, value))
            if number == 64:
                sustain[channel] = value >= 64
            elif number == 66:
                pressed = value >= 64
                if pressed and not sostenuto[channel]:
                    captured[channel] = {i for i in down if records[i]['channel'] == channel}
                elif not pressed:
                    captured[channel].clear()
                sostenuto[channel] = pressed
            close_dampers(channel)
    if down:
        raise ValueError('MIDI key-on has no matching key-off')
    for index in ringing:
        records[index].update(offset=now, damper_truncated=True)
    notes = tuple(MidiNote(**record) for record in records)
    return MidiPerformance(notes, tuple(controls), now, ppq, format_)
