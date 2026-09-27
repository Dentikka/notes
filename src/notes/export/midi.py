"""Standard MIDI File (type 1) export of a `Score`, for DAWs and notation software.

Track 0 carries tempo, meter and section markers; each IR track becomes a MIDI track with
its own channel (drums on channel 10) and a General MIDI program. Pitches are rounded to
the nearest semitone; exact rational times become ticks at 480 per quarter note.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from pathlib import Path

from notes.ir.score import Score, TrackSpec

__all__ = ["write_midi"]

_DRUM_CHANNEL = 9


def _vlq(value: int) -> bytes:
    """MIDI variable-length quantity."""
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _meta(kind: int, data: bytes) -> bytes:
    return bytes([0xFF, kind]) + _vlq(len(data)) + data


def _chunk(messages: Iterable[tuple[int, int, bytes]]) -> bytes:
    """Messages are (tick, order, bytes); order sorts note-offs before note-ons at one tick."""
    body, now = bytearray(), 0
    for tick, _, msg in sorted(messages, key=lambda m: (m[0], m[1])):
        body += _vlq(tick - now) + msg
        now = tick
    body += _vlq(0) + _meta(0x2F, b"")
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def _program(spec: TrackSpec) -> int:
    inst = spec.instrument
    if inst.get("program") is not None:
        return int(inst["program"])
    if inst.get("type") == "electric_guitar":
        drive = float(inst.get("drive", 0.5))
        return 30 if drive > 0.7 else 29 if drive > 0.35 else 27
    return 0


def write_midi(score: Score, path: str | Path, ppq: int = 480) -> Path:
    conductor: list[tuple[int, int, bytes]] = []
    if score.title:
        conductor.append((0, 0, _meta(0x03, score.title.encode("utf-8"))))
    num, den = score.meter
    conductor.append((0, 0, _meta(0x58, bytes([num, max(den.bit_length() - 1, 0), 24, 8]))))
    for beat, bpm in score.tempo:
        conductor.append((round(beat * ppq), 0, _meta(0x51, round(60_000_000 / bpm).to_bytes(3, "big"))))
    for m in score.markers:
        conductor.append((round(m.start * ppq), 1, _meta(0x06, m.name.encode("utf-8"))))
    chunks = [_chunk(conductor)]
    channels = iter(c for c in range(16) if c != _DRUM_CHANNEL)
    for spec in score.tracks:
        events = score.events_of(spec.name)
        if not events:
            continue
        drums = spec.instrument.get("type") == "drums"
        ch = _DRUM_CHANNEL if drums else next(channels, 0)
        msgs = [(0, 0, _meta(0x03, spec.name.encode("utf-8")))]
        if not drums:
            msgs.append((0, 1, bytes([0xC0 | ch, _program(spec) & 0x7F])))
        for e in events:
            key = min(127, max(0, round(e.pitch)))
            vel = min(127, max(1, round(e.vel * 127)))
            msgs.append((round(e.time * ppq), 3, bytes([0x90 | ch, key, vel])))
            msgs.append((round(e.end * ppq), 2, bytes([0x80 | ch, key, 0])))
        chunks.append(_chunk(msgs))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), ppq) + b"".join(chunks))
    return out
