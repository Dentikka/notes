"""IDMT-SMT-Guitar XML notes -> Music at BPM 60; no recorded audio is loaded."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from notes.lang.music import Music, Note, place

_TUNING = (40, 45, 50, 55, 59, 64)


def _required(row: ET.Element, name: str) -> str:
    text = row.findtext(name)
    if text is None or not text.strip():
        raise ValueError(f'IDMT event requires {name}')
    return text.strip()


def idmt_music(source: str | Path, *, velocity: float = .7) -> Music:
    """Import normal picked/muted notes with exact annotated seconds and string geometry.

    Bind the result to a guitar Track and use BPM 60. Source string 1 (low E)
    becomes DSL string 6. ``MU`` selects palm_mute; ``PK`` is a regular pick.
    Velocity is explicit, not inferred from annotations. Offsets are annotated
    acoustic ends, not measured physical finger-release times. They are used as
    approximate gates; this importer does not correct audio/annotation latency.

    Bends, slides, vibrato, harmonics, dead notes and finger-style require source
    controls not provided here and raise instead of becoming ordinary fixed notes.
    Only standard six-string tuning and frets 0..24 are supported.
    """
    if not math.isfinite(velocity) or not 0 <= velocity <= 1:
        raise ValueError('velocity must be finite and between 0 and 1')
    with Path(source).open('rb') as stream:
        data = stream.read(2_000_001)
    if len(data) > 2_000_000 or b'<!doctype' in data.lower() or b'<!entity' in data.lower():
        raise ValueError('IDMT XML must be bounded and contain no document type or entities')
    root = ET.fromstring(data)
    if root.tag != 'instrumentRecording':
        raise ValueError('Expected IDMT instrumentRecording XML')
    tuning = root.findtext('globalParameter/instrumentTuning')
    if tuning is not None and tuple(int(p) for p in tuning.split()) != _TUNING:
        raise ValueError('Only standard six-string IDMT tuning is supported')
    events, seen = [], set()
    end = Fraction(0)
    for index, row in enumerate(root.findall('transcription/event')):
        start, offset = Fraction(_required(row, 'onsetSec')), Fraction(_required(row, 'offsetSec'))
        pitch = int(_required(row, 'pitch'))
        string, fret = int(_required(row, 'stringNumber')), int(_required(row, 'fretNumber'))
        if start < 0 or offset <= start or not 0 <= pitch <= 127 or not 1 <= string <= 6 or not 0 <= fret <= 24:
            raise ValueError(f'Invalid IDMT time, pitch or fret/string at event {index}')
        if _TUNING[string-1]+fret != pitch:
            raise ValueError(f'IDMT pitch disagrees with its string and fret at event {index}')
        excitation, expression = _required(row, 'excitationStyle'), _required(row, 'expressionStyle')
        if excitation not in ('PK', 'MU') or expression != 'NO':
            raise ValueError(f'Unsupported IDMT articulation {excitation}/{expression} at event {index}')
        key = start, string, pitch
        if key in seen:
            raise ValueError(f'Duplicate IDMT note at event {index}')
        seen.add(key)
        note = Note(pitch, offset-start, velocity).param(string=7-string, palm_mute=excitation == 'MU')
        events.append((start, note))
        end = max(end, offset)
    if not events:
        raise ValueError('IDMT annotation contains no supported note events')
    return place(events, length=end)
