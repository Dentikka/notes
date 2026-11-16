from fractions import Fraction

import numpy as np
import pytest

from notes import Piano, Violin
from notes.engine import render_score
from notes.ir import NoteEvent, Score, TrackSpec


def score(instrument, pitch):
    return Score((NoteEvent(Fraction(0), Fraction(1, 5), pitch, .7, 'p'),),
                 (TrackSpec('p', instrument.to_spec()),), Fraction(1), sample_rate=8000, seed=18,
                 master={'loudness': None, 'ceiling': -1.})


@pytest.mark.parametrize('instrument', [Piano(), Violin(vibrato=0)])
@pytest.mark.parametrize('pitch', [69, 69.])
def test_stochastic_source_survives_json_roundtrip(instrument, pitch):
    original = score(instrument, pitch)
    restored = Score.from_json(original.to_json())
    assert type(restored.events[0].pitch) is type(pitch)
    assert np.array_equal(render_score(original).audio, render_score(restored).audio)


def test_unmarked_old_json_keeps_legacy_float_seed_interpretation():
    original = score(Piano(), 69.)
    data = original.to_dict()
    assert data['events'][0]['pitch'] == 69
    assert 'seed_pitch_int' not in data['events'][0]
    restored = Score.from_dict(data)
    assert type(restored.events[0].pitch) is float
    assert np.array_equal(render_score(original).audio, render_score(restored).audio)


@pytest.mark.parametrize('pitch,marker', [(69.2, True), (69, 1), (69, 'true')])
def test_invalid_seed_marker_rejected(pitch, marker):
    data = {'t': 0, 'dur': 1, 'pitch': pitch, 'vel': .7, 'track': 'p', 'seed_pitch_int': marker}
    with pytest.raises(ValueError):
        NoteEvent.from_dict(data)
