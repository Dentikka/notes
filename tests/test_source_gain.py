"""Source gain must precede nonlinear inserts and retain the performance velocity."""
from dataclasses import replace

import numpy as np
import pytest

from notes import ElectricGuitar, Note, Song, Track
from notes.engine import render_score


def _score(params=None):
    instrument = ElectricGuitar(drive=.8, level=.15, pick_noise=0)
    note = Note('A2', dur=.3, vel=.7)
    if params:
        note = note.param(**params)
    return Song(Track('guitar', instrument)(note), bpm=60, sample_rate=16000,
                seed=17, loudness=None).compile()


def test_source_gain_matches_instrument_excitation_gain_before_nonlinear_amp():
    score = _score({'source_gain_db': 6.})
    amplified = render_score(score)
    base = _score()
    spec = base.tracks[0]
    reference = replace(base, tracks=(replace(spec, instrument={**spec.instrument,
        'level': spec.instrument['level']*10**(6/20)}),))
    expected = render_score(reference)
    np.testing.assert_allclose(amplified.audio, expected.audio, rtol=1e-6, atol=1e-8)
    # A volume change after the amp would scale the existing distorted waveform.
    original = render_score(base).audio
    factor = float(np.sum(amplified.audio*original)/np.sum(original*original))
    assert np.linalg.norm(amplified.audio-factor*original)/np.linalg.norm(amplified.audio) > .03
    assert score.events[0].vel == base.events[0].vel == .7
    assert score.events[0].time == base.events[0].time and score.events[0].pitch == base.events[0].pitch


def test_explicit_zero_source_gain_is_bit_exact():
    np.testing.assert_array_equal(render_score(_score()).audio,
                                  render_score(_score({'source_gain_db': 0.})).audio)


@pytest.mark.parametrize('gain', [np.nan, np.inf, -121., 121.])
def test_source_gain_rejects_nonfinite_or_extreme_values(gain):
    with pytest.raises(ValueError, match='source_gain_db'):
        render_score(_score({'source_gain_db': gain}))
