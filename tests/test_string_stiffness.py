"""Measured note-level B must change audible partial frequencies through the public voices."""
from dataclasses import replace

import numpy as np
import pytest

from notes import BassGuitar, ElectricGuitar, Note, Song, Track
from notes.engine import RenderContext, Voice

SR = 22050


@pytest.fixture(params=[ElectricGuitar(amplified=False, pick_noise=0., polarization=0., pitch_attack=0.,
                                     dispersion_fit="bracketed"),
                       BassGuitar(clank=0., polarization=0., pitch_attack=0., dispersion_fit="bracketed")])
def instrument(request):
    return request.param


def voice(instrument, params):
    return instrument.voice(Voice(220., 57., 2., .7, params), RenderContext(SR, 60), np.random.default_rng(17))


def peak(wave, frequency):
    segment = wave[SR//4:SR*7//4]
    size = 8*len(segment)
    spectrum = abs(np.fft.rfft(segment*np.hanning(len(segment)), n=size))
    frequencies = np.fft.rfftfreq(size, 1/SR)
    region = np.flatnonzero(abs(frequencies-frequency) < 15)
    return float(frequencies[region[np.argmax(spectrum[region])]])


def test_note_stiffness_changes_partials_without_moving_the_fundamental(instrument):
    ordinary = voice(instrument, {"stiffness": 0.})
    stiff = voice(instrument, {"stiffness": .0003})
    for wave in (ordinary, stiff):
        assert abs(1200*np.log2(peak(wave, 220.)/220.)) < 2
    harmonic = 6  # The existing dispersion approximation is fitted near 1.5 kHz.
    difference = peak(stiff, 220.*harmonic)-peak(ordinary, 220.*harmonic)
    assert difference > 3.
    np.testing.assert_array_equal(stiff, voice(replace(instrument, stiffness=.0003), {}))
    np.testing.assert_array_equal(voice(instrument, {}), voice(instrument, {"stiffness": instrument.stiffness}))


@pytest.mark.parametrize("value", [-.001, float("nan"), float("inf")])
def test_invalid_note_stiffness_is_rejected(instrument, value):
    with pytest.raises(ValueError, match="stiffness"):
        voice(instrument, {"stiffness": value})


def test_two_notes_keep_separate_stiffness_through_ir_and_the_actual_mixer(instrument):
    from notes.engine import render_score
    from notes.ir.score import Score

    source = Note('A3', dur=.3).param(stiffness=0.) + Note('A3', dur=.3).param(stiffness=.0003)
    score = Song(Track('string', instrument)(source), bpm=60, sample_rate=SR).compile()
    # Use the same serialized IR route as the renderer's worker jobs.
    restored = Score.from_dict(score.to_dict())
    assert [event.param('stiffness') for event in restored.events] == [0., .0003]
    output = render_score(restored)
    assert np.isfinite(output.audio).all() and abs(output.audio).max() > 1e-5
