"""Noise colour must leave the harmonic source, random stream and old mode alone."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import welch

from notes import Cello, Note, Song, Track, Violin
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.ir import Score


def render(instrument, params=None, sr=22050, duration=.6):
    rng = np.random.default_rng(17)
    audio = instrument.voice(Voice(220., 57., duration, .65, params or {}), RenderContext(sr, 60), rng)
    return audio, rng.random(8)


@pytest.mark.parametrize('cls', [Violin, Cello])
@pytest.mark.parametrize('sr', [8000, 22050, 44100])
def test_noise_only_filters_preserve_harmonics_modulation_and_rng(cls, sr):
    source = cls(bow_noise=0, vibrato=.2, bow_noise_seed=41)
    original, state = render(source, sr=sr)
    coloured, other = render(replace(source, bow_noise_highpass=80., bow_noise_lowpass=1200.), sr=sr)
    assert np.array_equal(original, coloured)
    assert np.array_equal(state, other)
    changed, after = render(replace(source, bow_noise=.03, bow_noise_highpass=80., bow_noise_lowpass=1200.), sr=sr)
    assert np.linalg.norm(changed-original) > .001
    assert np.array_equal(state, after)
    assert changed[-1] == 0 and np.isfinite(changed).all()


@pytest.mark.parametrize('cls', [Violin, Cello])
def test_noise_corners_rebalance_low_and_high_noise_without_filtering_the_tone(cls):
    source = cls(vibrato=0, bow_noise=.1, bow_noise_seed=41)
    smooth, _ = render(replace(source, bow_noise=0), duration=3.)
    hp, _ = render(source, {'bow_noise_highpass': 80.}, duration=3.)
    lp, _ = render(source, {'bow_noise_highpass': 80., 'bow_noise_lowpass': 1500.}, duration=3.)
    legacy, _ = render(source, duration=3.)
    def energy(x, lo, hi):
        hz, power = welch((x-smooth)[22050:44100], fs=22050, nperseg=4096)
        return power[(hz >= lo) & (hz < hi)].sum()
    assert energy(hp, 100, 250) > energy(legacy, 100, 250)*20
    assert energy(lp, 5000, 8000) < energy(hp, 5000, 8000)/100
    assert energy(lp, 200, 500) > .95*energy(hp, 200, 500)


@pytest.mark.parametrize('cls', [Violin, Cello])
def test_note_overrides_default_restore_and_score_roundtrip(cls):
    controls = {'bow_noise_highpass': 80., 'bow_noise_lowpass': 1500.}
    source = cls(**controls, bow_noise_seed=29)
    assert build_instrument(source.to_spec()) == source
    assert np.array_equal(render(source)[0], render(cls(bow_noise_seed=29), controls)[0])
    reset = {name: None for name in controls}
    assert np.array_equal(render(source, reset)[0], render(cls(bow_noise_seed=29))[0])
    score = Song(Track('cello', cls())(Note(57, .3).param(**controls, bow_noise_seed=41)),
                 bpm=60, sample_rate=8000, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param('bow_noise_highpass') == 80.
    assert restored.events[0].param('bow_noise_lowpass') == 1500.
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize('name', ['bow_noise_highpass', 'bow_noise_lowpass'])
@pytest.mark.parametrize('value', [0., -1., 9.9, float('nan'), float('inf'), 'bad', []])
def test_invalid_corners_are_rejected_in_constructor_and_note(name, value):
    with pytest.raises(ValueError, match=name):
        Violin(**{name: value})
    with pytest.raises(ValueError, match=name):
        render(Violin(), {name: value})


@pytest.mark.parametrize('name', ['bow_noise_highpass', 'bow_noise_lowpass'])
def test_explicit_corner_above_output_band_is_rejected_rather_than_clamped(name):
    with pytest.raises(ValueError, match=name):
        render(Violin(**{name: 4000.}), sr=8000)
    with pytest.raises(ValueError, match=name):
        render(Violin(), {name: 4000.}, sr=8000)
    audio, _ = render(Violin(**{name: 3600.}), sr=8000)
    assert np.isfinite(audio).all()
