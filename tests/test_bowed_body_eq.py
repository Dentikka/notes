"""Per-note body coloration, unchanged source randomness and safe filter decay."""
import numpy as np
import pytest
from scipy.signal import lfilter

from notes import Cello, ModalBowedString, Note, Song, Track, Violin
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.engine.dsp import biquad
from notes.ir import Score

BANDS = ((201.80698982080165, 4.167141439100775, -35.99983684228459),
         (495.06552545529604, 11.999999999993051, -8.956455446225995))


def voice(instrument, params=None, sr=22050):
    return instrument.voice(Voice(220., 57., .6, .65, params or {}),
                            RenderContext(sr, 60), np.random.default_rng(17))


@pytest.mark.parametrize('cls', [Violin, Cello])
@pytest.mark.parametrize('sr', [8000, 22050, 44100])
def test_body_eq_matches_independent_filtering_and_is_overridden_per_note(cls, sr):
    source = cls(vibrato=0)
    bare = voice(source, sr=sr)
    expected = bare.copy()
    for frequency, q, gain in BANDS:
        b, a = biquad('peak', frequency, sr, q, gain)
        expected = lfilter(b, a, expected)
    count = min(len(expected), max(2, round(.002*sr)))
    expected[-count:] *= np.linspace(1., 0., count)
    corrected = voice(cls(vibrato=0, body_eq=BANDS), sr=sr)
    np.testing.assert_array_equal(corrected, expected)
    np.testing.assert_array_equal(corrected, voice(source, {'body_eq': [list(b) for b in BANDS]}, sr))
    np.testing.assert_array_equal(bare, voice(cls(vibrato=0, body_eq=BANDS), {'body_eq': ()}, sr))
    assert corrected[-1] == 0 and np.isfinite(corrected).all()


def test_empty_and_zero_eq_preserve_full_render_and_ir_roundtrip():
    instrument = Violin(vibrato=0)
    def score(parameters):
        return Song(Track('violin', instrument)(Note('A3', dur=.3).param(**parameters)),
                    bpm=60, seed=17, sample_rate=8000, loudness=None).compile()
    reference = render_score(score({})).audio
    for parameters in ({'body_eq': []}, {'body_eq': [(200., 4., 0.)]}):
        np.testing.assert_array_equal(reference, render_score(score(parameters)).audio)
    original = score({'body_eq': BANDS, 'expression_points': ((0., .3), (.2, 1.))})
    restored = Score.from_json(original.to_json())
    assert restored.events[0].param('body_eq')
    np.testing.assert_array_equal(render_score(original).audio, render_score(restored).audio)
    assert build_instrument(Violin(body_eq=BANDS).to_spec()) == Violin(body_eq=BANDS)


@pytest.mark.parametrize('value', [None, 'bad', [(200.,)], [(200., 0., -6.)],
                                  [(200., 1., float('nan'))], [(200., 1., 25.)],
                                  [(200., 1., -49.)], [(0., 1., -6.)], [BANDS[0]]*9])
def test_invalid_bands_are_rejected_in_constructor_and_note(value):
    with pytest.raises(ValueError, match='body_eq'):
        Violin(body_eq=value)
    with pytest.raises(ValueError, match='body_eq'):
        voice(Violin(), {'body_eq': value})


def test_sample_rate_and_release_cannot_silently_change_the_filter():
    with pytest.raises(ValueError, match='sample rate'):
        voice(Violin(body_eq=((5000., 1., -6.),)), sr=8000)
    with pytest.raises(ValueError, match='release'):
        voice(Violin(release=.001, body_eq=BANDS))


def test_mechanical_source_supports_the_same_output_control_without_changing_excitation():
    source = ModalBowedString(modes=16, bandwidth=4000., oversample=8)
    ctx = RenderContext(8000, 60)
    def render(instrument, params=None):
        return instrument.voice(Voice(110., 45., .3, .7, params or {}), ctx, np.random.default_rng(17))
    bare = render(source)
    expected = bare.copy()
    for frequency, q, gain in BANDS:
        b, a = biquad('peak', frequency, ctx.sr, q, gain)
        expected = lfilter(b, a, expected)
    expected[-16:] *= np.linspace(1., 0., 16)
    corrected = render(ModalBowedString(modes=16, bandwidth=4000., oversample=8, body_eq=BANDS))
    np.testing.assert_allclose(corrected, expected, atol=1e-15)
    np.testing.assert_array_equal(corrected, render(source, {'body_eq': BANDS}))
    np.testing.assert_array_equal(bare, render(source, {'body_eq': ()}))
