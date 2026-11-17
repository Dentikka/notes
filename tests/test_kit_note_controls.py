"""Independent drum tuning, parameter-aware tails and actual-sound chokes."""
from dataclasses import replace

import numpy as np
import pytest

from notes import DrumKit, Note, RockKit, Song, Track
from notes.engine import RenderContext, Voice, render_score
from notes.ir import Score


@pytest.mark.parametrize('sound,pitch', [('kick', 36), ('snare', 38), ('tom_mid', 47),
                                       ('hat', 42), ('open_hat', 46), ('crash', 49), ('ride', 51)])
def test_note_controls_equal_independent_instrument_without_changing_rng(sound, pitch):
    ctx = RenderContext(8000, 60)
    kit = RockKit()
    voice = Voice(220., float(pitch), .1, .7, {'sound': sound, 'tune': 6., 'decay': .5})
    a, b = np.random.default_rng(17), np.random.default_rng(17)
    x = kit.voice(voice, ctx, a)
    y = replace(kit, tune=6., decay=.5).voice(replace(voice, params={'sound': sound}), ctx, b)
    assert np.array_equal(x, y) and a.random() == b.random()
    assert kit.tune == 0 and kit.decay == 1


def test_long_per_note_ride_retains_its_whole_tail_and_json_roundtrip(tmp_path):
    music = Note(51, .1).param(tune=-6., decay=2.)
    score = Song(Track('kit', RockKit())(music), bpm=60, sample_rate=8000, seed=17, loudness=None).compile()
    path = tmp_path/'score.json'
    score.to_json(path)
    result = render_score(Score.from_json(path))
    equivalent = Song(Track('kit', RockKit(tune=-6., decay=2.))(Note(51, .1)),
                      bpm=60, sample_rate=8000, seed=17, loudness=None).render()
    assert np.array_equal(result.audio, equivalent.audio)
    assert result.duration > 5.8
    # Retained sound above the renderer's -80dBFS tail-trim floor, not roundoff.
    assert np.max(abs(result.audio[:, 5*8000:])) > 1e-4
    assert str(score.events[0].time) == '0' and str(score.events[0].dur) == '1/10'


def test_earlier_long_note_is_not_truncated_by_a_later_short_note():
    music = Note(51, .1).param(decay=2.) | Note(38, .1).param(decay=.5).shift(.2)
    result = Song(Track('kit', RockKit())(music), bpm=60, sample_rate=8000, seed=17, loudness=None).render()
    assert result.duration > 5.8 and np.max(abs(result.audio[:, 5*8000:])) > 1e-4


@pytest.mark.parametrize('cls', [DrumKit, RockKit])
@pytest.mark.parametrize('second_is_hat', [True, False])
def test_choke_uses_actual_sound_instead_of_placeholder_pitch(cls, second_is_hat, monkeypatch):
    # Constant voices make both the cut time and accidental double energy exact.
    monkeypatch.setattr(cls, 'voice', lambda self, voice, ctx, rng: np.full(ctx.sr, .01))
    first = Note(60, .05).param(sound='open_hat')
    second = Note(61 if second_is_hat else 42, .05).param(sound='hat' if second_is_hat else 'kick').shift(.1)
    result = Song(Track('kit', cls())(first | second), bpm=60, sample_rate=8000, loudness=None).render()
    expected = .01*np.cos(np.pi/4)*(1 if second_is_hat else 2)
    assert result.audio[0, 1600] == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize('params', [{'tune': np.nan}, {'tune': 'bad'}, {'decay': 0}, {'decay': -1},
                                  {'decay': np.inf}, {'decay': None}])
def test_invalid_note_controls_rejected_for_tail_and_voice(params):
    kit = RockKit()
    with pytest.raises(ValueError, match='per-note'):
        kit.release_time_for(params)
    with pytest.raises(ValueError, match='per-note'):
        kit.voice(Voice(220., 38., .1, .7, params), RenderContext(8000, 60), np.random.default_rng(17))


@pytest.mark.parametrize('sound,control', [('rim', 'tune'), ('rim', 'decay'), ('pedal_hat', 'tune'),
                                         ('pedal_hat', 'decay'), ('clap', 'tune')])
def test_unimplemented_piece_controls_raise_instead_of_silently_doing_nothing(sound, control):
    with pytest.raises(ValueError, match='unsupported'):
        RockKit().voice(Voice(220., 38., .1, .7, {'sound': sound, control: 1.}),
                        RenderContext(8000, 60), np.random.default_rng(17))


@pytest.mark.parametrize('kwargs', [{'decay': 0}, {'decay': np.nan}, {'tune': np.inf}])
def test_invalid_global_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        RockKit(**kwargs)
