import json
from dataclasses import replace

import numpy as np
import pytest
from notes import Note, Song, SpectralProfile, SpectralTone, Track
from notes.ears import spectral_profile
from notes.engine import RenderContext, Voice, build_instrument, render_score
from notes.ir import Score


def noisy_profile():
    return SpectralProfile(ratios=(1., 2.01), times=(0., .01, 1., 2.),
                           amplitudes=((0., .1, .1, 0.), (0., .03, .02, 0.)),
                           noise_bands=((300., 1500.),), noise_rms=((0., .01, .01, 0.),))


def voice(instrument, *, duration=1., frequency=220., velocity=.7, seed=21, params=None, sr=22050):
    return instrument.voice(Voice(frequency, 57., duration, velocity, params or {}),
                            RenderContext(sr=sr, bpm=60), np.random.default_rng(seed))


def test_profile_and_source_roundtrip_preserve_seeded_audio():
    instrument = SpectralTone(profile=json.loads(json.dumps(noisy_profile().to_dict())))
    restored = build_instrument(json.loads(json.dumps(instrument.to_spec())))
    assert restored == instrument
    score = Song(Track('spectral', instrument)(Note(57, dur=1)), sample_rate=22050).compile()
    assert np.array_equal(render_score(score).audio, render_score(Score.from_json(score.to_json())).audio)


def test_generated_noise_is_reproducible_without_depends_on_duration():
    source = SpectralTone(profile=noisy_profile())
    short = voice(source, duration=.8)
    long = voice(source, duration=1.4)
    assert np.array_equal(short[:15000], long[:15000])
    assert np.array_equal(short, voice(source, duration=.8))
    assert not np.array_equal(short, voice(source, duration=.8, seed=22))
    assert not np.any(voice(source, velocity=0))
    assert np.allclose(voice(source, velocity=.35), voice(source)*(.5**1.7), atol=1e-15)


def test_noise_band_rms_is_calibrated_from_math_filter():
    profile = SpectralProfile(times=(0., .01, 2.), amplitudes=((0., 0., 0.),),
                              noise_bands=((300., 1500.),), noise_rms=((0., .01, .01),))
    audio = voice(SpectralTone(profile=profile), duration=1.5)
    assert np.sqrt(np.mean(audio[4410:22050]**2)) == pytest.approx(.01, rel=.08)


def test_pitch_and_bend_controls_move_frequency_without_aliasing():
    profile = SpectralProfile(times=(0., .01, 2.), amplitudes=((0., .1, .1),))
    source = SpectralTone(profile=profile)
    cases = [(220., {}, 220.), (440., {}, 440.), (220., {'pitch_points': ((0., 12.),)}, 440.)]
    for frequency, params, expected in cases:
        audio = voice(source, frequency=frequency, params=params)
        x = audio[2205:19845]
        hz = np.fft.rfftfreq(len(x), 1/22050)[np.argmax(abs(np.fft.rfft(x*np.hanning(len(x)))))]
        assert hz == pytest.approx(expected, abs=1.3)
    assert not np.any(voice(source, frequency=11000.))


def test_profile_end_fades_instead_of_holding_unmeasured_sustain():
    source = SpectralTone(profile=SpectralProfile(times=(0., .01, .1), amplitudes=((0., .1, .1),)), release=.05)
    audio = voice(source, duration=1.)
    assert np.max(abs(audio[18000:20000])) < 1e-6


def test_analyzer_keeps_magnitudes_discards_polarity_and_accepts_shorter_than_default_knots():
    sr = 16000
    time = np.arange(sr)/sr
    audio = (.2*np.sin(2*np.pi*220*time+.6)+.1*np.sin(2*np.pi*440*time-1.))
    a = spectral_profile(audio, sr, 220., pitch=57., harmonics=2, normalize_rms=None)
    b = spectral_profile(-audio, sr, 220., pitch=57., harmonics=2, normalize_rms=None)
    assert a == b
    assert len(a.times) <= 16 and len(a.ratios) == 2 and a.times[-1] < 1
    assert np.interp(.2, a.times, a.amplitudes[0]) == pytest.approx(.2, abs=.004)
    assert np.interp(.2, a.times, a.amplitudes[1]) == pytest.approx(.1, abs=.004)
    assert set(a.to_dict()) == {
        'version', 'pitch', 'reference_hz', 'ratios', 'times', 'amplitudes', 'noise_rms', 'noise_bands'}
    assert SpectralProfile.from_dict(json.loads(json.dumps(a.to_dict()))) == a


@pytest.mark.parametrize('changes', [
    {'ratios': (1., 1.)}, {'ratios': (0.,)}, {'ratios': (1.,)*33},
    {'times': (0., .1, .1, 1.)}, {'times': (.01, .02, .3, 1.)},
    {'amplitudes': ((0., -.1, .2, 0.),)}, {'amplitudes': ((0., np.nan, 0., 0.),)},
    {'noise_bands': ((500., 100.),), 'noise_rms': ((0., .01, .01, 0.),)},
    {'noise_bands': ((100., 500.),), 'noise_rms': ((0., .01),)}, {'version': True},
])
def test_invalid_profiles_rejected(changes):
    with pytest.raises(ValueError):
        replace(SpectralProfile(), **changes)


@pytest.mark.parametrize('audio,sr', [(np.zeros(16000), 16000), (np.zeros((2, 16000)), 16000),
                                    (np.full(16000, np.nan), 16000), (np.ones(100), 16000)])
def test_invalid_analysis_inputs_rejected(audio, sr):
    with pytest.raises(ValueError):
        spectral_profile(audio, sr, 220., pitch=57.)


def test_lower_sample_rate_omits_out_of_band_noise_without_filter_error():
    profile = replace(noisy_profile(), noise_bands=((10000., 12000.),))
    a = voice(SpectralTone(profile=profile), sr=8000)
    b = voice(SpectralTone(profile=replace(profile, noise_rms=(), noise_bands=())), sr=8000)
    assert np.array_equal(a, b)
