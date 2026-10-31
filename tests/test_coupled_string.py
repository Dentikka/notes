"""Physical invariants of the optional string/body boundary and its public renderer."""
import numpy as np
import pytest

from notes import AcousticGuitar, Note, Song, Track
from notes.ears.note_profile import profile_note
from notes.engine import RenderContext, Voice, build_instrument
from notes.engine.coupled_string import body_mobility, coupled_string


def _frequency(y, sr, expected):
    window = y[:min(len(y), sr*2)]
    nfft = 16 * len(window)
    spectrum = abs(np.fft.rfft(window * np.hanning(len(window)), nfft))
    hz = np.fft.rfftfreq(nfft, 1/sr)
    region = np.flatnonzero(abs(hz-expected) < expected*.025)
    at = region[spectrum[region].argmax()]
    a, b, c = np.log(spectrum[at-1:at+2] + 1e-30)
    offset = .5 * (a-c) / (a-2*b+c)
    return (at+offset) * sr/nfft


@pytest.mark.parametrize("size", [.8, 1.1, 1.4])
def test_body_boundary_is_passive(size):
    y = body_mobility(65536, 22050, 0, size, .8)
    assert not y.flags.writeable
    assert y.real.min() >= -1e-12
    for impedance in (.15, 1., 3.):
        reflectance = (impedance*y-1)/(impedance*y+1)
        assert abs(reflectance).max() <= 1+1e-12


@pytest.mark.parametrize("pitch", [40, 44, 55, 64, 76, 80])
def test_loaded_string_remains_in_tune(pitch):
    sr = 22050
    freq = 440 * 2 ** ((pitch-69)/12)
    y = coupled_string(freq, sr*2, sr)
    cents = 1200 * np.log2(_frequency(y, sr, freq)/freq)
    assert abs(cents) < 3


def test_stronger_body_loading_extracts_string_energy():
    sr = 22050
    def relative_tail(coupling):
        y = coupled_string(198, sr*2, sr, coupling=coupling, diffuse=0, body_size=1.)
        profile = profile_note(y, sr, 198)
        early = profile.partial_db[(profile.time >= .05) & (profile.time < .15), 0].mean()
        late = profile.partial_db[(profile.time >= 1.) & (profile.time < 1.2), 0].mean()
        return late-early
    assert relative_tail(.4) < relative_tail(.02)-6


def test_duration_does_not_change_the_earlier_pluck():
    sr = 16000
    short = coupled_string(220, sr, sr, hardness=.4)
    long = coupled_string(220, sr*8, sr, hardness=.4)
    assert np.linalg.norm(short-long[:sr])/np.linalg.norm(short) < .01


@pytest.mark.parametrize("pitch,octave", [(45, True), (64, False)])
def test_coupled_courses_add_the_correct_octave_or_unison(pitch, octave):
    sr = 22050
    freq = 440 * 2 ** ((pitch-69)/12)
    # Separate the partner from the original partials to test course routing,
    # rather than requiring a particular interference gain at 4-cent detuning.
    target = freq * (2 if octave else 1) * 2 ** (40/1200)
    powers = []
    for twelve in (False, True):
        instrument = AcousticGuitar(model="coupled", twelve=twelve, course_detune=40,
                                    pick_noise=0, polarization=0)
        y = instrument.voice(Voice(freq, pitch, 2, .8, {}), RenderContext(sr, 60), np.random.default_rng(17))
        spectrum = abs(np.fft.rfft(y * np.hanning(len(y))))**2
        hz = np.fft.rfftfreq(len(y), 1/sr)
        powers.append(spectrum[abs(hz-target) < 1].sum())
    assert powers[1] > powers[0] * 10


def test_coupled_instrument_articulation_and_roundtrip():
    instrument = AcousticGuitar(model="coupled", fretting=True, pick_noise=0, polarization=0)
    assert build_instrument(instrument.to_spec()) == instrument
    assert instrument.inserts() == ()
    ctx = RenderContext(22050, 60)
    def voice(params, vel=.8):
        return instrument.voice(Voice(329.627557, 64, .8, vel, params), ctx, np.random.default_rng(17))
    a = voice({"string": 1})
    b = voice({"string": 2})
    c = voice({"pick_position": .3, "pick_hardness": .2})
    assert np.isfinite(a).all() and not np.allclose(a, b) and not np.allclose(a, c)
    assert not voice({}, vel=0).any()
    assert abs(a[-1]) < 1e-5
    song = Song(Track("guitar", instrument)(Note("E4", dur=.5).param(bend=.5)), sample_rate=22050)
    result = song.render()
    assert np.isfinite(result.audio).all()
    assert np.isfinite(result.stats["lufs"])
    assert abs(result.stats["lufs"] + 14) < .5


@pytest.mark.parametrize("kwargs", [{"model": "unknown"}, {"coupling": 0}, {"coupling": float("nan")},
                                   {"radiation": 1.1}])
def test_invalid_coupled_instrument_parameters(kwargs):
    with pytest.raises(ValueError):
        AcousticGuitar(**kwargs)
