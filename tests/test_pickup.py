import numpy as np
import pytest
from scipy.signal import freqz

from notes import ElectricGuitar
from notes.engine import RenderContext, Voice, build_instrument
from notes.engine.pickup import pickup_kernel, spatial_pickup


@pytest.mark.parametrize("sr", [22050, 44100])
@pytest.mark.parametrize("freq", [82.4069, 329.6276, 659.2551])
@pytest.mark.parametrize("geometry", [(.09, 0., 0.), (.06, .03, 0.), (.09, .02, .03)])
def test_spatial_fir_matches_closed_form_mode_response(sr, freq, geometry):
    position, width, spacing = geometry
    impulse = pickup_kernel(freq, sr, position, width, spacing)
    modes = np.arange(1, int(.35*sr/freq)+1)
    omega = 2*np.pi*modes*freq/sr
    _, response = freqz(impulse, worN=omega)
    response *= np.exp(1j*omega*16)
    expected = (1j*np.exp(-1j*np.pi*modes*(position+(width+spacing)/2))
                * np.sin(np.pi*modes*position)*np.cos(np.pi*modes*spacing/2)*np.sinc(modes*width/2))
    expected /= np.sin(np.pi*position)*np.cos(np.pi*spacing/2)*np.sinc(width/2)
    assert abs(response-expected).max() < .002
    assert abs(response[0]) == pytest.approx(1, abs=.001)
    assert not impulse.flags.writeable


def test_fractional_geometry_places_the_null_between_integer_delay_choices():
    freq, sr, position = 330., 22050, .1
    omega = 2*np.pi*(freq/position)/sr
    _, response = freqz(pickup_kernel(freq, sr, position), worN=np.array([omega]))
    old_lag = round(position*sr/freq)
    old_response = (1-np.exp(-1j*omega*old_lag))/(2*np.sin(np.pi*position))
    assert abs(response[0]) < .001
    assert abs(old_response) > .1


@pytest.mark.parametrize("position,width,spacing,mode", [(.25, 0., .1, 10), (.17, .1, 0., 20)])
def test_aperture_and_coil_spacing_have_their_own_nulls(position, width, spacing, mode):
    omega = np.array([2*np.pi*mode*220/44100])
    _, point = freqz(pickup_kernel(220, 44100, position), worN=omega)
    _, finite = freqz(pickup_kernel(220, 44100, position, width, spacing), worN=omega)
    assert abs(finite[0]) < abs(point[0])*.01


def test_spatial_readout_removes_only_numerical_lookahead_and_preserves_input():
    source = np.zeros(1024)
    source[100] = 1
    before = source.copy()
    result = spatial_pickup(source, 330, 22050, .1)
    assert result.shape == source.shape and abs(result).argmax() == 100
    assert np.array_equal(source, before)
    assert spatial_pickup(np.zeros(0), 330, 22050, .1).size == 0


def test_instrument_roundtrip_and_electrical_override_are_independent():
    instrument = ElectricGuitar(amplified=False, fretting=True, pickup_position=.06,
                                pickup_width=.04, pickup_cutoff=4000, pickup_q=.7)
    assert build_instrument(instrument.to_spec()) == instrument
    context = RenderContext(22050, 60)
    voice = Voice(329.6276, 64, .5, .7, {"string": 1})
    y = instrument.voice(voice, context, np.random.default_rng(17))
    assert y.shape == (12348,) and np.isfinite(y).all() and abs(y).max() > 1e-4
    legacy = ElectricGuitar(amplified=False)
    electrical = ElectricGuitar(amplified=False, pickup_cutoff=2800, pickup_q=.7)
    assert np.array_equal(legacy.voice(voice, context, np.random.default_rng(17)),
                          electrical.voice(voice, context, np.random.default_rng(17)))
    assert legacy.inserts() != electrical.inserts()


@pytest.mark.parametrize("kwargs", [
    {"pickup_position": 0}, {"pickup_width": -1}, {"coil_spacing": float("nan")},
    {"pickup_position": .02, "pickup_width": .1}, {"pickup_cutoff": 0}, {"pickup_q": float("inf")},
])
def test_invalid_hardware_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ElectricGuitar(**kwargs)


def test_fretted_aperture_cannot_move_past_the_fret():
    instrument = ElectricGuitar(pickup="neck", fretting=True, pickup_width=.04)
    with pytest.raises(ValueError, match="vibrating string"):
        instrument.voice(Voice(1318.5102, 88, .2, .7, {"string": 1}),
                         RenderContext(22050, 60), np.random.default_rng(17))
    with pytest.raises(ValueError, match="finite"):
        spatial_pickup(np.array([np.nan]), 330, 22050, .1)
