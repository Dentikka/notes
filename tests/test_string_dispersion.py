import numpy as np
import pytest

from notes import AcousticGuitar, BassGuitar, ElectricGuitar, NylonGuitar
from notes.engine import RenderContext, Voice
from notes.engine.registry import build_instrument
from notes.engine.string_model import steel_string


def _render(f0, sr, fit, stiffness=4e-5):
    return steel_string(f0, 3*sr, sr, np.random.default_rng(17), noise=0,
                        second=0, t60=7, t60_high=3, stiffness=stiffness, dispersion_fit=fit)


def _peak_cents(wave, sr, expected):
    # Measure the actual audio, independently of the filter-design phase function.
    wave = wave[round(.1*sr):round(2.5*sr)]
    size = 1 << 20
    spectrum = np.log(abs(np.fft.rfft(wave*np.hanning(len(wave)), size))+1e-30)
    hz = np.fft.rfftfreq(size, 1/sr)
    indices = np.flatnonzero(abs(hz-expected) < expected*.01)
    at = indices[np.argmax(spectrum[indices])]
    a, b, c = spectrum[at-1:at+2]
    offset = .5*(a-c)/(a-2*b+c)
    actual = (at+offset)*sr/size
    return 1200*np.log2(actual/expected)


@pytest.mark.parametrize(("f0", "sr"), [(196., 22050), (493.8833, 44100)])
def test_bracketed_fit_reaches_the_requested_stiff_partial_in_rendered_audio(f0, sr):
    harmonic = max(2, int(1500/f0))
    target = harmonic*f0*np.sqrt(1+4e-5*harmonic**2)
    old = _render(f0, sr, "legacy")
    corrected = _render(f0, sr, "bracketed")
    assert abs(_peak_cents(old, sr, target)) > .2
    assert abs(_peak_cents(corrected, sr, target)) < .1
    assert abs(_peak_cents(corrected, sr, f0)) < .1


def test_zero_stiffness_and_explicit_legacy_preserve_the_original_path():
    assert np.array_equal(_render(196, 22050, "legacy", 0), _render(196, 22050, "bracketed", 0))
    options = dict(noise=0, second=0, t60=7, t60_high=3, stiffness=4e-5)
    implicit = steel_string(196, 66150, 22050, np.random.default_rng(17), **options)
    assert np.array_equal(implicit, _render(196, 22050, "legacy"))


@pytest.mark.parametrize("instrument", [AcousticGuitar, NylonGuitar, ElectricGuitar, BassGuitar])
def test_instruments_serialize_the_optional_dispersion_solver_and_validate_it(instrument):
    selected = instrument(dispersion_fit="bracketed")
    assert build_instrument(selected.to_spec()) == selected
    wave = selected.voice(Voice(196., 55, .4, .7, {}), RenderContext(22050, 60), np.random.default_rng(17))
    assert np.isfinite(wave).all() and 1e-8 < abs(wave).max() < 10
    with pytest.raises(ValueError, match="dispersion_fit"):
        instrument(dispersion_fit="other")


@pytest.mark.parametrize("frequency", [55., 146.8324, 329.6276, 1046.5023])
def test_stiff_string_fit_is_finite_over_the_guitar_register(frequency):
    wave = _render(frequency, 22050, "bracketed", stiffness=.001)
    assert np.isfinite(wave).all() and 1e-8 < abs(wave).max() < 10


def test_string_rejects_unknown_dispersion_solver():
    with pytest.raises(ValueError, match="dispersion_fit"):
        _render(196, 22050, "other")
