import numpy as np
import pytest

from notes import AcousticGuitar, NylonGuitar
from notes.engine import RenderContext, build_effect, build_instrument
from notes.engine.effects.body import Body, body_ir


@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
def test_minimum_phase_preserves_body_magnitude_and_concentrates_early_energy(sr):
    old = body_ir(sr)
    new = body_ir(sr, phase="minimum")
    assert old.shape == new.shape and np.isfinite(new).all()
    a, b = abs(np.fft.rfft(old, 131072)), abs(np.fft.rfft(new, 131072))
    assert np.linalg.norm(a-b)/np.linalg.norm(a) < .001
    for milliseconds in (1, 3, 5, 10):
        end = round(sr*milliseconds/1000)
        assert np.sum(new[:end]**2) >= np.sum(old[:end]**2)*.999


@pytest.mark.parametrize("cls", [AcousticGuitar, NylonGuitar])
def test_instrument_phase_option_roundtrips_and_is_wired_into_its_body(cls):
    instrument = cls(body_phase="minimum")
    assert build_instrument(instrument.to_spec()) == instrument
    assert instrument.inserts()[-1].phase == "minimum"
    with pytest.raises(ValueError):
        cls(body_phase="unknown")


def test_effect_roundtrip_and_streaming_tail_contract():
    body = Body(phase="minimum")
    assert build_effect(body.to_spec()) == body
    ctx = RenderContext(22050, 60)
    x = np.zeros((2, 22050))
    x[:, 0] = 1
    y = body.process(x, ctx)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert np.array_equal(y[0], y[1])
    assert body.tail(ctx) == Body().tail(ctx)
    with pytest.raises(ValueError):
        Body(phase="unknown")


@pytest.mark.parametrize("mix", [0., .25, .5, 1.])
def test_phase_change_preserves_the_complete_dry_wet_transfer(mix):
    ctx = RenderContext(22050, 60)
    impulse = np.zeros((1, 22050))
    impulse[0, 0] = 1
    old = Body(mix=mix).process(impulse, ctx)[0]
    new = Body(mix=mix, phase="minimum").process(impulse, ctx)[0]
    a, b = abs(np.fft.rfft(old, 131072)), abs(np.fft.rfft(new, 131072))
    assert np.linalg.norm(a-b)/np.linalg.norm(a) < .001
