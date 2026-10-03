import numpy as np
import pytest

from notes import RockKit
from notes.engine import RenderContext, Voice
from notes.engine.drum_physics import MEMBRANE_RATIOS, membrane, wash

SR = 44100
SOUNDS = ("kick", "snare", "rim", "tom_lo", "tom_mid", "tom_hi", "hat", "pedal_hat", "open_hat", "crash", "ride",
          "clap", "cowbell")


def _hit(sound, vel=0.8, seed=0, **kw):
    return RockKit(**kw).voice(Voice(0.0, 0.0, 0.1, vel, {"sound": sound}), RenderContext(sr=SR, bpm=120),
                               np.random.default_rng(seed))


def _centroid(y):
    spec = np.abs(np.fft.rfft(y))
    return float(np.sum(spec * np.fft.rfftfreq(y.size, 1 / SR)) / np.sum(spec))


@pytest.mark.parametrize("sound", SOUNDS)
def test_every_rock_kit_sound_renders(sound):
    y = _hit(sound)
    assert y.ndim == 1 and np.isfinite(y).all() and 0.05 < np.max(np.abs(y)) < 1.0


def test_harder_hits_are_brighter_and_every_hit_differs():
    soft, hard = _hit("snare", vel=0.3), _hit("snare", vel=1.0)
    assert _centroid(hard[: SR // 10]) > _centroid(soft[: SR // 10])
    assert not np.allclose(_hit("snare", seed=1), _hit("snare", seed=2))


def test_membrane_rings_at_its_bessel_modes():
    t = np.arange(SR) / SR
    y = membrane(t, 100.0, 0.5, np.random.default_rng(0), strike=1.0, n_modes=4)
    from scipy.signal import find_peaks

    spec = np.abs(np.fft.rfft(y))
    found, _ = find_peaks(spec, distance=8)
    peaks = sorted(found[np.argsort(spec[found])[-4:]])  # 1 Hz bins
    assert np.allclose(peaks, 100.0 * np.array(MEMBRANE_RATIOS[:4]), rtol=0.04)


def test_wash_stays_in_band_and_decays_faster_at_the_top():
    t = np.arange(2 * SR) / SR
    y = wash(t, SR, np.random.default_rng(0), low=2000.0, high=12000.0, tau_low=0.8, tau_high=0.2)

    def band(seg, lo, hi):
        spec = np.abs(np.fft.rfft(seg)) ** 2
        f = np.fft.rfftfreq(seg.size, 1 / SR)
        return spec[(f > lo) & (f < hi)].sum()

    early, late = y[: SR // 4], y[SR // 2 : 3 * SR // 4]
    assert band(early, 50, 300) < 1e-3 * band(early, 2000, 12000)
    assert band(late, 8000, 11000) / band(early, 8000, 11000) < band(late, 2000, 3000) / band(early, 2000, 3000)
