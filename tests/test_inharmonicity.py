import json
from dataclasses import asdict

import numpy as np
import pytest

from notes.ears import string_inharmonicity


def _note(f0=329.6276, stiffness=4e-5, glide=0., *, missing_fundamental=False, wrong_partial=False):
    sr = 44100
    time = np.arange(2*sr)/sr
    instantaneous = f0*2**((glide*np.exp(-time/.3))/1200)
    phase = 2*np.pi*np.cumsum(instantaneous)/sr
    wave = np.zeros(len(time))
    for h in range(2 if missing_fundamental else 1, 13):
        ratio = np.sqrt((1+stiffness*h*h)/(1+stiffness))
        ratio *= 2**(20/1200) if wrong_partial and h == 5 else 1
        wave += np.sin(h*phase*ratio+.27*h)/h*np.exp(-time*(1+.04*h))
    return wave, sr


@pytest.mark.parametrize("f0", [82.4069, 329.6276, 1108.7305])
@pytest.mark.parametrize("stiffness", [0., 4e-5, .0003])
@pytest.mark.parametrize("glide", [0., 20.])
def test_recovers_string_stiffness_without_mistaking_common_glide_for_it(f0, stiffness, glide):
    wave, sr = _note(f0, stiffness, glide)
    result = string_inharmonicity(wave, sr, f0)
    assert result.reliable
    assert result.stiffness == pytest.approx(stiffness, abs=max(2e-6, stiffness*.1))


def test_gain_invariance_and_input_preservation():
    wave, sr = _note()
    original = wave.copy()
    a = string_inharmonicity(wave, sr, 329.6276)
    b = string_inharmonicity(wave*.17, sr, 329.6276)
    assert a.reliable and b.reliable
    assert a.stiffness == pytest.approx(b.stiffness, abs=1e-9)
    assert np.array_equal(wave, original)
    json.dumps(asdict(a), allow_nan=False)


@pytest.mark.parametrize("seed", [1, 17, 31])
def test_noise_does_not_become_a_reliable_stiff_string(seed):
    wave = np.random.default_rng(seed).normal(size=88200)
    result = string_inharmonicity(wave, 44100, 329.6276)
    assert not result.reliable
    json.dumps(asdict(result), allow_nan=False)


def test_rejects_missing_fundamental_wrong_note_and_non_string_partial_pattern():
    missing, sr = _note(missing_fundamental=True)
    assert not string_inharmonicity(missing, sr, 329.6276).reliable
    ordinary, sr = _note()
    assert not string_inharmonicity(ordinary, sr, 311.127).reliable
    inconsistent, sr = _note(wrong_partial=True)
    assert not string_inharmonicity(inconsistent, sr, 329.6276).reliable


def test_input_validation():
    wave, sr = _note()
    for invalid in (np.zeros(sr), np.full(sr, np.nan), wave[None, :]):
        with pytest.raises(ValueError, match="mono"):
            string_inharmonicity(invalid, sr, 329.6276)
    for options in ({"start":.01}, {"end":3}, {"start":.5,"end":.6}, {"harmonics":2}):
        with pytest.raises(ValueError):
            string_inharmonicity(wave, sr, 329.6276, **options)
    with pytest.raises(ValueError, match="frequency"):
        string_inharmonicity(wave, sr, float("nan"))
