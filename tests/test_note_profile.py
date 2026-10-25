import numpy as np
import pytest
from notes.ears.note_profile import compare_profiles, profile_note

SR = 22050


def tone(f=220, second=0.3, tau=1.0, silence=0.0):
    t = np.arange(SR*2)/SR
    y = (np.sin(2*np.pi*f*t)+second*np.sin(4*np.pi*f*t))*np.exp(-t/tau)*(1-np.exp(-t/.005))
    return np.pad(y, (round(silence*SR), 0))


def test_gain_and_leading_silence_are_nuisance_variables():
    a = profile_note(tone(), SR, 220)
    b = profile_note(tone(silence=.23)*.2, SR, 220)
    report = compare_profiles(a, b)
    assert abs(report["pitch_cents"]) < 1
    assert report["harmonic_mae_db"] < .3
    assert b.onset-a.onset == pytest.approx(.23, abs=.005)


def test_wrong_timbre_pitch_and_decay_are_visible():
    a = profile_note(tone(), SR, 220)
    bright = compare_profiles(a, profile_note(tone(second=1.8), SR, 220))
    detuned = compare_profiles(a, profile_note(tone(f=224), SR, 220))
    short = compare_profiles(a, profile_note(tone(tau=.3), SR, 220))
    assert bright["harmonic_mae_db"] > 4
    assert detuned["pitch_cents"] > 25
    assert short["envelope_mae_db"] > 4


@pytest.mark.parametrize("x", [np.zeros(SR), np.zeros((2, SR)), np.full(SR, np.nan)])
def test_invalid_note_rejected(x):
    with pytest.raises(ValueError):
        profile_note(x, SR, 220)
