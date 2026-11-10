"""Independent amplitude/partial-ratio controls for the timbre-only diagnostic."""
from dataclasses import replace

import numpy as np
import pytest

from notes.ears import harmonic_shape_distance
from notes.ears.note_profile import compare_profiles, profile_note


def test_arbitrary_per_frame_gain_is_removed_but_relative_partial_change_remains():
    sr = 8000
    time = np.arange(sr*2)/sr
    wave = (np.sin(2*np.pi*220*time)+.3*np.sin(2*np.pi*440*time))*np.exp(-time)
    profile = profile_note(wave, sr, 220, harmonics=4)
    ramp = np.linspace(-80., 120., len(profile.time))[:, None]
    expressive = replace(profile, partial_db=profile.partial_db+ramp)
    assert harmonic_shape_distance(profile, expressive) < 1e-12
    # Arbitrarily large common offsets must also stay finite.
    assert harmonic_shape_distance(profile, replace(profile, partial_db=profile.partial_db+10000.)) < 1e-11
    changed = expressive.partial_db.copy()
    changed[:, 1] += 12.
    assert harmonic_shape_distance(profile, replace(expressive, partial_db=changed)) > 3.


def test_two_partial_power_ratios_have_the_expected_db_distance():
    sr = 8000
    time = np.arange(sr)/sr
    base = profile_note(np.sin(2*np.pi*220*time), sr, 220, harmonics=2)
    equal = replace(base, partial_db=np.zeros((len(base.time), 2)))
    # Total power remains 2: the ratios move from (1,1) to (1.5,.5).
    other = replace(base, partial_db=np.tile(10*np.log10([1.5, .5]), (len(base.time), 1)))
    expected = (10*np.log10(1.5)-10*np.log10(.5))/2
    assert harmonic_shape_distance(equal, other) == pytest.approx(expected, abs=1e-12)
    assert harmonic_shape_distance(other, equal) == pytest.approx(expected, abs=1e-12)
    assert harmonic_shape_distance(equal, equal) == 0.


def test_new_and_missing_high_partials_are_not_masked_out():
    sr = 8000
    time = np.arange(sr*2)/sr
    envelope = np.exp(-time)*(1-np.exp(-time/.005))
    wave = envelope*(np.sin(2*np.pi*220*time)+.3*np.sin(2*np.pi*440*time))
    ref = profile_note(wave, sr, 220, harmonics=10)
    extra = profile_note(wave+.4*envelope*np.sin(2*np.pi*220*9*time), sr, 220, harmonics=10)
    assert harmonic_shape_distance(ref, extra) > 3.
    assert harmonic_shape_distance(extra, ref) == pytest.approx(harmonic_shape_distance(ref, extra))


def test_audio_expression_is_distinguished_from_harmonic_shape():
    sr = 16000
    time = np.arange(sr*2)/sr
    carrier = np.sin(2*np.pi*220*time)+.4*np.sin(2*np.pi*440*time)
    fast = profile_note(carrier*np.exp(-time/.3), sr, 220, harmonics=2)
    slow = profile_note(carrier*np.exp(-time/1.5), sr, 220, harmonics=2)
    assert harmonic_shape_distance(fast, slow, start=.2, end=1.4) < .2
    assert compare_profiles(fast, slow)['envelope_mae_db'] > 5.


@pytest.mark.parametrize('options', [
    {'start': -.1}, {'end': 3.}, {'end': .15}, {'step': 0}, {'step': float('nan')},
    {'start': float('inf')}, {'end': float('nan')}, {'end': .17},
    {'audibility_db': 0}, {'floor_db': float('nan')}, {'audibility_db': 61.},
    {'audibility_db': .01},
])
def test_invalid_windows_and_thresholds_fail_explicitly(options):
    sr = 8000
    time = np.arange(sr)/sr
    profile = profile_note(np.sin(2*np.pi*220*time), sr, 220, harmonics=2)
    if options == {'audibility_db': .01}:
        profile = replace(profile, partial_db=np.zeros((len(profile.time), 2)))
    with pytest.raises(ValueError):
        harmonic_shape_distance(profile, profile, **options)
