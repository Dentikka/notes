import numpy as np

from notes.ears.transcribe import PitchTrack, notes_from_track, onsets_from_track


def test_accompaniment_attacks_do_not_restrike_a_sustained_lead():
    t = np.arange(200) * 0.005
    track = PitchTrack(t, np.full(200, 330.0), np.ones(200), np.full(200, -20.0))
    # Full-mix onsets split the same held note into four artificial plucks.
    assert len(notes_from_track(track, [0, 0.25, 0.5, 0.75])) == 4
    assert len(notes_from_track(track, onsets_from_track(track))) == 1


def test_a_real_repick_in_pitch_energy_still_splits_the_note():
    t = np.arange(200) * 0.005
    level = -20 - t * 12
    level[100:] += 8
    track = PitchTrack(t, np.full(200, 330.0), np.ones(200), level)
    onsets = onsets_from_track(track)
    assert len(onsets) == 2
    assert abs(onsets[1] - 0.5) < 0.04


def test_unvoiced_silence_has_no_pitch_onsets():
    t = np.arange(200) * 0.005
    track = PitchTrack(t, np.full(200, np.nan), np.zeros(200), np.full(200, -200.0))
    assert not len(onsets_from_track(track))


def test_harmonic_levels_ignore_low_accompaniment_attacks_in_audio():
    from notes.ears.transcribe import melody_track

    sr = 16000
    t = np.arange(sr * 2) / sr
    x = 0.2 * np.sin(2 * np.pi * 330 * t)
    for start in (0.3, 0.8, 1.3):
        attack = (t >= start) & (t < start + 0.15)
        x[attack] += 3 * np.sin(2 * np.pi * 120 * t[attack]) * np.hanning(attack.sum())
    whole = melody_track(x, sr, 300, 350, n_fft=2048, min_clarity=0.1)
    selected = melody_track(x, sr, 300, 350, n_fft=2048, min_clarity=0.1, pitch_level=True)
    assert len(onsets_from_track(whole)) >= 3
    assert len(onsets_from_track(selected)) == 1
