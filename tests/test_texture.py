"""Known spectral fractions and inverse-control verification without audio ML."""
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from notes.ears.texture import estimate_roughness, fit_roughness, interharmonic_energy

SR = 22050
TIME = np.arange(3*SR)/SR
TONE = np.sin(2*np.pi*200*TIME)+np.sin(2*np.pi*2000*TIME)
BAND = (1000., 3500.)


def measurement(y):
    return interharmonic_energy(y, SR, 200., band=BAND)


def renderer(roughness, seed):
    return TONE+roughness*np.random.default_rng(seed).normal(size=len(TONE))


def test_known_between_harmonics_tone_fraction_and_gain_invariance():
    wave = TONE+.2*np.sin(2*np.pi*2100*TIME)
    a, b = measurement(wave), measurement(wave*.1)
    assert a.fraction == pytest.approx(.2**2/(1+.2**2), rel=.01)
    assert a.fraction == pytest.approx(b.fraction, abs=1e-12)
    assert a.db == pytest.approx(b.db, abs=1e-10)
    assert a.band_audible and -4 < a.band_relative_db < -2
    assert a.tracked_f0_min == pytest.approx(200., abs=1.)
    assert a.tracked_f0_max == pytest.approx(200., abs=1.)


def test_added_noise_is_monotonic_and_inaudible_upper_bands_are_flagged():
    fractions = [measurement(renderer(r, 17)).fraction for r in (0., .01, .1)]
    assert fractions[0] < fractions[1] < fractions[2]
    quiet = measurement(np.sin(2*np.pi*200*TIME))
    assert not quiet.band_audible and quiet.band_relative_db < -60
    estimate = estimate_roughness(quiet, quiet)
    assert estimate.roughness is None and estimate.reason == 'inaudible_band'
    decay = TONE.copy()
    decay[SR:] = np.random.default_rng(17).normal(size=len(decay)-SR)*1e-7
    tail = measurement(decay)
    assert tail.window_relative_db < -60 and not tail.band_audible


def test_pure_frequency_modulation_also_creates_residual_without_random_noise():
    amplitudes = 1/np.arange(1, 21)
    steady = sum(a*np.sin(2*np.pi*200*h*TIME) for h, a in enumerate(amplitudes, 1))
    phase = 2*np.pi*np.cumsum(200*2**(.3*np.sin(2*np.pi*5*TIME)/12))/SR
    modulated = sum(a*np.sin(h*phase) for h, a in enumerate(amplitudes, 1))
    assert interharmonic_energy(modulated, SR, 200.).fraction > 100*interharmonic_energy(steady, SR, 200.).fraction


def test_quadratic_estimate_recovers_a_small_known_control_and_bounds_it():
    baseline = measurement(renderer(0., 17))
    target = measurement(renderer(.06, 41))
    probes = [measurement(renderer(.1, seed)) for seed in (17, 29)]
    estimate = estimate_roughness(target, baseline, probes, probe_roughness=.1)
    assert estimate.roughness == pytest.approx(.06, abs=.006)
    assert estimate.reason == 'prediction' and not estimate.limited
    bounded = estimate_roughness(target, baseline, probes, probe_roughness=.1, max_roughness=.02)
    assert bounded.roughness == .02 and bounded.limited
    same = estimate_roughness(target, target)
    assert same.roughness == 0 and same.reason == 'baseline_within_tolerance'
    excess = estimate_roughness(baseline, target)
    assert excess.roughness == 0 and excess.reason == 'baseline_excess'
    unsupported = estimate_roughness(target, baseline, [baseline], probe_roughness=.1)
    assert unsupported.roughness is None and unsupported.reason == 'probe_below_resolution'


def test_a_sufficient_baseline_skips_all_positive_renders_and_preserves_samples():
    calls = []

    def render(r, seed):
        calls.append((r, seed))
        return renderer(r, seed)

    ref = renderer(0., 17)
    result = fit_roughness(ref, render, SR, 200., band=BAND)
    assert calls == [(0., 17)] and result.roughness == 0 and result.verified_within_tolerance
    np.testing.assert_array_equal(result.audio, ref)


def test_probe_mean_and_fresh_seed_are_recorded_and_verified():
    reference = renderer(.06, 41)
    observed = []
    result = fit_roughness(reference, renderer, SR, 200., band=BAND, probe_roughness=.1, on_trial=observed.append)
    assert len(result.history) == len(observed) == 4
    assert [r.seed for r in observed] == [17, 17, 29, 41]
    assert result.roughness == pytest.approx(.06, abs=.006) and result.verified_within_tolerance
    assert not result.verification_regressed
    json.dumps([asdict(r) for r in observed])


def test_bad_verification_rolls_back_even_if_renderer_reuses_a_buffer():
    buffer = TONE.copy()

    def render(r, seed):
        buffer[:] = renderer(0. if r == 0 else (r if seed != 41 else 2.), seed)
        return buffer

    result = fit_roughness(renderer(.06, 41), render, SR, 200., band=BAND, probe_roughness=.1)
    assert result.verification_regressed and result.roughness == 0
    np.testing.assert_array_equal(result.audio, TONE)


def test_inaudible_verification_is_not_accepted_by_its_ratio():
    def render(r, seed):
        return np.sin(2*np.pi*200*TIME) if seed == 41 else renderer(r, seed)

    result = fit_roughness(renderer(.06, 41), render, SR, 200., band=BAND, probe_roughness=.1)
    assert result.verification_regressed and not result.verified_within_tolerance and result.roughness == 0


def test_protocol_mismatch_and_invalid_fractions_cannot_drive_a_fit():
    a = measurement(renderer(.06, 17))
    with pytest.raises(ValueError, match='same protocol'):
        estimate_roughness(a, replace(a, start=.5))
    with pytest.raises(ValueError, match='fractions'):
        estimate_roughness(a, replace(a, fraction=float('nan')))


@pytest.mark.parametrize('options', [{'start': -1}, {'end': 5}, {'start': 1., 'end': 1.01},
    {'band': (3000, 1000)}, {'band': (float('nan'), 3000)}, {'band': (12000, 13000)},
    {'sr': 0}, {'sr': True}, {'f0': 10}, {'f0': float('nan')}])
def test_invalid_texture_measurement_controls(options):
    with pytest.raises(ValueError):
        interharmonic_energy(TONE, **{'sr': SR, 'f0': 200., **options})


@pytest.mark.parametrize('options', [{'probe_roughness': 0}, {'probe_roughness': .3}, {'max_roughness': 0},
    {'tolerance_db': -1}, {'seeds': ()}, {'seeds': (17, 17)}, {'seeds': (True,)}, {'verification_seed': 17}])
def test_invalid_fit_controls_fail_before_rendering(options):
    def render(r, seed):
        raise AssertionError('invalid fitting controls must not render')

    with pytest.raises(ValueError):
        fit_roughness(TONE, render, SR, 200., **options)


def test_cli_exposes_texture_as_a_separate_optional_axis(tmp_path):
    from notes.cli import main
    from notes.engine.audio import write_wav
    reference, candidate, output = (tmp_path/p for p in ('reference.wav', 'candidate.wav', 'report.json'))
    write_wav(reference, TONE*.1, SR)
    write_wav(candidate, (TONE+.2*np.sin(2*np.pi*2100*TIME))*.1, SR)
    assert main(['audit-note', str(reference), str(candidate), '--pitch', 'G3', '--a4', str(200/2**(-14/12)),
                 '--texture-window', '1', '2.5', '--texture-band', '1000', '3500', '-o', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['status'] == 'diagnostic-only' and report['texture']['difference_db'] > 10
    assert report['texture']['reference']['band_audible']
