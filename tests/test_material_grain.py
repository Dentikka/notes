"""Spatial invariants and constitutive work, independent of timbre scores."""
import numpy as np
import pytest

from notes.engine.material_grain import SpatialGrain
from notes.engine.thermal_bow import contact, thermal_bowed_string


def test_spatial_history_is_reproducible_under_extension_reversal_and_revisits():
    locations = np.r_[np.linspace(0, .012, 400), np.linspace(.012, -.008, 401), np.linspace(-.008, 0, 300)]
    short, extended = SpatialGrain(seed=17), SpatialGrain(seed=17)
    expected = np.asarray([short(x) for x in locations])
    extended(.05)
    extended(-.1)
    np.testing.assert_array_equal(expected, [extended(x) for x in locations])
    assert abs(expected).max() <= 3.
    assert short(.003) == short(.003)
    assert short(.003) != SpatialGrain(seed=29)(.003)
    assert np.min(1+.25*expected) >= .25


def test_spatial_samples_have_the_designed_correlation_scale_without_sample_normalization():
    field = SpatialGrain(seed=17)
    values = np.asarray([field(i*field.spacing) for i in range(16384)])
    assert abs(values.mean()) < .1
    assert .8 < values.var() < 1.2
    assert np.corrcoef(values[:-1], values[1:])[0, 1] == pytest.approx(np.exp(-.25), abs=.05)
    assert np.corrcoef(values[:-4], values[4:])[0, 1] == pytest.approx(np.exp(-1.), abs=.06)


def test_varying_yield_limit_preserves_contact_work_and_nonnegative_dissipation():
    field = SpatialGrain()
    rng = np.random.default_rng(17)
    previous = 0.
    for i in range(1000):
        limit = .3*(1+.03*field((i-400)*1e-6))
        force, slip, plastic, numerical, _ = contact(float(rng.normal()*.3), .08, previous, limit, 1e5, 1/128000)
        expected = (force**2-previous**2)/(2e5)+plastic+numerical
        assert force*slip/128000 == pytest.approx(expected, abs=1e-16)
        assert plastic >= 0 and numerical >= 0
        previous = force


@pytest.mark.parametrize('factor', [32, 64])
def test_thermal_source_accounts_for_material_variation_without_injecting_energy(factor):
    _, stats = thermal_bowed_string(196., 3200, 8000, gate=.16, attack=.05, modes=16,
        bandwidth=4000., tension=55., length=.328, bodies=((460., 20., .01),), audit=True,
        oversample=factor, roughness=.03, grain_length=1e-4, grain_seed=17)
    stored = sum(stats[k] for k in ('final_mechanical_energy', 'final_contact_energy', 'final_heat_energy'))
    supplied = stats['bow_work']-sum(stats[k] for k in ('mechanical_loss', 'numerical_loss', 'cooling_loss'))
    assert stored == pytest.approx(supplied, abs=1e-10)
    assert stats['maximum_energy_balance_error'] < 1e-12
    assert stats['roughness'] == .03 and stats['grain_seed'] == 17 and stats['grain_length_m'] == 1e-4
    assert stats['material_position'] > 0 and stats['max_plastic_step_over_grain_cell'] < .25


def test_grain_does_not_add_output_noise_or_move_material_while_contact_is_sticking():
    options = {'gate': .2, 'modes': 8, 'bandwidth': 4000., 'bodies': (), 'force_ratio': 10000., 'oversample': 16}
    clean, _ = thermal_bowed_string(196., 40, 8000, **options)
    grain, stats = thermal_bowed_string(196., 40, 8000, roughness=.03, **options)
    assert stats['yielded_fraction'] == 0 and stats['material_position'] == 0
    np.testing.assert_array_equal(clean, grain)


def test_zero_roughness_is_independent_of_grain_seed_and_material_length():
    options = {'gate': .2, 'modes': 8, 'bandwidth': 4000., 'bodies': (), 'oversample': 16}
    a, _ = thermal_bowed_string(196., 640, 8000, roughness=0., grain_seed=17, grain_length=1e-4, **options)
    b, _ = thermal_bowed_string(196., 640, 8000, roughness=0., grain_seed=29, grain_length=2e-4, **options)
    np.testing.assert_array_equal(a, b)


def test_render_duration_does_not_change_material_prefix():
    args = {'gate': .3, 'attack': .05, 'modes': 8, 'bandwidth': 4000., 'bodies': (),
            'roughness': .03, 'grain_length': 1e-4, 'grain_seed': 17, 'oversample': 32}
    a, b = {}, {}
    thermal_bowed_string(196., 640, 8000, trace=a, **args)
    thermal_bowed_string(196., 1280, 8000, trace=b, **args)
    for name in ('force', 'slip', 'temperature'):
        np.testing.assert_array_equal(a[name], b[name][:len(a[name])])


@pytest.mark.parametrize('args', [{'length': 0}, {'length': float('nan')}, {'length': 'bad'},
    {'length': float('inf')}, {'seed': -1}, {'seed': 2**64}, {'seed': 17.}, {'seed': True}])
def test_invalid_field_controls(args):
    with pytest.raises(ValueError):
        SpatialGrain(**args)


@pytest.mark.parametrize('position', [float('inf'), float('nan'), None, 'bad', 1e30])
def test_invalid_or_unallocatable_positions_fail_before_growing_history(position):
    with pytest.raises(ValueError):
        SpatialGrain()(position)


@pytest.mark.parametrize('args', [{'roughness': -.1}, {'roughness': .3}, {'roughness': float('nan')},
    {'grain_length': 0}, {'grain_length': float('inf')}, {'grain_seed': True}, {'grain_seed': -1},
    {'grain_seed': 2**64}, {'grain_seed': 17.}])
def test_invalid_source_grain_controls(args):
    with pytest.raises(ValueError):
        thermal_bowed_string(196., 1, 8000, **args)
