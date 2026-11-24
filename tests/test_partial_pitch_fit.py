from dataclasses import replace

import numpy as np
import pytest
from notes import ElectricGuitar
from notes.ears import fit_partial_pitch_cents
from notes.ears.inharmonicity import (
    InharmonicPartial,
    StringInharmonicity,
    string_inharmonicity,
)
from notes.engine import RenderContext, Voice


def measurement(values, *, f0=130.8, **changes):
    rows = tuple(replace(InharmonicPartial(h, 2 ** (cents / 1200), -10., 40., .02, 1., True), **changes)
                 for h, cents in values.items())
    return StringInharmonicity(f0, None, None, rows, False, 'shape_mismatch')


def test_pool_individual_support_without_assuming_global_B_law():
    model = measurement({2: 3., 5: 1.})
    pairs = [(measurement({2: 3.2, 5: 6.5}), model), (measurement({2: 3.4, 5: 6.7}), model)]
    fit = fit_partial_pitch_cents(pairs, current=(.4, 2., 0., 0., 0.), harmonics=5)
    assert fit.offsets[0] == .4
    assert fit.offsets[1] == pytest.approx(2.3)
    assert fit.offsets[4] == pytest.approx(5.6)
    assert fit.spans_cents[4] == pytest.approx(.2)
    assert fit.support_counts == (0, 2, 0, 0, 2)
    assert fit.reasons[3] == 'few_supported_observations'
    subset = fit_partial_pitch_cents(pairs, current=(.4, 2., 0., 0., 0.), harmonics=5, numbers=(5,))
    assert subset.offsets[1] == 2.
    assert subset.reasons[1] == 'not_selected'
    single = fit_partial_pitch_cents(pairs[:1], current=(.4, 2., 0., 0., 0.), harmonics=5)
    assert single.offsets == (.4, 2., 0., 0., 0.)
    assert single.reasons[4] == 'few_supported_observations'


@pytest.mark.parametrize('changes', [dict(accepted=False), dict(coverage=.5),
                                   dict(phase_residual_rms=.3), dict(relative_level_db=-50.)])
def test_unusable_components_preserve_existing_offsets(changes):
    reference, model = measurement({5: 7.}, **changes), measurement({5: 1.})
    fit = fit_partial_pitch_cents([(reference, model)] * 2, current=(0., 0., 0., 0., 3.), harmonics=5)
    assert fit.offsets[4] == 3.
    assert not fit.accepted[4]
    assert fit.support_counts[4] == 0


def test_inconsistent_and_out_of_range_corrections_are_not_clipped():
    model = measurement({5: 0.})
    fit = fit_partial_pitch_cents([(measurement({5: 1.}), model), (measurement({5: 5.}), model)], harmonics=5)
    assert fit.offsets[4] == 0.
    assert fit.reasons[4] == 'inconsistent_relative_frequency'
    pairs = [(measurement({5: 6.}), model)] * 2
    fit = fit_partial_pitch_cents(pairs, harmonics=5, current=(0., 0., 0., 0., 98.))
    assert fit.offsets[4] == 98.
    assert fit.reasons[4] == 'offset_limit'
    fit = fit_partial_pitch_cents([(measurement({5: 70.}), model)] * 2, harmonics=5)
    assert fit.reasons[4] == 'delta_limit'


def test_absolute_tuning_is_not_copied_and_wrong_anchor_is_rejected():
    pairs = [(measurement({5: 6.}, f0=131.1), measurement({5: 1.}, f0=130.))] * 2
    fit = fit_partial_pitch_cents(pairs, harmonics=5)
    assert fit.deltas[4] == pytest.approx(5.)
    with pytest.raises(ValueError, match='same known anchor'):
        fit_partial_pitch_cents([(measurement({5: 6.}, f0=196.), measurement({5: 1.}))] * 2)


def test_native_source_frequency_control_reduces_measured_error():
    sr = 12000
    ctx = RenderContext(sr, 60)
    v = Voice(130.8, 48., 1.6, .7, {'string': 5, 'pitch_interpolation': 'sinc'})
    inst = ElectricGuitar(amplified=False, fretting=True, pickup_position=.06, pickup_width=.03,
                         pluck_sampling='spectral', dispersion_fit='bracketed', pitch_attack=0., pick_noise=0.,
                         pick_position=.09)

    def render(offsets):
        model = replace(inst, partial_pitch_cents=offsets)
        x = model.voice(v, ctx, np.random.default_rng(91))
        for effect in model.inserts():
            x = effect.process(x, ctx)
        return x

    target, model = render((0., 0., 0., 0., 6.)), render(())
    windows = [(.2, .65), (.65, 1.3)]
    refs = [string_inharmonicity(target, sr, 130.8, start=a, end=b, harmonics=6) for a, b in windows]
    sources = [string_inharmonicity(model, sr, 130.8, start=a, end=b, harmonics=6) for a, b in windows]
    fit = fit_partial_pitch_cents(list(zip(refs, sources, strict=True)), harmonics=6, numbers=(5,))
    assert fit.accepted[4]
    assert fit.offsets[4] == pytest.approx(6., abs=.1)
    corrected = render(fit.offsets)
    after = [string_inharmonicity(corrected, sr, 130.8, start=a, end=b, harmonics=6) for a, b in windows]
    residual = fit_partial_pitch_cents(list(zip(refs, after, strict=True)), harmonics=6, numbers=(5,))
    assert abs(residual.deltas[4]) < .1


@pytest.mark.parametrize('kwargs', [dict(harmonics=1), dict(min_observations=0), dict(max_span_cents=-1.),
                                  dict(numbers=(5, 5)), dict(numbers=(1,)), dict(current=(101.,))])
def test_invalid_guards_or_controls_are_rejected(kwargs):
    pair = (measurement({5: 2.}), measurement({5: 1.}))
    with pytest.raises(ValueError):
        fit_partial_pitch_cents([pair, pair], **kwargs)
