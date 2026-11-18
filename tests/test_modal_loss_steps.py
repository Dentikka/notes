"""State continuity, passive loss switches and independent physical-coordinate ODEs."""

import numpy as np
import pytest
from notes import ModalString, Note, Song, Track
from notes.engine import RenderContext, Voice, build_instrument
from notes.engine.mechanical_string import _initial_state, _output, _point, modal_string
from notes.engine.palm_contact import _sample_damping_steps, _state, palm_string
from notes.ir import Score
from scipy.linalg import expm, solve


@pytest.mark.parametrize("rounding", ["spatial", "release"])
@pytest.mark.parametrize("peak", [0.0, 0.01])
def test_switch_matches_independent_physical_dynamics_including_filter_memory(rounding, peak):
    args = (110.0, 8, 0.0001, 125.0, 20.0, peak, 70.0, 0.65, 9.0, 2.5, 0.08, 0.04, 0.001)
    a = _state(*args, partial_t60s=(8.0, 3.0))
    b = _state(*args, partial_t60s=(0.2, 12.0, None, 0.4))
    initial = _initial_state(a, 0.2, 1800.0 if rounding == "spatial" else None, 0.25)
    output = _output(a, 0.1, 0.03, 0.0, True)
    sr = 16000
    boundary = 321
    n = 800
    cutoff = 1800.0
    y = _sample_damping_steps(a, [(boundary / sr, b)], initial, output, n, sr, cutoff, rounding)
    size = len(a["M"])
    q = a["U"] @ (initial[:size] / a["omega"])
    v = a["U"] @ initial[size:]
    state = np.r_[q, v, 0.0, 0.0] if rounding == "release" else np.r_[q, v]
    readout = _point(a["h"], 0.1, body=peak > 0, relative=True, width=0.03)
    previous_energy = (q @ a["K"] @ q + v @ a["M"] @ v) / 2
    for model, start, end in [(a, 0, boundary), (b, boundary, n)]:
        for key in ["M", "K", "U", "omega"]:
            np.testing.assert_array_equal(a[key], model[key])
        assert np.linalg.eigvalsh(model["A"] + model["A"].T).max() < 1e-9
        A = np.block(
            [[np.zeros((size, size)), np.eye(size)], [-solve(model["M"], model["K"]), -solve(model["M"], model["C"])]]
        )
        if rounding == "release":
            A = np.pad(A, ((0, 2), (0, 2)))
            c = 2 * np.pi * cutoff
            A[-2, size : 2 * size], A[-2, -2], A[-1, -2], A[-1, -1] = c * readout, -c, c, -c
        for k in [start, start + 1, (start + end) // 2, end - 1]:
            physical = expm(A * ((k - start) / sr)) @ state
            expected = physical[-1] if rounding == "release" else readout @ physical[size : 2 * size]
            assert y[k] == pytest.approx(expected, abs=2e-9)
        state = expm(A * ((end - start) / sr)) @ state
        q, v = state[:size], state[size : 2 * size]
        energy = (q @ model["K"] @ q + v @ model["M"] @ v) / 2
        assert 0 <= energy <= previous_energy * (1 + 1e-10)
        previous_energy = energy


@pytest.mark.parametrize("rounding", ["spatial", "release"])
def test_empty_and_identity_schedule_preserve_pcm_and_shorter_prefix(rounding):
    kwargs = dict(
        position=0.08,
        contact_width=0.04,
        resistance=0.0,
        modes=16,
        peak=0.01,
        mechanical_band=3000.0,
        rounding=rounding,
        partial_t60s=(8.0, 2.0),
    )
    base = palm_string(110.0, 2400, 8000, **kwargs)
    np.testing.assert_array_equal(base, palm_string(110.0, 2400, 8000, partial_t60_steps=(), **kwargs))
    np.testing.assert_array_equal(
        base, palm_string(110.0, 2400, 8000, partial_t60_steps=((0.1, (8.0, 2.0)),), **kwargs)
    )
    steps = ((0.1, (0.5, 4.0)), (0.2, ()))
    long = palm_string(110.0, 2400, 8000, partial_t60_steps=steps, **kwargs)
    short = palm_string(110.0, 1200, 8000, partial_t60_steps=steps, **kwargs)
    np.testing.assert_allclose(long[: len(short)], short, atol=1e-12, rtol=1e-12)
    assert np.max(abs(long - base)) > 1e-6 and np.isfinite(long).all()


def test_zero_step_is_initial_damping_and_later_overdamping_is_retained():
    opts = dict(position=0.08, contact_width=0.04, resistance=0.0, modes=8, peak=0.0, mechanical_band=2000.0)
    a = palm_string(110.0, 1200, 8000, partial_t60_steps=((0.0, (4.0, 0.8)),), **opts)
    b = palm_string(110.0, 1200, 8000, partial_t60s=(4.0, 0.8), **opts)
    np.testing.assert_array_equal(a, b)
    c = palm_string(110.0, 1200, 8000, partial_t60_steps=((0.05, (0.0002,)),), **opts)
    assert np.isfinite(c).all() and np.max(abs(c)) > 0


def test_redundant_schedule_preserves_original_noncontact_path():
    base = modal_string(110., 1200, 8000, modes=8, partial_t60s=(4., .8))
    same = modal_string(110., 1200, 8000, modes=8, partial_t60s=(4., .8),
                        partial_t60_steps=((.05, (4., .8)),))
    np.testing.assert_array_equal(base, same)
    zero = modal_string(110., 1200, 8000, modes=8, partial_t60_steps=((0., (4., .8)),))
    np.testing.assert_array_equal(base, zero)


def test_far_future_loss_change_does_not_overflow_or_change_the_current_note():
    opts = dict(position=.08, contact_width=.04, resistance=0., modes=8, peak=0., mechanical_band=2000.)
    base = palm_string(110., 400, 8000, **opts)
    later = palm_string(110., 400, 8000, partial_t60_steps=((1e308, (1.,)),), **opts)
    np.testing.assert_array_equal(base, later)


@pytest.mark.parametrize(
    "steps",
    [
        None,
        "bad",
        [(0.0,)],
        [(-1.0, ())],
        [(True, ())],
        [(np.inf, ())],
        [(0.0, (0.0,))],
        [(0.0, (True,))],
        [(0.0, (1.0,)), (0.0, (2.0,))],
        [(1.0, ()), (0.5, ())],
        [(0.0, ())] * 65,
    ],
)
def test_invalid_loss_schedules_rejected_before_synthesis(steps):
    with pytest.raises(ValueError):
        ModalString(partial_t60_steps=steps)
    with pytest.raises(ValueError):
        modal_string(110.0, 0, 8000, partial_t60_steps=steps)


def test_note_override_and_json_roundtrip_reach_dsp(tmp_path):
    inst = ModalString(modes=8, bridge_mobility=0, pick_noise=0, partial_t60_steps=((0.05, (0.3, 6.0)),))
    assert build_instrument(inst.to_spec()).partial_t60_steps == inst.partial_t60_steps
    ctx = RenderContext(8000, 60)
    a = inst.voice(Voice(110.0, 45.0, 0.15, 0.7, {}), ctx, np.random.default_rng(1))
    plain = ModalString(modes=8, bridge_mobility=0, pick_noise=0)
    b = plain.voice(
        Voice(110.0, 45.0, 0.15, 0.7, {"partial_t60_steps": inst.partial_t60_steps}), ctx, np.random.default_rng(1)
    )
    np.testing.assert_array_equal(a, b)
    reset = inst.voice(Voice(110.0, 45.0, 0.15, 0.7, {"partial_t60_steps": ()}), ctx, np.random.default_rng(1))
    np.testing.assert_array_equal(reset, plain.voice(Voice(110.0, 45.0, 0.15, 0.7, {}), ctx, np.random.default_rng(1)))
    score = Song(Track("g", plain)(Note("A2", 0.15).param(partial_t60_steps=((0.05, (0.3, 6.0)),)))).compile()
    path = tmp_path / "score.json"
    score.to_json(path)
    restored = Score.from_json(path)
    assert restored.events[0].param("partial_t60_steps")[0][0] == 0.05
