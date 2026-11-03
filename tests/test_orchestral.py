import numpy as np
import pytest

from notes import Cello, Piano, Song, Track, Violin, melody
from notes.engine import RenderContext, Voice, build_instrument


def render(instrument, vel=0.8, params=None, sr=22050):
    return instrument.voice(Voice(220, 57, 0.7, vel, params or {}),
                            RenderContext(sr, 60), np.random.default_rng(7))


@pytest.mark.parametrize("instrument", [
    Piano(), Piano(model="soundboard"), Piano(model="hammer"), Violin(vibrato=0), Cello(vibrato=0),
])
def test_tuning_roundtrip_and_dynamics(instrument):
    assert build_instrument(instrument.to_spec()) == instrument
    y = render(instrument)
    assert np.isfinite(y).all()
    spectrum = abs(np.fft.rfft(y[:11025] * np.hanning(11025)))
    freq = np.fft.rfftfreq(11025, 1 / 22050)
    mask = (freq > 200) & (freq < 240)
    assert freq[mask][spectrum[mask].argmax()] == pytest.approx(220, abs=2)
    assert np.sqrt(np.mean(render(instrument, 0.3) ** 2)) < np.sqrt(np.mean(y ** 2)) * 0.5
    assert np.array_equal(y, render(instrument))
    assert abs(y[-1]) < 1e-8


def test_pedal_preserves_string_after_key_release():
    piano = Piano(sustain=2)
    dry = render(piano)
    pedal = render(piano, params={"pedal": True})
    assert len(pedal) > len(dry)
    assert np.sqrt(np.mean(pedal[22050:33075] ** 2)) > 1e-4


def test_bowed_sustain_and_explicit_vibrato_override():
    y = render(Violin(vibrato=0))
    assert np.sqrt(np.mean(y[11025:13230] ** 2)) > 0.03
    assert not np.array_equal(y, render(Violin(vibrato=0), params={"vibrato": 0.4}))


@pytest.mark.parametrize("cls", [Piano, Violin, Cello])
def test_dsl_render(cls):
    result = Song(Track("solo", cls())(melody("A3 C4 E4", step=0.25)),
                  sample_rate=22050, loudness=None).render()
    assert np.isfinite(result.audio).all()


@pytest.mark.parametrize("kwargs", [{"release": 0}, {"level": float("nan")}])
def test_invalid_parameters(kwargs):
    for cls in (Piano, Violin, Cello):
        with pytest.raises(ValueError):
            cls(**kwargs)


def test_soundboard_retains_upper_partials_after_attack():
    def upper_ratio(model):
        y = render(Piano(model=model), params={"pedal": True})[6615:13230]
        spectrum = abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        freq = np.fft.rfftfreq(len(y), 1 / 22050)
        return spectrum[(freq > 600) & (freq < 4000)].sum() / spectrum[(freq > 180) & (freq < 260)].sum()
    assert upper_ratio("soundboard") > 2 * upper_ratio("legacy")


def test_soundboard_zero_velocity_and_low_sample_rate():
    assert not np.any(render(Piano(model="soundboard"), vel=0))
    assert np.isfinite(render(Piano(model="soundboard"), sr=8000)).all()


@pytest.mark.parametrize("pitch", [24, 48, 72, 96])
def test_hammer_contact_registers_and_silence(pitch):
    from notes.engine.hammer import struck_string
    freq = 440 * 2 ** ((pitch - 69) / 12)
    y = struck_string(freq, 22050, 22050, 0.8)
    assert np.isfinite(y).all() and 1e-5 < abs(y).max() < 10
    assert not np.any(struck_string(freq, 2205, 22050, 0))


def test_hammer_velocity_changes_normalized_spectrum():
    from notes.engine.hammer import struck_string
    def centroid(velocity):
        y = struck_string(220, 11025, 22050, velocity)
        power = abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        freq = np.fft.rfftfreq(len(y), 1 / 22050)
        return (freq @ power) / power.sum()
    assert centroid(0.95) > centroid(0.25)


def test_register_readout_preserves_low_note_and_rebalances_upper_modes():
    from notes.engine.hammer import struck_string
    low = struck_string(110, 11025, 22050, 0.8)
    assert np.array_equal(low, struck_string(110, 11025, 22050, 0.8, radiation=True))
    def ratio(radiation):
        f = 440 * 2 ** (7 / 12)
        y = struck_string(f, 22050, 22050, 0.8, radiation=radiation)[2205:11025]
        spectrum = abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        hz = np.fft.rfftfreq(len(y), 1 / 22050)
        return spectrum[abs(hz-2*f)<20].sum() / spectrum[abs(hz-f)<20].sum()
    assert ratio(True) < 0.4 * ratio(False)


def test_attack_presence_preserves_low_register_and_late_tail():
    from notes.engine.hammer import struck_string
    low = struck_string(110, 22050, 22050, 0.8, radiation=True)
    assert np.array_equal(low, struck_string(110, 22050, 22050, 0.8, radiation=True, attack_presence=0.6))
    freq = 440 * 2 ** (7 / 12)
    base = struck_string(freq, 22050, 22050, 0.8, radiation=True)
    changed = struck_string(freq, 22050, 22050, 0.8, radiation=True, attack_presence=0.6)
    def relative(a, b):
        return np.linalg.norm(a-b) / np.linalg.norm(a)
    assert relative(base[:2205], changed[:2205]) > 0.01
    assert relative(base[11025:], changed[11025:]) < 0.0001
    with pytest.raises(ValueError):
        Piano(attack_presence=1.1)


def test_hammer_mass_changes_contact_spectrum_and_supports_per_note_override():
    piano = Piano(model="hammer_damped", hammer_mass_scale=1.7)
    assert build_instrument(piano.to_spec()) == piano
    changed = render(piano)
    original = render(Piano(model="hammer_damped"))
    # Equal output level cannot explain a changed normalized spectrum.
    a, b = original/np.linalg.norm(original), changed/np.linalg.norm(changed)
    assert np.linalg.norm(a-b) > 0.1
    assert np.array_equal(changed, render(Piano(model="hammer_damped"), params={"hammer_mass_scale": 1.7}))
    assert np.array_equal(original, render(piano, params={"hammer_mass_scale": 1.0}))


@pytest.mark.parametrize("mass", [0.25, 4.0])
@pytest.mark.parametrize("freq", [65.4064, 1046.5023, 2093.0045])
def test_hammer_mass_extremes_are_finite_and_end_smoothly(mass, freq):
    piano = Piano(model="hammer_damped", hammer_mass_scale=mass)
    y = piano.voice(Voice(freq, 69+12*np.log2(freq/440), .2, .8, {}),
                    RenderContext(22050, 60), np.random.default_rng(7))
    assert np.isfinite(y).all() and 1e-7 < abs(y).max() < 10
    assert abs(y[-1]) < 1e-10


@pytest.mark.parametrize("mass", [0.0, .2, 4.1, float("nan"), float("inf")])
def test_hammer_mass_rejects_invalid_constructor_and_note_values(mass):
    with pytest.raises(ValueError, match="hammer_mass_scale"):
        Piano(hammer_mass_scale=mass)
    with pytest.raises(ValueError, match="hammer_mass_scale"):
        render(Piano(model="hammer_damped"), params={"hammer_mass_scale": mass})


@pytest.mark.parametrize("model", ["hammer", "hammer_damped"])
@pytest.mark.parametrize("sr", [22050, 44100])
def test_per_note_piano_stiffness_matches_constructor_override(model, sr):
    changed = render(Piano(model=model), params={"stiffness": .0004}, sr=sr)
    assert np.array_equal(changed, render(Piano(model=model, stiffness=.0004), sr=sr))
    assert np.linalg.norm(changed-render(Piano(model=model), sr=sr)) > .01


def test_zero_note_stiffness_and_score_json_roundtrip():
    from notes import Note
    from notes.engine.mixer import render_score
    from notes.ir.score import Score

    piano = Piano(model="hammer_damped")
    assert np.array_equal(render(piano, params={"stiffness": 0}), render(Piano(model="hammer_damped", stiffness=0)))
    score = Song(Track("piano", piano)(Note(76, dur=.2).param(stiffness=.0004, hammer_mass_scale=3.2)),
                 bpm=60, sample_rate=22050, loudness=None).compile()
    restored = Score.from_json(score.to_json())
    assert restored.events[0].param("stiffness") == .0004
    assert np.array_equal(render_score(score).audio, render_score(restored).audio)


@pytest.mark.parametrize("stiffness", [-.001, float("nan"), float("inf"), "bad", None])
def test_invalid_note_stiffness_is_rejected(stiffness):
    with pytest.raises(ValueError, match="stiffness"):
        render(Piano(model="hammer_damped"), params={"stiffness": stiffness})
