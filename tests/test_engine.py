import numpy as np
import pytest
from _audio import estimate_f0

from notes import (
    Bass,
    BassGuitar,
    Bell,
    DrumKit,
    ElectricGuitar,
    EPiano,
    Lead,
    Pad,
    Pluck,
    Reverb,
    Song,
    Track,
    chords,
    steps,
)
from notes.engine import Chorus, Compressor, Delay, RenderContext, Voice, read_wav
from notes.engine.dsp import feedback_delay, karplus_strong
from notes.engine.loudness import limit, lufs
from notes.engine.mixer import _render_voices
from notes.export.midi import write_midi
from notes.ir import TempoMap

SR = 44100


@pytest.mark.parametrize("freq", [82.41, 110.0, 220.0, 440.0, 880.0])
def test_string_model_is_in_tune(freq):
    y = karplus_strong(freq, SR, SR, np.random.default_rng(0), t60=4.0)
    cents = 1200 * np.log2(estimate_f0(y[2000:20000], SR) / freq)
    assert abs(cents) < 3


def reference_feedback(x: np.ndarray, lag: int, gain: float) -> np.ndarray:
    """Sample-by-sample y[n] = x[n] + gain * h[n], h = (0.3 + 0.3 z^-1) / (1 - 0.2 z^-1) applied to y[n - lag]."""
    y, h = np.zeros_like(x), 0.0
    for n in range(len(x)):
        u0 = y[n - lag] if n >= lag else 0.0
        u1 = y[n - lag - 1] if n >= lag + 1 else 0.0
        h = 0.3 * u0 + 0.3 * u1 + 0.2 * h
        y[n] = x[n] + gain * h
    return y


@pytest.mark.parametrize("lag", [100, 200])  # dense path and block path
def test_feedback_delay_matches_the_recursion(lag):
    x = np.random.default_rng(1).standard_normal(3000)
    y = feedback_delay(x, lag, 0.9, np.array([0.3, 0.3]), np.array([1.0, -0.2]))
    assert np.allclose(y, reference_feedback(x, lag, 0.9), atol=1e-9)


def test_lufs_calibration_matches_bs1770():
    t = np.arange(10 * SR) / SR
    left = np.sin(2 * np.pi * 997 * t)
    assert lufs(np.stack([left, np.zeros_like(left)]), SR) == pytest.approx(-3.01, abs=0.05)


def test_limiter_respects_ceiling():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, SR)) * 0.5
    y, reduction = limit(x, SR, -1.0)
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-12
    assert reduction > 0


@pytest.mark.parametrize(
    "instrument",
    [Bass(), Pad(), Lead(), EPiano(), Bell(), Pluck(), ElectricGuitar(drive=0.8), ElectricGuitar(drive=0.0),
     BassGuitar()],
)
def test_every_instrument_renders_a_finite_voice(instrument):
    ctx = RenderContext(sr=SR, bpm=120)
    v = Voice(freq=220.0, pitch=57.0, dur=0.5, vel=0.8, params={})
    y = instrument.voice(v, ctx, np.random.default_rng(0))
    assert y.ndim == 1 and len(y) >= int(0.5 * SR)
    assert np.isfinite(y).all() and 0 < np.max(np.abs(y)) < 4


def test_every_drum_sound_renders():
    kit = DrumKit()
    ctx = RenderContext(sr=SR, bpm=120)
    for sound in ("kick", "snare", "clap", "rim", "hat", "pedal_hat", "open_hat", "crash", "ride", "tom_lo", "cowbell"):
        y = kit.voice(Voice(0.0, 0.0, 0.1, 0.8, {"sound": sound}), ctx, np.random.default_rng(0))
        assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.05


@pytest.mark.parametrize("effect", [Reverb(), Delay(), Chorus(), Compressor()])
def test_effects_keep_signals_finite_and_stereo(effect):
    ctx = RenderContext(sr=SR, bpm=120)
    x = np.random.default_rng(3).standard_normal((2, SR)) * 0.1
    y = effect.process(x, ctx)
    assert y.shape == (2, SR) and np.isfinite(y).all()


def tiny_song() -> Song:
    drums = Track("drums", DrumKit())
    keys = Track("keys", EPiano(), fx=[Reverb(mix=0.2)])
    return Song(
        drums(steps(kick="x...x...x...x...", snare="....x.......x...", hat="x.x.x.x.x.x.x.x.")) | keys(chords("Am")),
        bpm=120,
    )


def test_render_is_deterministic_and_sane(tmp_path):
    first, second = tiny_song().render(), tiny_song().render()
    assert np.array_equal(first.audio, second.audio)
    assert np.isfinite(first.audio).all()
    assert np.max(np.abs(first.audio)) <= 10 ** (-1 / 20) + 1e-9
    assert first.duration >= 2.0
    assert first.stats["lufs"] == pytest.approx(-14, abs=1.0)
    path = first.write(tmp_path / "t.wav")
    audio, sr = read_wav(path)
    assert sr == SR and audio.shape == first.audio.shape
    assert np.max(np.abs(audio - first.audio)) < 1e-3


def test_voice_noise_depends_on_content_not_position():
    """Inserting a note must not re-seed the notes after it (seeds hash time and pitch)."""
    kit = DrumKit()
    drums = Track("d", kit)
    ctx, tempo = RenderContext(sr=SR, bpm=120), TempoMap([(0, 120)])
    plain = Song(drums(steps(hat="x.x.x.x."))).compile().events
    extra = Song(drums(steps(hat="xxx.x.x."))).compile().events
    y1 = _render_voices(kit, plain, tempo, ctx, SR, "d")
    y2 = _render_voices(kit, extra, tempo, ctx, SR, "d")
    after_extra = int(0.25 * SR) + 1  # the inserted hat is choked by the hat at 0.25 s
    assert not np.array_equal(y1, y2)
    assert np.array_equal(y1[after_extra:], y2[after_extra:])


def test_midi_export(tmp_path):
    score = tiny_song().compile()
    data = write_midi(score, tmp_path / "t.mid").read_bytes()
    assert data[:4] == b"MThd"
    assert int.from_bytes(data[10:12], "big") == 1 + len(score.tracks)
    assert data.count(b"MTrk") == 1 + len(score.tracks)


def test_parallel_render_is_bit_identical_to_sequential():
    song = tiny_song()
    one = song.render(workers=1)
    many = song.render(workers=2)
    assert many.stats["workers"] == 2
    assert np.array_equal(one.audio, many.audio)


def test_a_late_track_renders_only_its_span():
    from notes.engine.mixer import _render_track

    late = Track("late", EPiano())(chords("Am").shift(8))
    score = Song(Track("d", DrumKit())(steps(kick="x...") * 4) | late).compile()
    offset, stem, _ = _render_track(score, "late")
    assert offset == int(round(score.tempo_map().seconds(8) * score.sample_rate))
    assert stem.shape[0] == 2 and stem.dtype == np.float32


def test_span_rendering_matches_a_full_length_render():
    """Cutting a track to its active span must not change it (effects keep song time)."""
    from dataclasses import replace

    from notes.engine import build_effect, build_instrument
    from notes.engine.mixer import _pan, _render_track, _render_voices

    late = Track("late", EPiano(), fx=[Chorus(mix=0.5), Reverb(mix=0.3)])(chords("Am F").shift(8))
    score = Song(Track("d", DrumKit())(steps(kick="x...") * 8) | late).compile()
    offset, stem, _ = _render_track(score, "late")

    spec, tempo = score.track("late"), score.tempo_map()
    ctx = RenderContext(sr=score.sample_rate, bpm=tempo.bpm_at(0))
    n = offset + stem.shape[1]
    x = _render_voices(build_instrument(spec.instrument), score.events_of("late"), tempo, ctx, n, "late")[None, :]
    x = _pan(x, spec.pan)
    for f in spec.fx:
        x = build_effect(f).process(x, replace(ctx, offset=0))
    assert np.allclose(x[:, offset:], stem, atol=1e-5)
    assert np.max(np.abs(x[:, :offset])) < 1e-12  # FFT convolution leaves only round-off there


def test_every_effect_and_instrument_round_trips_through_its_spec():
    """A parameter may share a name with registry internals (Filter.kind) without breaking specs."""
    from notes.engine import EFFECTS, INSTRUMENTS, build_effect, build_instrument
    from notes.engine.effects import Filter

    spec = Filter(kind="highpass", cutoff=110.0).to_spec()
    assert spec["type"] == "filter" and spec["kind"] == "highpass"
    assert build_effect(spec) == Filter(kind="highpass", cutoff=110.0)
    for name, cls in EFFECTS.items():
        assert build_effect(cls().to_spec()) == cls() and cls().to_spec()["type"] == name
    for name, cls in INSTRUMENTS.items():
        assert build_instrument(cls().to_spec()) == cls() and cls().to_spec()["type"] == name
