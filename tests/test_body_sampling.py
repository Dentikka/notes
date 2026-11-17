"""A fixed procedural filter must retain gain, phase and latency across rates."""
import numpy as np
import pytest
from scipy.signal import freqz, resample_poly

from notes import AcousticGuitar, Body, Note, NylonGuitar, Song, Track
from notes.engine import RenderContext, Voice, build_effect, build_instrument, render_score
from notes.engine.effects.body import body_ir
from notes.ir import Score


@pytest.mark.parametrize('phase', ['designed', 'minimum'])
@pytest.mark.parametrize('mix', [0., .35, 1.])
@pytest.mark.parametrize('sr', [4000, 8000, 22050, 48000])
def test_reference_complex_transfer_is_retained_across_sample_rates(sr, mix, phase):
    responses = []
    frequencies = np.geomspace(80, min(8000, sr*.3), 128)
    for rate in [44100, sr]:
        impulse = np.zeros((1, rate//2))
        impulse[0, 0] = 1.
        response = Body(sampling='fixed', mix=mix, phase=phase).process(impulse, RenderContext(rate, 60))[0]
        _, spectrum = freqz(response, worN=frequencies, fs=rate)
        responses.append(spectrum)
    assert np.linalg.norm(responses[0]-responses[1])/np.linalg.norm(responses[0]) < .004


@pytest.mark.parametrize('sr', [4000, 22050, 44100, 48000])
def test_dry_reference_keeps_unit_dc_gain_and_a_common_causal_delay(sr):
    ctx = RenderContext(sr, 60)
    impulse = np.zeros((1, sr//2))
    impulse[0, 0] = 1.
    response = Body(sampling='fixed', mix=0).process(impulse, ctx)[0]
    # This independently catches a missing reference_rate/output_rate factor.
    assert abs(response.sum()-1) < .002
    delay_seconds = np.arange(len(response))@response/response.sum()/sr
    assert delay_seconds == pytest.approx(132/44100, abs=.15/sr)
    assert np.max(abs(response[:max(1, round(.0004*sr))])) < 1e-12
    assert len(body_ir(sr, sampling='fixed')) <= round(Body(sampling='fixed').tail(ctx)*sr)


@pytest.mark.parametrize('phase', ['designed', 'minimum'])
def test_at_reference_rate_the_original_filter_is_only_delayed(phase):
    response = body_ir(44100, phase=phase, sampling='fixed')
    original = body_ir(44100, phase=phase)
    assert np.array_equal(response[132:-132], original)
    assert not np.any(response[:132]) and not np.any(response[-132:])


@pytest.mark.parametrize('cls', [AcousticGuitar, NylonGuitar])
@pytest.mark.parametrize('model', ['waveguide', 'modal'])
def test_public_instrument_registry_and_json_pcm_retain_fixed_body(cls, model, tmp_path):
    instrument = cls(model=model, body_sampling='fixed', body_phase='minimum', pick_noise=0)
    assert build_instrument(instrument.to_spec()) == instrument
    assert build_effect(Body(sampling='fixed').to_spec()) == Body(sampling='fixed')
    if model == 'waveguide':
        assert instrument.inserts()[-1].sampling == 'fixed'
    else:
        assert instrument.release_time() == pytest.approx(instrument.release+.257)
    score = Song(Track('g', instrument)(Note('A3', .1)), bpm=60, sample_rate=8000,
                 loudness=None, seed=117).compile()
    path = tmp_path/'score.json'
    score.to_json(path)
    a, b = render_score(score), render_score(Score.from_json(path))
    assert np.array_equal(a.audio, b.audio) and np.isfinite(a.audio).all()


@pytest.mark.parametrize('freq,pitch', [(82.4, 40.), (329.6, 64.), (659.3, 76.)])
def test_modal_voice_keeps_the_same_body_after_resampling(freq, pitch):
    waves = []
    inst = AcousticGuitar(model='modal', body_sampling='fixed', pick_noise=0, fretting=True)
    for sr in [22050, 44100]:
        wave = inst.voice(Voice(freq, pitch, .3, .7, {}), RenderContext(sr, 60), np.random.default_rng(117))
        waves.append(wave[:round(.3*sr)] if sr == 22050 else resample_poly(wave[:round(.3*sr)], 1, 2))
    a, b = waves
    assert np.linalg.norm(a-b)/np.linalg.norm(a) < .04


def test_unknown_sampling_and_unsupported_mechanical_body_are_rejected():
    with pytest.raises(ValueError, match='sampling'):
        Body(sampling='unknown')
    with pytest.raises(ValueError, match='sampling'):
        AcousticGuitar(body_sampling='unknown')
    with pytest.raises(ValueError, match='requires'):
        AcousticGuitar(model='coupled', body_sampling='fixed')
    for sr in [3000, 4000.5, np.nan]:
        with pytest.raises(ValueError, match='integer rate'):
            body_ir(sr, sampling='fixed')
