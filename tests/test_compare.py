import json

import numpy as np
import pytest
from scipy.signal import butter, sosfilt, sosfiltfilt

from notes.cli import _seconds, main
from notes.ears.compare import ANALYSIS_RATE as SR
from notes.ears.compare import compare, listening, load
from notes.engine.audio import write_wav

ARPEGGIO = [57, 60, 64, 69, 64, 60, 57, 52]


def _arpeggio(pitches, step=0.3, tau=0.8, tail=1.0):
    """Plucked-string stand-ins: 30 harmonics at 1/k, the higher ones dying faster."""
    hop = int(step * SR)
    n = len(pitches) * hop + int(tail * SR)
    x = np.zeros(n)
    for i, p in enumerate(pitches):
        f0 = 440.0 * 2.0 ** ((p - 69) / 12.0)
        t = np.arange(n - i * hop) / SR
        x[i * hop :] += 0.1 * sum(np.sin(2 * np.pi * k * f0 * t) * np.exp(-t * k**0.5 / tau) / k
                                  for k in range(1, 31) if k * f0 < 18000.0)
    return np.vstack((x, x))


def _nearest(report, hz):
    bands = report["tone"]["bands_hz"]
    return report["tone"]["diff_db"][int(np.argmin(np.abs(np.log(np.array(bands) / hz))))]


def test_a_signal_matches_itself():
    x = _arpeggio(ARPEGGIO)
    rep = compare(x, x).report
    assert rep["alignment"]["lag_ms"] == 0 and abs(rep["alignment"]["gain_db"]) < 0.01
    assert rep["tone"]["rms_db"] < 0.01 and rep["spectrogram"]["mrstft"] < 1e-6
    assert rep["onsets"]["reference"] == len(ARPEGGIO) and rep["onsets"]["f1"] == 1.0
    assert rep["harmony"]["chroma_similarity"] > 0.999 and not rep["harmony"]["dips"]


def test_a_late_quiet_render_is_aligned_and_matched_in_loudness():
    x = _arpeggio(ARPEGGIO)
    late = 0.5 * np.pad(x, ((0, 0), (int(0.03 * SR), 0)))
    rep = compare(x, late).report
    assert abs(rep["alignment"]["lag_ms"] - 30.0) < 3.0
    assert abs(rep["alignment"]["gain_db"] - 6.02) < 0.1
    assert rep["tone"]["rms_db"] < 0.3 and rep["onsets"]["f1"] == 1.0


def test_a_brighter_render_shows_in_the_tone_curve():
    x = _arpeggio(ARPEGGIO)
    # zero phase, so the boost adds up: +6 dB well above 3 kHz, nothing well below
    bright = x + sosfiltfilt(butter(4, 3000.0, "highpass", fs=SR, output="sos"), x)
    rep = compare(x, bright).report
    assert _nearest(rep, 6000.0) - _nearest(rep, 300.0) == pytest.approx(6.0, abs=1.0)
    assert rep["tone"]["tilt_db_per_octave"] > 0.5


def test_a_wrong_note_shows_as_a_chroma_dip_where_it_is():
    wrong = list(ARPEGGIO)
    wrong[4] = 65  # E4 played as F4, at 2.0-2.5 s
    rep = compare(_arpeggio(ARPEGGIO, step=0.5, tau=0.3), _arpeggio(wrong, step=0.5, tau=0.3)).report
    dips = rep["harmony"]["dips"]
    assert dips and all(start > 1.8 and end < 2.9 for start, end, _ in dips)
    assert rep["onsets"]["f1"] == 1.0  # the timing is still right


def test_a_longer_sustain_shows_as_a_slower_decay():
    rep = compare(_arpeggio(ARPEGGIO, step=0.6, tau=0.3), _arpeggio(ARPEGGIO, step=0.6, tau=0.9)).report
    decay = rep["envelope"]["decay"]
    assert decay["segments"] >= 5
    assert decay["render_db_per_s"] > decay["reference_db_per_s"] + 5.0
    assert decay["tail_render_db_per_s"] > decay["tail_reference_db_per_s"]


def test_pick_clicks_show_in_the_attacks_not_in_the_whole():
    x = _arpeggio(ARPEGGIO, step=0.4)
    rng = np.random.default_rng(0)
    clicks = np.zeros(x.shape[1])
    for i in range(len(ARPEGGIO)):
        start = int(i * 0.4 * SR)
        clicks[start : start + int(0.003 * SR)] = 0.3 * rng.uniform(-1.0, 1.0, int(0.003 * SR))
    # where the strings have partials too (1.5-7 kHz), so the clicks hardly change the whole
    clicked = x + sosfiltfilt(butter(4, [1500.0, 7000.0], "bandpass", fs=SR, output="sos"), clicks)
    rep = compare(x, clicked).report
    assert rep["attack"]["notes"] >= len(ARPEGGIO) - 1
    assert rep["attack"]["level_db"] > 0.3
    assert rep["attack"]["tilt_db_per_octave"] > rep["tone"]["tilt_db_per_octave"] + 0.5
    assert rep["attack"]["rms_db"] > 3.0 * rep["tone"]["rms_db"]


def test_tremolo_shows_as_flutter():
    x = _arpeggio(ARPEGGIO, step=0.4, tau=1.5)
    shimmer = x * (1.0 + 0.3 * np.sin(2 * np.pi * 6.0 * np.arange(x.shape[1]) / SR))
    spec = compare(x, shimmer).report["spectrogram"]
    assert spec["render_flutter_db"] > spec["reference_flutter_db"] + 0.25


def test_listening_files_play_both_and_toggle_between_them():
    x = _arpeggio(ARPEGGIO)
    comp = compare(x, x + sosfilt(butter(2, 2000.0, "highpass", fs=SR, output="sos"), x))
    ab, toggle = listening(comp, block=1.0, gap=0.5)
    n = comp.reference.shape[1]
    assert ab.shape == (2, 2 * n + SR // 2) and toggle.shape == (2, n)
    scale = ab[0, 1000] / comp.reference[0, 1000]
    mid_ref, mid_render = SR // 2, SR + SR // 2  # the middles of the first two blocks
    assert toggle[0, mid_ref] == pytest.approx(scale * comp.reference[0, mid_ref])
    assert toggle[0, mid_render] == pytest.approx(scale * comp.render[0, mid_render])


def test_load_reads_a_compressed_file_and_cuts_the_fragment(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from notes.export.media import to_mp3

    x = _arpeggio(ARPEGGIO)
    write_wav(tmp_path / "x.wav", x, SR)
    to_mp3(tmp_path / "x.wav", tmp_path / "x.mp3")
    cut = load(tmp_path / "x.mp3", start=0.5, end=2.0)
    assert abs(cut.shape[1] - int(1.5 * SR)) < 100
    rep = compare(x[:, int(0.5 * SR) : int(2.0 * SR)], cut).report
    assert abs(rep["alignment"]["lag_ms"]) < 2.0 and rep["tone"]["rms_db"] < 0.5


def test_seconds_parse_minutes():
    assert _seconds("75.5") == 75.5 and _seconds("1:15.5") == 75.5 and _seconds("0:01:15.5") == 75.5
    assert _seconds(None) is None


def test_cli_writes_the_report_the_figure_and_the_listening_files(tmp_path):
    pytest.importorskip("matplotlib")
    x = _arpeggio(ARPEGGIO)
    write_wav(tmp_path / "original.wav", x, SR)
    write_wav(tmp_path / "attempt.wav", 0.7 * np.pad(x, ((0, 0), (2000, 0))), SR)
    assert main(["compare", str(tmp_path / "original.wav"), str(tmp_path / "attempt.wav"), "--end", "2.5"]) == 0
    report = json.loads((tmp_path / "attempt.vs.original.json").read_text(encoding="utf-8"))
    assert report["seconds"] == 2.5 and abs(report["alignment"]["lag_ms"] - 1000 * 2000 / SR) < 3.0
    for suffix in ("png", "ab.wav", "toggle.wav"):
        assert (tmp_path / f"attempt.vs.original.{suffix}").exists()
