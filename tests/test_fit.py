import json

import numpy as np
import pytest

from notes import BassGuitar, Song, Track, melody
from notes.cli import main
from notes.ears.fit import fit_reference
from notes.engine.audio import write_wav


def _bass(settings):
    return Song(Track("bass", BassGuitar(**settings))(melody("A1 D2", step=0.5)),
                bpm=120, loudness=None, seed=17).render().audio


def test_recovers_known_instrument_setting_and_preserves_inputs():
    initial = {"tone": 0.1}
    calls = []

    def render(settings):
        calls.append(dict(settings))
        return _bass(settings)

    result = fit_reference(_bass({"tone": 0.9}), render, initial, {"tone": [0.1, 0.5, 0.9]}, max_lag=0)
    assert result.settings == {"tone": 0.9}
    assert result.loss < 1e-5
    assert result.history[0]["loss"] > result.loss + 0.1
    assert len(calls) == len(result.history) == 3
    assert initial == {"tone": 0.1}
    assert np.isfinite(result.comparison.render).all()


def test_budget_and_trial_callback():
    trials = []
    result = fit_reference(_bass({"tone": 0.9}), _bass, {"tone": 0.1},
                           {"tone": [0.5, 0.9]}, max_evaluations=1, on_trial=trials.append)
    assert len(trials) == 1 and result.settings == {"tone": 0.1}
    json.dumps(trials, allow_nan=False)


@pytest.mark.parametrize("options", [{"passes": 0}, {"max_evaluations": 0}, {"max_lag": -1}])
def test_rejects_invalid_search_controls(options):
    with pytest.raises(ValueError):
        fit_reference(np.zeros(44100), _bass, {"tone": 0.1}, {"tone": [0.9]}, **options)


def test_cli_fits_and_saves_reviewable_outputs(tmp_path):
    script = tmp_path / "model.py"
    script.write_text('from notes import BassGuitar, Song, Track, melody\n'
                      'initial = {"tone": 0.1}\ngrid = {"tone": [0.9]}\n'
                      'def build_song(s):\n'
                      '    return Song(Track("bass", BassGuitar(**s))(melody("A1 D2", step=0.5)), '
                      'bpm=120, loudness=None, seed=17)\n', encoding="utf-8")
    write_wav(tmp_path / "ref.wav", _bass({"tone": 0.9}), 44100, bits=24)
    out = tmp_path / "fit"
    assert main(["fit", str(tmp_path / "ref.wav"), str(script), "-o", str(out), "--max-lag", "0"]) == 0
    report = json.loads((out / "fitted.json").read_text(encoding="utf-8"))
    assert report["settings"] == {"tone": 0.9}
    for name in ("trials.jsonl", "fitted.wav", "fitted.ab.wav", "fitted.toggle.wav", "fitted.png", "model.py"):
        assert (out / name).exists()
