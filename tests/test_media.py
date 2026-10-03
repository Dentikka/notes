import subprocess

import numpy as np
import pytest

pytest.importorskip("imageio_ffmpeg")
pytest.importorskip("matplotlib")

from notes import E, ElectricGuitar, Song, Track, melody  # noqa: E402
from notes.export.media import cover, ffmpeg_exe, to_mp3, to_mp4  # noqa: E402


def _duration(path):
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = out.split("Duration: ")[1].split(",")[0].split(":")
    return 3600 * float(h) + 60 * float(m) + float(s)


def test_mp3_and_mp4_last_as_long_as_the_render(tmp_path):
    song = Song(Track("g", ElectricGuitar(drive=0.3))(melody("A4 C5 E5 A5", step=E)), bpm=120, title="Probe")
    result = song.render(tmp_path / "probe.wav")
    seconds = result.audio.shape[-1] / result.sample_rate
    mp3 = to_mp3(tmp_path / "probe.wav", tmp_path / "probe.mp3", tags={"title": "Probe"})
    image = cover(song.compile(), tmp_path / "probe.png", seconds=seconds, subtitle="a test")
    mp4 = to_mp4(tmp_path / "probe.wav", image, tmp_path / "probe.mp4", seconds)
    assert abs(_duration(mp3) - seconds) < 0.1
    assert abs(_duration(mp4) - seconds) < 0.1  # a looped still must not outrun the audio
    assert np.isclose(seconds, result.audio.shape[-1] / result.sample_rate)
