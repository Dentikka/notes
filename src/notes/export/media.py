"""Share a render: MP3, and MP4 video (a still cover over the audio), through ffmpeg; and
read any audio file ffmpeg can (a reference recording to compare a render with).

ffmpeg comes from the optional ``imageio-ffmpeg`` package (``pip install "notes[media]"``)
or, failing that, from the PATH. The cover is drawn with matplotlib: the title, a line of
facts (key, tempo, meter, length) and every track's notes as a lane across the song.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from notes.ir.score import Score

__all__ = ["cover", "decode", "ffmpeg_exe", "to_mp3", "to_mp4"]

logger = logging.getLogger(__name__)


def ffmpeg_exe() -> str:
    """Path of an ffmpeg executable: imageio-ffmpeg's bundled one, else the PATH's."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError('ffmpeg not found: pip install "notes[media]" (imageio-ffmpeg) or put ffmpeg on the PATH')


def _metadata(tags: dict[str, str] | None) -> list[str]:
    args: list[str] = []
    for k, v in (tags or {}).items():
        args += ["-metadata", f"{k}={v}"]
    return args


def _run(args: list[str]) -> None:
    logger.debug("ffmpeg %s", " ".join(args))
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def decode(
    path: str | Path, sr: int = 44100, *, start: float = 0.0, seconds: float | None = None, channels: int = 2
) -> np.ndarray:
    """Any audio file ffmpeg reads (MP3, FLAC, M4A, float WAV...) as a ``(channels, samples)``
    float array at `sr`, from `start` for `seconds` (to the end if None)."""
    args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if seconds is not None:
        args += ["-t", f"{seconds:.3f}"]
    args += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg could not read {path}: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(float).reshape(-1, channels).T


def to_mp3(wav: str | Path, out: str | Path, *, bitrate: str = "320k", tags: dict[str, str] | None = None) -> Path:
    """Encode a WAV as MP3 (LAME, constant `bitrate`) with ID3 `tags` (title, artist, album...)."""
    out = Path(out)
    _run(["-i", str(wav), "-codec:a", "libmp3lame", "-b:a", bitrate, *_metadata(tags), str(out)])
    return out


def to_mp4(
    wav: str | Path,
    image: str | Path,
    out: str | Path,
    seconds: float,
    *,
    audio_bitrate: str = "256k",
    tags: dict[str, str] | None = None,
) -> Path:
    """A video of a still `image` for `seconds` (the audio's length): H.264 (tuned for a
    still picture, yuv420p for every player) and AAC, the index up front for streaming.
    The length is given explicitly: with a looped picture ``-shortest`` overshoots by
    tens of seconds, or cuts the audio to the last frame at a low frame rate."""
    out = Path(out)
    _run([
        "-loop", "1", "-framerate", "2", "-t", f"{seconds:.3f}", "-i", str(image),
        "-i", str(wav), "-t", f"{seconds:.3f}",
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a", "aac", "-b:a", audio_bitrate, "-movflags", "+faststart",
        *_metadata(tags), str(out),
    ])
    return out


_LANE_COLOURS = ("#c9a96e", "#6fa8c9", "#d9776a", "#8fbf7f", "#b48fd1", "#e0c060", "#7fc9c0", "#d58fb0")


def cover(score: Score, out: str | Path, *, seconds: float, subtitle: str | None = None) -> Path:
    """A 1920x1080 cover: the title, a line of facts, and below, one lane per track with its
    notes drawn across the song (pitch within the lane; drums by sound)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bpm = score.tempo[0][1]
    length = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
    facts = [score.key, f"{bpm:g} BPM", f"{score.meter[0]}/{score.meter[1]}", length]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#101418")
    fig.text(0.05, 0.9, score.title or Path(out).stem, color="#f2efe8", fontsize=60, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.05, 0.79, subtitle, color="#c9a96e", fontsize=28, va="top")
    fig.text(0.05, 0.72 if subtitle else 0.79, " · ".join(f for f in facts if f), color="#8fa3b0", fontsize=24,
             va="top")
    ax = fig.add_axes((0.11, 0.06, 0.84, 0.56), facecolor="#101418")
    tempo = score.tempo_map()
    names = [t.name for t in score.tracks]
    for lane, name in enumerate(names):
        events = [e for e in score.events if e.track == name]
        if not events:
            continue
        pitches = [e.pitch for e in events]
        lo, hi = min(pitches), max(pitches)
        span = max(hi - lo, 1.0)
        top = len(names) - lane
        colour = _LANE_COLOURS[lane % len(_LANE_COLOURS)]
        for e in events:
            t0 = tempo.seconds(e.time)
            t1 = tempo.seconds(e.time + e.dur)
            y = top - 0.9 + 0.8 * (e.pitch - lo) / span
            alpha = min(1.0, 0.35 + 0.6 * e.vel)
            ax.add_patch(plt.Rectangle((t0, y), max(t1 - t0, 0.02), 0.06, color=colour, alpha=alpha, linewidth=0))
        ax.text(-0.01 * seconds, top - 0.5, name, color="#8fa3b0", fontsize=14, ha="right", va="center")
    for m in score.markers:
        x = tempo.seconds(m.start)
        ax.axvline(x, color="#2a333b", linewidth=1)
        ax.text(x + 0.005 * seconds, len(names) + 0.15, m.name, color="#5d6d78", fontsize=12, va="bottom")
    ax.set_xlim(0, seconds)
    ax.set_ylim(0, len(names) + 0.6)
    ax.set_axis_off()
    out = Path(out)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
