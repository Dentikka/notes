"""Command line: render a composition script to audio plus its IR, MIDI and pictures.

    notes render song.py -o ../renders/0927-first-groove/
    notes info song.py

A script defines ``song = Song(...)``. When the output directory differs from the script's,
the script is copied there, so every render folder holds the exact source it came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import runpy
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from notes import __version__
from notes.engine.mixer import render_score
from notes.export.midi import write_midi
from notes.lang.song import Song

__all__ = ["load_song", "main"]


def load_song(script: Path) -> Song:
    """Run a composition script and return its `song` (or its only Song)."""
    namespace = runpy.run_path(str(script), run_name="__notes__")
    song = namespace.get("song")
    if isinstance(song, Song):
        return song
    songs = [v for v in namespace.values() if isinstance(v, Song)]
    if len(songs) == 1:
        return songs[0]
    raise SystemExit(f"{script}: define exactly one Song, ideally as `song = Song(...)`")


def _git_revision() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "src"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return rev + ("+dirty" if dirty else "")


def _render(args: argparse.Namespace) -> int:
    script = Path(args.script).resolve()
    song = load_song(script)
    out_dir = Path(args.out).resolve() if args.out else script.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or script.stem
    score = song.compile()
    result = render_score(score)
    written = [result.write(out_dir / f"{stem}.wav", bits=args.bits)]
    if not args.no_ir:
        score.to_json(out_dir / f"{stem}.ir.json")
        written.append(out_dir / f"{stem}.ir.json")
    if not args.no_midi:
        written.append(write_midi(score, out_dir / f"{stem}.mid"))
    if not args.no_roll or args.spectrogram:
        from notes.viz import pianoroll, spectrogram

        if not args.no_roll:
            written.append(pianoroll(score, out_dir / f"{stem}.png"))
        if args.spectrogram:
            written.append(spectrogram(result.audio, result.sample_rate, out_dir / f"{stem}.spec.png", score.title))
    if out_dir != script.parent:
        written.append(Path(shutil.copy2(script, out_dir / script.name)))
    meta = {
        "script": str(script),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "notes_version": __version__,
        "notes_git": _git_revision(),
        "rendered_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": score.summary(),
        **result.stats,
    }
    meta_path = out_dir / f"{stem}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(meta_path)

    s = result.stats
    print(score.summary())
    print(
        f"rendered {s['duration_s']} s in {s['render_s']} s: {s['lufs']} LUFS, peak {s['peak_dbfs']} dBFS, "
        f"master gain {s['master_gain_db']:+} dB, limiter {s['limiter_db']} dB"
    )
    for name, t in s["tracks"].items():
        print(f"  {name:>10}: {t['lufs_in_mix']:>7} LUFS, peak {t['peak_dbfs_in_mix']:>6} dBFS, {t['notes']} notes")
    for path in written:
        print(f"wrote {path}")
    return 0


def _info(args: argparse.Namespace) -> int:
    print(load_song(Path(args.script).resolve()).describe())
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="notes", description="Music as code: render composition scripts.")
    parser.add_argument("--version", action="version", version=f"notes {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="render a script to WAV plus IR, MIDI and a piano roll")
    render.add_argument("script")
    render.add_argument("-o", "--out", help="output directory (default: next to the script)")
    render.add_argument("--name", help="base name of the outputs (default: the script's name)")
    render.add_argument("--bits", type=int, choices=(16, 24), default=16)
    render.add_argument("--no-ir", action="store_true", help="skip the IR JSON")
    render.add_argument("--no-midi", action="store_true", help="skip the MIDI file")
    render.add_argument("--no-roll", action="store_true", help="skip the piano-roll PNG")
    render.add_argument("--spectrogram", action="store_true", help="also draw a spectrogram PNG")
    render.set_defaults(func=_render)
    info = sub.add_parser("info", help="describe a script's song without rendering")
    info.add_argument("script")
    info.set_defaults(func=_info)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
