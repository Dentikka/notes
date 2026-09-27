"""Command line: render a composition script to audio plus its IR, MIDI and pictures.

    notes render song.py -o ../renders/0927-first-groove/ [--listen --prompt "..."]
    notes listen song.wav [--prompt "..."] [--reference other.wav]
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
from notes.ears.clap import DEFAULT_MODEL
from notes.engine.audio import read_wav
from notes.engine.mixer import render_score
from notes.export.midi import write_midi
from notes.ir.score import Score
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
    heard = None
    if args.listen:
        from notes.ears import Ear

        heard = Ear(args.model).listen(result.audio, result.sample_rate, score, prompts=args.prompt or ())
        written.append(_write_json(out_dir / f"{stem}.listen.json", heard))
    meta = {
        "script": str(script),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "notes_version": __version__,
        "notes_git": _git_revision(),
        "rendered_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": score.summary(),
        **result.stats,
    }
    written.append(_write_json(out_dir / f"{stem}.meta.json", meta))

    s = result.stats
    print(score.summary())
    print(
        f"rendered {s['duration_s']} s in {s['render_s']} s: {s['lufs']} LUFS, peak {s['peak_dbfs']} dBFS, "
        f"master gain {s['master_gain_db']:+} dB, limiter {s['limiter_db']} dB"
    )
    for name, t in s["tracks"].items():
        print(f"  {name:>10}: {t['lufs_in_mix']:>7} LUFS, peak {t['peak_dbfs_in_mix']:>6} dBFS, {t['notes']} notes")
    if heard is not None:
        from notes.ears import format_report

        print(format_report(heard))
    for path in written:
        print(f"wrote {path}")
    return 0


def _write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _listen(args: argparse.Namespace) -> int:
    from notes.ears import Ear, cosine, format_report

    wav = Path(args.audio).resolve()
    audio, sr = read_wav(wav)
    ir = Path(args.ir).resolve() if args.ir else wav.with_name(f"{wav.stem}.ir.json")
    score = Score.from_json(ir) if ir.exists() else None
    ear = Ear(args.model)
    report = ear.listen(audio, sr, score, prompts=args.prompt or (), top=args.top)
    if args.reference:
        ref, ref_sr = read_wav(args.reference)
        similarity = cosine(ear.embed_audio(audio, sr), ear.embed_audio(ref, ref_sr))
        report["reference"] = {"path": str(Path(args.reference).resolve()), "similarity": round(similarity, 4)}
    print(format_report(report))
    print(f"wrote {_write_json(wav.with_name(f'{wav.stem}.listen.json'), report)}")
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
    render.add_argument("--listen", action="store_true", help="also tag the render with CLAP (needs notes[ears])")
    render.set_defaults(func=_render)
    listen = sub.add_parser("listen", help="CLAP tags per section and similarity to text prompts")
    listen.add_argument("audio", help="a WAV file; sections come from the IR JSON next to it")
    listen.add_argument("--ir", help="IR JSON with the section markers (default: <audio stem>.ir.json)")
    listen.add_argument("--reference", help="another WAV to compare with (cosine of CLAP embeddings)")
    listen.add_argument("--top", type=int, default=3, help="labels to show per category")
    listen.set_defaults(func=_listen)
    for p in (render, listen):
        p.add_argument("--prompt", action="append", help="text to score the audio against (repeatable)")
        p.add_argument("--model", default=DEFAULT_MODEL, help="CLAP checkpoint on Hugging Face")
    info = sub.add_parser("info", help="describe a script's song without rendering")
    info.add_argument("script")
    info.set_defaults(func=_info)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
