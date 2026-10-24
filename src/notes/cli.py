"""Command line: render a composition script to audio plus its IR, MIDI and pictures.

    notes render song.py -o ../renders/0927-first-groove/ [--listen --prompt "..."]
    notes listen song.wav [--prompt "..."] [--reference other.wav]
    notes info song.py
    notes export song.wav [--mp3] [--mp4] [--subtitle "..."]
    notes compare original.flac attempt.wav --start 0:12 --end 0:31

A script defines ``song = Song(...)``. When the output directory differs from the script's,
the script is copied there, so every render folder holds the exact source it came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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
    result = render_score(score, workers=args.workers)
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


def _export(args: argparse.Namespace) -> int:
    from notes.export.media import cover, to_mp3, to_mp4

    wav = Path(args.audio).resolve()
    ir = wav.with_suffix(".ir.json")
    score = Score.from_json(ir.read_text(encoding="utf-8")) if ir.exists() else None
    title = args.title or (score.title if score else None) or wav.stem
    tags = {"title": title, "artist": args.artist, "album": args.album or "", "comment": "rendered by notes"}
    tags = {k: v for k, v in tags.items() if v}
    audio, sr = read_wav(wav)
    seconds = audio.shape[-1] / sr
    do_mp3, do_mp4 = args.mp3 or not args.mp4, args.mp4 or not args.mp3
    if do_mp3:
        print(f"wrote {to_mp3(wav, wav.with_suffix('.mp3'), bitrate=args.bitrate, tags=tags)}")
    if do_mp4:
        image = Path(args.image) if args.image else None
        if image is None:
            if score is None:
                raise SystemExit(f"{wav.name}: no {ir.name} to draw a cover from; pass --image")
            image = cover(score, wav.with_name(wav.stem + ".cover.png"), seconds=seconds, subtitle=args.subtitle)
            print(f"wrote {image}")
        print(f"wrote {to_mp4(wav, image, wav.with_suffix('.mp4'), seconds, tags=tags)}")
    return 0


def _seconds(text: str | None) -> float | None:
    """'75.5', '1:15.5' or '0:01:15.5' as seconds."""
    if text is None:
        return None
    value = 0.0
    for part in text.split(":"):
        value = 60.0 * value + float(part)
    return value


def _compare(args: argparse.Namespace) -> int:
    from notes.ears.compare import ANALYSIS_RATE, compare, format_comparison, listening, load
    from notes.engine.audio import write_wav
    from notes.viz import comparison

    reference, render = Path(args.reference).resolve(), Path(args.render).resolve()
    start = _seconds(args.start) or 0.0
    comp = compare(load(reference, start=start, end=_seconds(args.end)), load(render), max_lag=args.max_lag)
    report = {"reference": str(reference), "reference_start_s": start, "render": str(render), **comp.report}
    if args.clap:
        from notes.ears import Ear, cosine

        ear = Ear(args.model)
        report["clap"] = {"cosine": round(cosine(ear.embed_audio(comp.reference, ANALYSIS_RATE),
                                                 ear.embed_audio(comp.render, ANALYSIS_RATE)), 4)}
    out = Path(args.out).resolve() if args.out else render.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{render.stem}.vs.{reference.stem}"
    print(format_comparison(report))
    written = [_write_json(stem.with_name(stem.name + ".json"), report),
               comparison(comp, stem.with_name(stem.name + ".png"), f"{render.name} vs {reference.name}")]
    if not args.no_ab:
        ab, toggle = listening(comp, block=args.block)
        written += [write_wav(stem.with_name(stem.name + ".ab.wav"), ab, ANALYSIS_RATE),
                    write_wav(stem.with_name(stem.name + ".toggle.wav"), toggle, ANALYSIS_RATE)]
    for path in written:
        print(f"wrote {path}")
    return 0


def _fit(args: argparse.Namespace) -> int:
    from notes.ears.compare import ANALYSIS_RATE, listening, load
    from notes.ears.fit import fit_reference
    from notes.engine.audio import write_wav
    from notes.viz import comparison

    script = Path(args.script).resolve()
    namespace = runpy.run_path(str(script), run_name="__notes__")
    build = namespace.get("build_song")
    if not callable(build):
        raise ValueError("fit script must define build_song(settings) -> Song, initial and grid")
    reference = load(args.reference, start=_seconds(args.start) or 0.0, end=_seconds(args.end))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if script != out / script.name:
        shutil.copy2(script, out / script.name)

    def render(settings):
        song = build(settings)
        if not isinstance(song, Song) or song.sample_rate != ANALYSIS_RATE:
            raise ValueError("build_song must return a Song at 44100 Hz")
        return song.render(workers=args.workers).audio

    with (out / "trials.jsonl").open("w", encoding="utf-8") as log:
        def record(trial):
            log.write(json.dumps(trial, allow_nan=False) + "\n")
            log.flush()
            print(f"trial {trial['evaluation']}: {trial['loss']:.4f} {trial['settings']}", flush=True)

        result = fit_reference(reference, render, namespace["initial"], namespace["grid"],
                               passes=args.passes, max_lag=args.max_lag,
                               max_evaluations=args.max_evaluations, on_trial=record)
    _write_json(out / "fitted.json", {"settings": result.settings, "loss": result.loss,
                "initial_loss": result.history[0]["loss"], "evaluations": len(result.history),
                "reference": str(Path(args.reference).resolve()), "start": _seconds(args.start) or 0.0,
                "end": _seconds(args.end), "git_revision": _git_revision(),
                "max_lag": args.max_lag, "passes": args.passes, "max_evaluations": args.max_evaluations,
                "audio": "aligned and loudness-matched to reference", "report": result.comparison.report})
    write_wav(out / "fitted.wav", result.comparison.render, ANALYSIS_RATE, bits=24)
    ab, toggle = listening(result.comparison)
    write_wav(out / "fitted.ab.wav", ab, ANALYSIS_RATE)
    write_wav(out / "fitted.toggle.wav", toggle, ANALYSIS_RATE)
    comparison(result.comparison, out / "fitted.png", "Fitted instrument vs reference")
    print(f"best loss {result.loss:.4f}; wrote {out}")
    return 0


def _transcribe(args: argparse.Namespace) -> int:
    import numpy as np

    from notes import AcousticGuitar, BassGuitar, ElectricGuitar
    from notes.ears.compare import ANALYSIS_RATE, _onset_strength, _onsets, compare, listening, load
    from notes.ears.polyphonic import instrument_templates, notes_from_templates, template_track
    from notes.ears.transcribe import (
        TrackedNote,
        melody_track,
        notes_from_track,
        onsets_from_track,
        pitch_track,
        tracked_music,
    )
    from notes.engine.audio import write_wav
    from notes.lang.song import Track
    from notes.viz import comparison, pianoroll, spectrogram

    if args.high < args.low or not 0 <= args.low <= args.high <= 127 or args.a4 <= 0:
        raise ValueError("need 0 <= low <= high <= 127 and positive tuning")
    start, end = _seconds(args.start) or 0.0, _seconds(args.end)
    ref = load(args.reference, start=start, end=end)
    mono = {"mid": ref.mean(axis=0), "side": (ref[0] - ref[-1]) / 2,
            "left": ref[0], "right": ref[-1]}[args.channel]
    ref = np.stack((mono, mono))
    params = json.loads(Path(args.settings).read_text(encoding="utf-8")) if args.settings else {}
    constructors = {"bass": BassGuitar, "electric": ElectricGuitar,
                    "acoustic": AcousticGuitar, "twelve": lambda **kw: AcousticGuitar(**(kw | {"twelve": True}))}
    instrument = constructors[args.instrument](**params)
    onsets = _onsets(*_onset_strength(mono))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    residual = None
    if args.method == "basic-pitch":
        from notes.ears.neural import basic_pitch_track, notes_from_onsets, notes_from_probabilities

        if not args.model_path:
            raise ValueError("basic-pitch requires --model-path to a local official nmp.onnx")
        track = basic_pitch_track(mono, ANALYSIS_RATE, args.model_path, a4=args.a4)
        threshold = args.onset_threshold if args.onset_threshold is not None else (
            0.7 if args.note_decoder == "onsets" else 0.5)
        options = {"low": args.low, "high": args.high, "min_dur": args.min_duration,
                   "onset_threshold": threshold, "frame_threshold": args.frame_threshold}
        if args.note_decoder == "onsets":
            notes = notes_from_onsets(track, **options, release_threshold=args.release_threshold,
                                      harmonic_ratio=args.harmonic_ratio)
        else:
            notes = notes_from_probabilities(track, **options)
    elif args.method == "templates":
        templates = instrument_templates(instrument, range(args.low, args.high + 1), a4=args.a4)
        track = template_track(mono, ANALYSIS_RATE, templates, n_fft=8192, sparsity=args.sparsity)
        notes = notes_from_templates(track, onsets, min_dur=args.min_duration,
                                     max_polyphony=args.max_polyphony, relative_db=args.relative_db)
        residual = float(np.mean(track.residual))
    else:
        fmin, fmax = args.a4 * 2 ** ((args.low - 69) / 12), args.a4 * 2 ** ((args.high - 69) / 12)
        track = (pitch_track(mono, ANALYSIS_RATE, fmin, fmax) if args.method == "mono"
                 else melody_track(mono, ANALYSIS_RATE, fmin, fmax, pitch_level=args.melody_onsets == "pitch"))
        if args.method == "melody" and args.melody_onsets == "pitch":
            onsets = onsets_from_track(track)
        notes = notes_from_track(track, onsets, a4=args.a4, min_dur=args.min_duration,
                                 sustain_unvoiced=args.method == "mono", jump=0.7 if args.method == "mono" else 3.0)
    reference_check = None
    if args.reference_notes:
        from notes.ears.note_comparison import compare_notes

        gold = json.loads(Path(args.reference_notes).read_text(encoding="utf-8"))
        gold = gold["notes"] if isinstance(gold, dict) else gold
        gold_notes = [TrackedNote(n["start"], n["end"], n["pitch"], 0, np.array([]), np.array([])) for n in gold]
        reference_check = compare_notes(gold_notes, notes)
    model_hash = (hashlib.sha256(Path(args.model_path).read_bytes()).hexdigest()
                  if args.method == "basic-pitch" else None)
    _write_json(out / "notes.json", {"reference": str(Path(args.reference).resolve()),
                "status": "reference-scored" if reference_check is not None else "unverified",
                "reference_check": reference_check, "model_sha256": model_hash,
                "melody_onsets": args.melody_onsets,
                "note_decoder": args.note_decoder,
                "onset_threshold": threshold if args.method == "basic-pitch" else args.onset_threshold,
                "frame_threshold": args.frame_threshold, "release_threshold": args.release_threshold,
                "harmonic_ratio": args.harmonic_ratio,
                "contours_enabled": not args.no_contours and args.method in ("melody", "basic-pitch"),
                "start": start, "end": end, "method": args.method, "channel": args.channel,
                "a4": args.a4, "instrument": args.instrument, "settings": params,
                "low": args.low, "high": args.high, "min_duration": args.min_duration,
                "sparsity": args.sparsity, "relative_db": args.relative_db, "max_polyphony": args.max_polyphony,
                "git_revision": _git_revision(), "residual": residual,
                "notes": [{"start": n.start, "end": n.end, "pitch": n.pitch, "level_db": n.level_db,
                           "contour_t": n.contour_t.tolist(), "contour": n.contour.tolist()} for n in notes]})
    if not notes:
        print(f"No notes found; wrote {out / 'notes.json'}")
        return 1
    music = tracked_music(notes, duration=len(mono) / ANALYSIS_RATE,
                          contours=not args.no_contours and args.method in ("melody", "basic-pitch"))
    song = Song(Track("transcribed", instrument)(music), bpm=60, a4=args.a4, loudness=None, seed=17)
    score = song.compile()
    score.to_json(out / "transcribed.ir.json")
    write_midi(score, out / "transcribed.mid")
    pianoroll(score, out / "transcribed.png")
    audio = song.render().audio
    write_wav(out / "transcribed.wav", audio, ANALYSIS_RATE, bits=24)
    spectrogram(audio, ANALYSIS_RATE, out / "transcribed.spec.png", "Transcribed stem")
    comp = compare(ref, audio, max_lag=0.05)
    _write_json(out / "comparison.json", comp.report)
    comparison(comp, out / "comparison.png", "Transcribed stem vs reference")
    ab, toggle = listening(comp)
    write_wav(out / "transcribed.ab.wav", ab, ANALYSIS_RATE)
    write_wav(out / "transcribed.toggle.wav", toggle, ANALYSIS_RATE)
    status = "reference-scored" if reference_check is not None else "UNVERIFIED (audio metrics are not note accuracy)"
    print(f"{len(notes)} candidate notes; {status}; wrote {out}")
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
    render.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="processes rendering tracks in parallel (default: up to 8)")
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
    comp = sub.add_parser("compare", help="measure a render against a reference recording (tone, envelope, "
                                          "timing, harmony, space) and write A/B listening files")
    comp.add_argument("reference", help="the recording to match: WAV, or MP3/FLAC/M4A... (needs notes[media])")
    comp.add_argument("render", help="our attempt, a WAV")
    comp.add_argument("--start", help="where the fragment starts in the reference, seconds or m:ss")
    comp.add_argument("--end", help="where it ends")
    comp.add_argument("--max-lag", type=float, default=1.0, help="largest shift (s) tried when aligning")
    comp.add_argument("--block", type=float, default=2.0, help="seconds between switches in the toggle file")
    comp.add_argument("--no-ab", action="store_true", help="skip the listening files")
    comp.add_argument("--clap", action="store_true", help="also the cosine of CLAP embeddings (needs notes[ears])")
    comp.add_argument("-o", "--out", help="output directory (default: the render's)")
    comp.set_defaults(func=_compare)
    for p in (render, listen, comp):
        p.add_argument("--model", default=DEFAULT_MODEL, help="CLAP checkpoint on Hugging Face")
    for p in (render, listen):
        p.add_argument("--prompt", action="append", help="text to score the audio against (repeatable)")
    info = sub.add_parser("info", help="describe a script's song without rendering")
    info.add_argument("script")
    info.set_defaults(func=_info)
    fit = sub.add_parser("fit", help="fit a fixed part's instrument parameters to an isolated reference")
    fit.add_argument("reference")
    fit.add_argument("script", help="Python script defining build_song(settings), initial and grid")
    fit.add_argument("-o", "--out", required=True)
    fit.add_argument("--start", help="reference fragment start in seconds or m:ss")
    fit.add_argument("--end", help="reference fragment end")
    fit.add_argument("--passes", type=int, default=2)
    fit.add_argument("--max-evaluations", type=int, default=100)
    fit.add_argument("--max-lag", type=float, default=0.05)
    fit.add_argument("--workers", type=int, default=1)
    fit.set_defaults(func=_fit)
    trans = sub.add_parser("transcribe", help="extract notes from an isolated stem and render a reconstruction")
    trans.add_argument("reference")
    trans.add_argument("-o", "--out", required=True)
    trans.add_argument("--method", choices=("mono", "melody", "templates", "basic-pitch"), default="mono")
    trans.add_argument("--model-path", help="local Basic Pitch nmp.onnx (requires notes[transcription])")
    trans.add_argument("--note-decoder", choices=("frames", "onsets"), default="frames",
                       help="Basic Pitch decoder: legacy frame runs or pitch-specific attacks")
    trans.add_argument("--onset-threshold", type=float, help="model onset threshold (frames: 0.5, onsets: 0.7)")
    trans.add_argument("--frame-threshold", type=float, default=0.3)
    trans.add_argument("--release-threshold", type=float,
                       help="extend accepted onset-decoded notes at a lower threshold")
    trans.add_argument("--harmonic-ratio", type=float, help="optional pruning of weaker coincident model harmonics")
    trans.add_argument("--no-contours", action="store_true", help="render fixed pitches rather than estimated bends")
    trans.add_argument("--reference-notes", help="explicit reference note JSON for onset/pitch/offset scoring")
    trans.add_argument("--melody-onsets", choices=("mix", "pitch"), default="mix",
                       help="melody re-picks: legacy mix attacks, or experimental pitch-energy attacks")
    trans.add_argument("--instrument", choices=("bass", "electric", "acoustic", "twelve"), default="bass")
    trans.add_argument("--settings", help="JSON object of instrument constructor parameters")
    trans.add_argument("--start")
    trans.add_argument("--end")
    trans.add_argument("--channel", choices=("mid", "side", "left", "right"), default="mid")
    trans.add_argument("--a4", type=float, default=440)
    trans.add_argument("--low", type=int, default=28, help="lowest candidate MIDI pitch")
    trans.add_argument("--high", type=int, default=76, help="highest candidate MIDI pitch")
    trans.add_argument("--min-duration", type=float, default=0.08)
    trans.add_argument("--sparsity", type=float, default=0.03)
    trans.add_argument("--relative-db", type=float, default=18)
    trans.add_argument("--max-polyphony", type=int)
    trans.set_defaults(func=_transcribe)
    export = sub.add_parser("export", help="MP3 and MP4 (still cover + audio) of a render (needs notes[media])")
    export.add_argument("audio", help="a rendered WAV; title and cover come from the IR JSON and piano roll beside it")
    export.add_argument("--mp3", action="store_true", help="only the MP3 (default: both)")
    export.add_argument("--mp4", action="store_true", help="only the MP4 (default: both)")
    export.add_argument("--bitrate", default="320k", help="MP3 bitrate (default 320k)")
    export.add_argument("--title", help="title tag and cover title (default: the song's)")
    export.add_argument("--subtitle", help="a line under the title on the cover, e.g. the style")
    export.add_argument("--artist", default="notes", help="artist tag")
    export.add_argument("--album", help="album tag")
    export.add_argument("--image", help="use this picture for the video instead of drawing a cover")
    export.set_defaults(func=_export)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
