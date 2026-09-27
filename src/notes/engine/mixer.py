"""The renderer: turns an IR `Score` into a stereo master.

Per track: every note becomes a mono voice, the voices are summed, the instrument's own
inserts run on the sum, the result is panned to stereo, gained, and passed through the
track's effects. The tracks are summed, the master is normalised to a loudness target and
peak-limited. Every voice gets its own seed derived from its content, so a render is
reproducible bit for bit and editing one phrase leaves the noise of the others unchanged.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from notes.engine.audio import write_wav
from notes.engine.base import Instrument, RenderContext, Voice
from notes.engine.dsp import db_to_gain, stable_seed
from notes.engine.loudness import limit, lufs, peak_db
from notes.engine.registry import build_effect, build_instrument
from notes.ir.score import NoteEvent, Score, TempoMap

__all__ = ["RenderResult", "render_score"]

_SILENCE = 1e-4  # about -80 dBFS: where the tail is trimmed


@dataclass(frozen=True)
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    stats: dict[str, Any]

    @property
    def duration(self) -> float:
        return self.audio.shape[1] / self.sample_rate

    def write(self, path: str | Path, bits: int = 16) -> Path:
        return write_wav(path, self.audio, self.sample_rate, bits, seed=int(self.stats.get("seed", 0)))


def _choke_cuts(groups: Mapping[int, str], events: Sequence[NoteEvent]) -> dict[int, Fraction]:
    """Index of each note that is cut short -> beat at which the next note of its group starts."""
    members: dict[str, list[int]] = {}
    for i, e in enumerate(events):
        group = groups.get(int(round(e.pitch)))
        if group is not None:
            members.setdefault(group, []).append(i)
    cuts: dict[int, Fraction] = {}
    for idx in members.values():
        for pos, i in enumerate(idx):
            later = next((events[j].time for j in idx[pos + 1 :] if events[j].time > events[i].time), None)
            if later is not None:
                cuts[i] = later
    return cuts


def _render_voices(
    instrument: Instrument,
    events: Sequence[NoteEvent],
    tempo: TempoMap,
    ctx: RenderContext,
    n: int,
    track: str,
    offset: int = 0,
) -> np.ndarray:
    """Sum of all voices into a buffer of `n` samples that starts `offset` samples into the song."""
    out = np.zeros(n)
    cuts = _choke_cuts(instrument.choke_groups(), events)
    seen: dict[tuple[Fraction, float], int] = {}
    for i, e in enumerate(events):
        start_s = tempo.seconds(e.time)
        dup = seen.get((e.time, e.pitch), 0)
        seen[(e.time, e.pitch)] = dup + 1
        rng = np.random.default_rng(stable_seed(ctx.seed, track, str(e.time), e.pitch, dup))
        voice = Voice(
            freq=ctx.a4 * 2.0 ** ((e.pitch - 69.0) / 12.0),
            pitch=e.pitch,
            dur=tempo.seconds(e.end) - start_s,
            vel=e.vel,
            params=dict(e.params),
        )
        wave = instrument.voice(voice, ctx, rng)
        if i in cuts:
            keep = ctx.samples(tempo.seconds(cuts[i]) - start_s)
            if keep < len(wave):
                wave = wave[:keep].copy()
                ramp = min(keep, ctx.samples(0.005))
                if ramp:
                    wave[keep - ramp :] *= np.linspace(1.0, 0.0, ramp)
        start = ctx.samples(start_s) - offset
        end = min(n, start + len(wave))
        if end > start:
            out[start:end] += wave[: end - start]
    return out


def _pan(x: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan for mono input, balance for stereo input."""
    if x.shape[0] == 1:
        theta = (pan + 1.0) * np.pi / 4.0
        return np.vstack([np.cos(theta) * x[0], np.sin(theta) * x[0]])
    return x * np.array([[min(1.0, 1.0 - pan)], [min(1.0, 1.0 + pan)]])


def _trim(x: np.ndarray, sr: int, keep: int) -> np.ndarray:
    """Cut the silent tail (never before `keep` samples) and fade the last 30 ms."""
    loud = np.flatnonzero(np.max(np.abs(x), axis=0) > _SILENCE)
    end = min(x.shape[1], max(keep, (int(loud[-1]) + 1) if loud.size else 0) + int(0.05 * sr))
    y = x[:, :end].copy()
    ramp = min(end, int(0.03 * sr))
    if ramp:
        y[:, end - ramp :] *= np.linspace(1.0, 0.0, ramp)
    return y


def _render_track(score: Score, name: str) -> tuple[int, np.ndarray, dict[str, Any]]:
    """Render one track over its active span: (start sample, stereo float32 stem, raw stats).

    The buffer runs from the first onset to the last gate plus the instrument's release and
    the effects' tails, so a track that plays only in the outro costs only the outro.
    """
    spec = score.track(name)
    tempo = score.tempo_map()
    ctx = RenderContext(sr=score.sample_rate, bpm=tempo.bpm_at(0), a4=score.a4, seed=score.seed)
    events = score.events_of(name)
    instrument = build_instrument(spec.instrument)
    inserts = instrument.inserts()
    effects = [build_effect(f) for f in spec.fx]
    tail = instrument.release_time() + sum(f.tail(ctx) for f in (*inserts, *effects)) + 0.05
    first = min((tempo.seconds(e.time) for e in events), default=0.0)
    last = max((tempo.seconds(e.end) for e in events), default=0.0)
    offset = ctx.samples(first)
    n = ctx.samples(last + tail) - offset + 1
    ctx = replace(ctx, offset=offset)
    x = _render_voices(instrument, events, tempo, ctx, n, name, offset)[None, :]
    for fx in inserts:
        x = fx.process(x, ctx)
    x = _pan(x, spec.pan) * db_to_gain(spec.gain_db)
    for fx in effects:
        x = fx.process(x, ctx)
    stats = {"notes": len(events), "lufs": lufs(x, score.sample_rate), "peak_dbfs": peak_db(x)}
    return offset, x.astype(np.float32), stats


def _import_modules(modules: Sequence[str]) -> None:
    """Worker initializer: import the modules that register the score's instruments and effects."""
    for module in modules:
        importlib.import_module(module)


def _registering_modules(score: Score) -> list[str] | None:
    """Modules a fresh process must import to rebuild the score's instruments and effects.

    None when some class lives in a script (``__main__`` or a runpy namespace) that a worker
    cannot import; such scores render in-process.
    """
    classes = {type(build_instrument(t.instrument)) for t in score.tracks}
    classes |= {type(build_effect(f)) for t in score.tracks for f in t.fx}
    modules = sorted({c.__module__ for c in classes})
    return None if any(m.startswith("__") for m in modules) else modules


def render_score(score: Score, workers: int = 1) -> RenderResult:
    """Render a score to a stereo master plus statistics (loudness, peaks, per-track levels).

    Tracks are independent until the master bus, so with `workers` > 1 they render in
    separate processes. Stems are summed in track order either way: the output is identical
    for any number of workers.
    """
    started = time.perf_counter()
    sr = score.sample_rate
    song_n = int(round(score.duration_seconds() * sr)) + 1
    names = [t.name for t in score.tracks]
    modules = _registering_modules(score) if workers > 1 and len(names) > 1 else None
    if modules is None:
        workers = 1
        results = (_render_track(score, name) for name in names)
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=min(workers, len(names)), initializer=_import_modules, initargs=(modules,)
        )
        results = pool.map(_render_track, [score] * len(names), names)
    mix = np.zeros((2, song_n))
    track_stats: dict[str, dict[str, Any]] = {}
    try:
        for name, (offset, stem, raw) in zip(names, results, strict=True):
            end = offset + stem.shape[1]
            if end > mix.shape[1]:
                mix = np.pad(mix, ((0, 0), (0, end - mix.shape[1])))
            mix[:, offset:end] += stem
            track_stats[name] = raw
    finally:
        if pool is not None:
            pool.shutdown()
    raw_lufs = lufs(mix, sr)
    target = score.master.get("loudness")
    gain_db = target - raw_lufs if target is not None and np.isfinite(raw_lufs) else 0.0
    mix, reduction = limit(mix * db_to_gain(gain_db), sr, float(score.master.get("ceiling", -1.0)))
    mix = _trim(mix, sr, song_n - 1)
    stats = {
        "duration_s": round(mix.shape[1] / sr, 3),
        "lufs": round(lufs(mix, sr), 2),
        "peak_dbfs": round(peak_db(mix), 2),
        "master_gain_db": round(gain_db, 2),
        "limiter_db": round(reduction, 2),
        "tracks": {
            name: {
                "notes": t["notes"],
                "lufs_in_mix": round(t["lufs"] + gain_db, 2),
                "peak_dbfs_in_mix": round(t["peak_dbfs"] + gain_db, 2),
            }
            for name, t in track_stats.items()
        },
        "render_s": round(time.perf_counter() - started, 2),
        "workers": workers,
        "seed": score.seed,
    }
    return RenderResult(mix, sr, stats)
