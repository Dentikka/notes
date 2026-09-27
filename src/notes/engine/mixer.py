"""The renderer: turns an IR `Score` into a stereo master.

Per track: every note becomes a mono voice, the voices are summed, the instrument's own
inserts run on the sum, the result is panned to stereo, gained, and passed through the
track's effects. The tracks are summed, the master is normalised to a loudness target and
peak-limited. Every voice gets its own seed derived from its content, so a render is
reproducible bit for bit and editing one phrase leaves the noise of the others unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    instrument: Instrument, events: Sequence[NoteEvent], tempo: TempoMap, ctx: RenderContext, n: int, track: str
) -> np.ndarray:
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
        start = ctx.samples(start_s)
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


def render_score(score: Score) -> RenderResult:
    """Render a score to a stereo master plus statistics (loudness, peaks, per-track levels)."""
    started = time.perf_counter()
    tempo = score.tempo_map()
    sr = score.sample_rate
    ctx = RenderContext(sr=sr, bpm=tempo.bpm_at(0), a4=score.a4, seed=score.seed)
    song_s = tempo.seconds(score.length)
    stems: dict[str, np.ndarray] = {}
    for spec in score.tracks:
        instrument = build_instrument(spec.instrument)
        inserts = instrument.inserts()
        effects = [build_effect(f) for f in spec.fx]
        tail = instrument.release_time() + sum(f.tail(ctx) for f in (*inserts, *effects)) + 0.05
        n = ctx.samples(song_s + tail) + 1
        x = _render_voices(instrument, score.events_of(spec.name), tempo, ctx, n, spec.name)[None, :]
        for fx in inserts:
            x = fx.process(x, ctx)
        x = _pan(x, spec.pan) * db_to_gain(spec.gain_db)
        for fx in effects:
            x = fx.process(x, ctx)
        stems[spec.name] = x
    n = max((s.shape[1] for s in stems.values()), default=ctx.samples(song_s) + 1)
    mix = np.zeros((2, n))
    for s in stems.values():
        mix[:, : s.shape[1]] += s
    raw = lufs(mix, sr)
    target = score.master.get("loudness")
    gain_db = target - raw if target is not None and np.isfinite(raw) else 0.0
    mix, reduction = limit(mix * db_to_gain(gain_db), sr, float(score.master.get("ceiling", -1.0)))
    mix = _trim(mix, sr, ctx.samples(song_s))
    stats = {
        "duration_s": round(mix.shape[1] / sr, 3),
        "lufs": round(lufs(mix, sr), 2),
        "peak_dbfs": round(peak_db(mix), 2),
        "master_gain_db": round(gain_db, 2),
        "limiter_db": round(reduction, 2),
        "tracks": {
            name: {
                "notes": len(score.events_of(name)),
                "lufs_in_mix": round(lufs(s, sr) + gain_db, 2),
                "peak_dbfs_in_mix": round(peak_db(s) + gain_db, 2),
            }
            for name, s in stems.items()
        },
        "render_s": round(time.perf_counter() - started, 2),
        "seed": score.seed,
    }
    return RenderResult(mix, sr, stats)
