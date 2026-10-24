"""Fit a fixed part's instrument/effects to an isolated reference, using only numpy/scipy.

The caller supplies a deterministic ``render(settings)`` returning audio at ``sr``.
Notes, tuning, seed and timing must stay fixed. This searches a finite parameter grid;
it cannot recover a sound outside the renderer's capabilities or verify transcription.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from notes.ears.compare import ANALYSIS_RATE, Comparison, compare

Parameter = float | int | str | bool


def fit_loss(report: Mapping[str, Any]) -> float:
    """Time-dependent spectrum plus tone/attack/envelope/width errors; lower is better.

    Overall gain and a bounded global offset are nuisance parameters handled by compare.
    Chroma is deliberately excluded: broad-band noise can score highly on that axis.
    This is a search objective, not a perceptual score or a note-accuracy measure.
    """
    env = report["envelope"]
    space = report["space"]
    width = np.mean([abs(space["render_width_db"][b] - space["reference_width_db"][b])
                     for b in ("low", "mid", "high")])
    return float(report["spectrogram"]["mel_l1_db"] + report["tone"]["rms_db"]
                 + 0.5 * report["attack"].get("rms_db", 0.0)
                 + np.mean([env[b]["l1_db"] for b in ("low", "mid", "high")])
                 + 0.2 * width)


@dataclass(frozen=True)
class FitResult:
    settings: dict[str, Parameter]
    loss: float
    history: list[dict[str, Any]]
    comparison: Comparison


def fit_reference(
    reference: np.ndarray,
    render: Callable[[dict[str, Parameter]], np.ndarray],
    initial: Mapping[str, Parameter],
    grid: Mapping[str, Sequence[Parameter]],
    *,
    sr: int = ANALYSIS_RATE,
    passes: int = 2,
    max_lag: float = 0.05,
    max_evaluations: int = 100,
    on_trial: Callable[[dict[str, Any]], None] | None = None,
) -> FitResult:
    """Greedy coordinate search with caching, a strict render budget and early stopping.

    Keys are arbitrary flat names consumed by ``render`` (e.g. ``amp.drive``).
    Each coordinate's alternatives share the same starting settings. The initial
    candidate is always evaluated and retained on ties. The history contains every
    unique evaluation, including its full comparison report. ``on_trial`` can persist
    each trial as JSONL, so an interrupted search still leaves useful evidence.
    """
    if passes < 1 or max_evaluations < 1 or not np.isfinite(max_lag) or max_lag < 0:
        raise ValueError("passes/budget must be positive and max_lag finite and nonnegative")
    if not grid or any(k not in initial or not values for k, values in grid.items()):
        raise ValueError("grid must be nonempty, with nonempty values and keys present in initial")
    history: list[dict[str, Any]] = []
    cache: dict[tuple, tuple[float, dict[str, Any]]] = {}
    best_comparison: Comparison | None = None
    best_seen = float("inf")

    def evaluate(settings: dict[str, Parameter]) -> float:
        nonlocal best_comparison, best_seen
        key = tuple(sorted(settings.items()))
        if key in cache:
            return cache[key][0]
        comp = compare(reference, render(dict(settings)), sr=sr, max_lag=max_lag)
        loss = fit_loss(comp.report)
        if not np.isfinite(loss):
            raise ValueError("nonfinite fitting objective")
        trial = {"evaluation": len(history) + 1, "settings": dict(settings), "loss": loss,
                 "report": comp.report}
        history.append(trial)
        cache[key] = loss, comp.report
        if loss < best_seen:
            best_seen, best_comparison = loss, comp
        if on_trial is not None:
            on_trial(trial)
        return loss

    best = dict(initial)
    loss = evaluate(best)
    for _ in range(passes):
        previous = loss
        for name, values in grid.items():
            anchor = dict(best)
            for value in values:
                trial = {**anchor, name: value}
                if len(history) >= max_evaluations:
                    break
                candidate_loss = evaluate(trial)
                if candidate_loss < loss:
                    best, loss = trial, candidate_loss
        if loss == previous or len(history) >= max_evaluations:
            break
    assert best_comparison is not None
    return FitResult(best, loss, history, best_comparison)
