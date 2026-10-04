"""Pictures of a piece: a per-track piano roll with section markers, and a spectrogram.

Both are PNG files meant for people and for multimodal agents, which cannot listen but can
look. matplotlib is an optional dependency (``pip install notes[viz]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import stft

from notes.ir.gm import GM_DRUMS
from notes.ir.score import Score

__all__ = ["comparison", "pianoroll", "spectrogram"]

_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("Plots need matplotlib: pip install 'notes[viz]'") from err
    return plt


def pianoroll(score: Score, path: str | Path) -> Path:
    """One lane per track (pitch for instruments, sound for drums), x in bars, sections marked."""
    plt = _pyplot()
    from matplotlib.colors import to_rgba
    from matplotlib.ticker import FixedLocator

    tracks = [t for t in score.tracks if score.events_of(t.name)]
    if not tracks:
        raise ValueError("Nothing to draw: the score has no notes")
    bpb = float(score.beats_per_bar)
    total = float(score.length) / bpb
    lanes, heights = [], []
    for spec in tracks:
        evs = score.events_of(spec.name)
        if not any(e.pitched for e in evs):
            names = sorted({str(e.param("sound")) for e in evs}, key=lambda s: GM_DRUMS.get(s, 0))
            lanes.append(names)
            heights.append(max(1.0, 0.3 * len(names)))
        else:
            pitches = [e.pitch for e in evs if e.pitched]
            lanes.append((min(pitches), max(pitches)))
            heights.append(max(1.0, min(3.5, 0.1 * (max(pitches) - min(pitches) + 6))))
    fig, axes = plt.subplots(
        len(tracks), 1, sharex=True, squeeze=False, figsize=(14, sum(heights) + 1.2),
        gridspec_kw={"height_ratios": heights},
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    step = 1 if total <= 16 else 4
    for i, (ax, spec, lane) in enumerate(zip(axes[:, 0], tracks, lanes, strict=True)):
        evs = score.events_of(spec.name)
        if isinstance(lane, list):
            row = {name: k for k, name in enumerate(lane)}
            ys = [row[str(e.param("sound"))] for e in evs]
            ax.set_yticks(range(len(lane)), lane, fontsize=7)
            ax.set_ylim(-0.7, len(lane) - 0.3)
        else:
            ys = [e.pitch for e in evs]
            lo, hi = lane
            cs = [p for p in range(int(lo) - 12, int(hi) + 13) if p % 12 == 0 and lo - 1 <= p <= hi + 1]
            ax.set_yticks(cs, [f"C{p // 12 - 1}" for p in cs], fontsize=7)
            ax.set_ylim(lo - 1.5, hi + 1.5)
        ax.barh(
            ys,
            [max(float(e.dur) / bpb, 0.012) for e in evs],
            left=[float(e.time) / bpb + 1 for e in evs],
            height=0.8,
            color=[to_rgba(colors[i % len(colors)], 0.3 + 0.7 * e.vel) for e in evs],
            linewidth=0,
        )
        ax.set_ylabel(spec.name, rotation=0, ha="right", va="center", fontsize=9)
        ax.xaxis.set_major_locator(FixedLocator(list(range(1, int(total) + 2, step))))
        ax.grid(axis="x", color="0.88", lw=0.6)
        for m in score.markers:
            ax.axvline(float(m.start) / bpb + 1, color="0.35", lw=0.8, ls="--")
    top = axes[0, 0]
    for m in score.markers:
        top.text(float(m.start) / bpb + 1.05, 1.02, m.name, transform=top.get_xaxis_transform(), fontsize=8)
    axes[-1, 0].set_xlim(1, total + 1)
    axes[-1, 0].set_xlabel("bar")
    fig.suptitle(score.title or "", x=0.01, ha="left", fontsize=11)
    out = Path(path)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def comparison(comp: Any, path: str | Path, title: str | None = None) -> Path:
    """A render against its reference (`notes.ears.compare.Comparison`): the long-term
    spectra and their difference, both log-mel spectrograms and the difference, the band
    envelopes with the onsets, and the chroma similarity over time."""
    plt = _pyplot()
    rep, tr = comp.report, comp.traces
    tone = rep["tone"]
    t = tr["t"]
    mel_a, mel_b = tr["mel"]
    fig = plt.figure(figsize=(14, 17))
    grid = fig.add_gridspec(6, 2, height_ratios=[1.3, 1, 1, 1, 1.1, 0.7], hspace=0.45, top=0.95)
    ax = fig.add_subplot(grid[0, 0])
    ax.semilogx(tone["bands_hz"], tone["reference_db"], "o-", ms=3, color="0.2", label="reference")
    ax.semilogx(tone["bands_hz"], tone["render_db"], "o--", ms=3, color="tab:red", label="render")
    ax.set_ylim(max(tone["reference_db"]) - 60, max(tone["reference_db"] + tone["render_db"]) + 5)
    ax.set(xlabel="Hz", ylabel="dB", title="long-term spectrum, third octaves")
    ax.legend(fontsize=8)
    ax.grid(color="0.9")
    ax = fig.add_subplot(grid[0, 1])
    pairs = [(f, d) for f, d in zip(tone["bands_hz"], tone["diff_db"], strict=True) if d is not None]
    ax.axhspan(-2, 2, color="0.92")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.semilogx([f for f, _ in pairs], [d for _, d in pairs], "o-", ms=3, color="tab:purple", label="whole")
    attack = rep.get("attack") or {}
    if attack:
        hits = [(f, d) for f, d in zip(tone["bands_hz"], attack["diff_db"], strict=True) if d is not None]
        ax.semilogx([f for f, _ in hits], [d for _, d in hits], "o--", ms=3, color="tab:orange",
                    label=f"attacks, first 20 ms ({attack['level_db']:+.1f} dB vs whole)")
        ax.legend(fontsize=8)
    ax.set(xlabel="Hz", ylabel="dB", title=f"render − reference: tilt {tone['tilt_db_per_octave']:+.2f} dB/oct, "
           f"rms {tone['rms_db']:.1f} dB")
    ax.grid(color="0.9")
    extent = (0.0, float(t[-1]) if len(t) else 1.0, 0.0, float(mel_a.shape[0]))
    top = float(max(mel_a.max(), mel_b.max()))
    for row, (name, mel) in enumerate((("reference", mel_a), ("render", mel_b)), start=1):
        ax = fig.add_subplot(grid[row, :])
        ax.imshow(mel, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=tr["floor"], vmax=top)
        ax.set_ylabel(f"{name}\nmel band")
    ax = fig.add_subplot(grid[3, :])
    image = ax.imshow(mel_b - mel_a, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=-20, vmax=20)
    ax.set_ylabel("render − ref\nmel band")
    fig.colorbar(image, ax=ax, pad=0.01, label="dB")
    ax = fig.add_subplot(grid[4, :])
    colors = {"low": "tab:blue", "mid": "tab:green", "high": "tab:orange"}
    for name, (ea, eb) in tr["envelopes"].items():
        ax.plot(t, ea, color=colors[name], lw=1, label=f"{name}, reference")
        ax.plot(t, eb, color=colors[name], lw=1, ls="--", label=f"{name}, render")
    onsets_ref, onsets_est = tr["onsets"]
    ax.vlines(onsets_ref, 0.97, 1.0, transform=ax.get_xaxis_transform(), color="0.2", lw=0.8)
    ax.vlines(onsets_est, 0.93, 0.96, transform=ax.get_xaxis_transform(), color="tab:red", lw=0.8)
    ax.set(ylabel="dB", title="band envelopes (ticks: onsets, reference above, render below)")
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    ax.set_xlim(extent[0], extent[1])
    ax = fig.add_subplot(grid[5, :])
    ax.plot(tr["chroma_t"], tr["chroma_similarity"], color="tab:purple", lw=1)
    ax.axhline(0.75, color="0.6", lw=0.8, ls=":")
    ax.set(ylim=(0, 1.02), xlabel="seconds", ylabel="chroma sim.",
           title=f"harmony: mean {rep['harmony']['chroma_similarity']}")
    ax.set_xlim(extent[0], extent[1])
    fig.suptitle(title or "render vs reference", x=0.01, ha="left", fontsize=12)
    out = Path(path)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def spectrogram(audio: np.ndarray, sr: int, path: str | Path, title: str | None = None) -> Path:
    """Log-frequency spectrogram of the mono sum, 90 dB of dynamic range."""
    plt = _pyplot()
    mono = np.atleast_2d(audio).mean(axis=0)
    f, t, z = stft(mono, fs=sr, nperseg=2048, noverlap=1024)
    db = 20.0 * np.log10(np.abs(z[1:]) + 1e-9)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.pcolormesh(t, f[1:], db, shading="auto", cmap="magma", vmin=db.max() - 90.0, vmax=db.max(), rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(30, sr / 2)
    ax.set_xlabel("seconds")
    ax.set_ylabel("Hz")
    ax.set_title(title or "")
    out = Path(path)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
