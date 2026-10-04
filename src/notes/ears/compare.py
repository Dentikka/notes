"""Compare a render with a reference recording: what differs, where, and by how much.

The reference is usually a real recording and the render our attempt at it. The render is
aligned to the reference in time (cross-correlation of onset envelopes) and matched to it
in loudness (BS.1770); then both are measured on the axes one would adjust:

- tone: the long-term spectrum in third-octave bands; render minus reference in dB, its
  mean, RMS and tilt (dB per octave) over the bands the reference fills;
- attack: the same curve for the first ~20 ms of every note, and how much the render's
  attacks stand out from its whole sound against the reference's (pick, hardness);
- time-frequency: log-mel spectrogram distance, and the multi-resolution STFT distance
  (spectral convergence plus log-magnitude L1 at three FFT sizes, as in Parallel WaveGAN);
  the flutter of each, the frame-to-frame change away from note starts (chorus, vibrato);
- envelope: loudness over time in a low, mid and high band; how fast the sound decays
  between onsets and in the final tail, in dB per second;
- timing: onsets of both, matched within 50 ms: F-measure and timing errors;
- harmony: chroma similarity per frame; dips mark wrong or missing notes;
- space: stereo width (side over mid energy) per band.

The figures are relative: compare versions of a render against one reference with them,
and let the ear have the last word.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import fftconvolve, resample_poly, stft, welch

from notes.engine.audio import read_wav
from notes.engine.loudness import lufs

__all__ = ["ANALYSIS_RATE", "Comparison", "compare", "format_comparison", "listening", "load"]

#: Both signals are analysed at this rate (the renders' own).
ANALYSIS_RATE = 44100
#: Broad bands for envelopes and stereo width (Hz).
_BANDS = {"low": (20.0, 250.0), "mid": (250.0, 2500.0), "high": (2500.0, 20000.0)}
#: Third-octave band centres, 31.5 Hz to 16 kHz.
_THIRDS = 1000.0 * 2.0 ** (np.arange(-15, 13) / 3.0)
#: Bands quieter than the reference's loudest by more than this stay out of the tone summary.
_TONE_RANGE_DB = 50.0
#: Range of spectrograms (dB below the reference's maximum) and of band envelopes.
_FLOOR_DB, _ENVELOPE_RANGE_DB = 70.0, 60.0
_ONSET_FFT, _ONSET_HOP = 2048, 256
_MEL_FFT, _MEL_HOP, _MELS = 2048, 512, 96
_CHROMA_FFT, _CHROMA_HOP = 8192, 2048
_ATTACK_FFT = 1024
#: Onsets of the two signals within this distance (s) count as the same note.
_TOLERANCE = 0.05
#: Chroma similarity below this marks a span where the notes probably differ.
_CHROMA_DIP = 0.75


@dataclass(frozen=True)
class Comparison:
    """The numbers (`report`, JSON-ready), the aligned and loudness-matched signals at
    `ANALYSIS_RATE`, and the curves behind the numbers (`traces`, for plots)."""

    report: dict[str, Any]
    reference: np.ndarray
    render: np.ndarray
    traces: dict[str, Any]


def load(path: str | Path, *, start: float = 0.0, end: float | None = None, sr: int = ANALYSIS_RATE) -> np.ndarray:
    """A ``(channels, samples)`` float signal at `sr`, from `start` to `end` seconds: PCM WAV
    directly, anything else (MP3, FLAC, M4A, float WAV...) through ffmpeg."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            x, rate = read_wav(path)
        except (ValueError, EOFError, wave.Error):
            pass
        else:
            x = _resample(x, rate, sr)
            return x[:, int(round(start * sr)) : None if end is None else int(round(end * sr))]
    from notes.export.media import decode

    return decode(path, sr, start=start, seconds=None if end is None else end - start)


def _resample(x: np.ndarray, rate: int, sr: int) -> np.ndarray:
    if rate == sr:
        return x
    g = gcd(rate, sr)
    return resample_poly(x, sr // g, rate // g, axis=-1)


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return np.vstack((x[0], x[0])) if x.shape[0] == 1 else x[:2]


def _db(p: np.ndarray | float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, 1e-20))


def _spec(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude STFT of a mono signal (Hann): frequencies, frame times, |X|."""
    f, t, z = stft(x, fs=ANALYSIS_RATE, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=True)
    return f, t, np.abs(z)


def _mel_filters(freqs: np.ndarray, n: int = _MELS, fmin: float = 40.0, fmax: float = 16000.0) -> np.ndarray:
    mel = 2595.0 * np.log10(1.0 + np.array([fmin, fmax]) / 700.0)
    edges = 700.0 * (10.0 ** (np.linspace(mel[0], mel[1], n + 2) / 2595.0) - 1.0)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0.0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))


def _band_levels(freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Third-octave levels (dB) of a power spectrum."""
    return np.array([_db(power[(freqs >= c * 2.0 ** (-1 / 6)) & (freqs < c * 2.0 ** (1 / 6))].sum()) for c in _THIRDS])


def _third_octaves(x: np.ndarray) -> np.ndarray:
    """Long-term level (dB) in each third-octave band."""
    f, p = welch(x, fs=ANALYSIS_RATE, nperseg=min(8192, len(x)))
    return _band_levels(f, p * (f[1] - f[0]))


def _curve(ref: np.ndarray, est: np.ndarray) -> dict[str, Any]:
    """Render minus reference over the bands the reference fills: per band, mean, RMS, tilt
    (dB per octave) and the three largest gaps."""
    keep = (ref > ref.max() - _TONE_RANGE_DB) & (_THIRDS < 0.45 * ANALYSIS_RATE)
    diff = est - ref
    tilt = float(np.polyfit(np.log2(_THIRDS[keep] / 1000.0), diff[keep], 1)[0]) if keep.sum() >= 2 else 0.0
    order = np.argsort(-np.abs(np.where(keep, diff, 0.0)))[:3]
    return {
        "diff_db": [round(float(d), 2) if k else None for d, k in zip(diff, keep, strict=True)],
        "mean_db": round(float(diff[keep].mean()), 2),
        "rms_db": round(float(np.sqrt(np.mean(diff[keep] ** 2))), 2),
        "tilt_db_per_octave": round(tilt, 2),
        "largest": [[round(float(_THIRDS[i]), 1), round(float(diff[i]), 2)] for i in order if keep[i]],
    }


def _onset_strength(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SuperFlux (Böck and Widmer, DAFx 2013): the rise of the log mel spectrum over the
    maximum of three neighbouring bands two frames back, so the slow sweeps of vibrato and
    chorus do not count as new notes. Silence is put in front, so a note sounding from the
    first sample has an onset too; the curve is scaled to its 99th percentile."""
    f, t, mag = _spec(np.concatenate((np.zeros(_ONSET_FFT), x)), _ONSET_FFT, _ONSET_HOP)
    mel = _mel_filters(f, _MELS, 30.0, 16000.0) @ mag
    spec = np.log1p(1000.0 * mel / max(float(mel.max()), 1e-12))
    rise = np.maximum(0.0, spec[:, 2:] - maximum_filter1d(spec, size=3, axis=0)[:, :-2]).sum(axis=0)
    flux = np.concatenate((np.zeros(2), rise))
    return t - _ONSET_FFT / ANALYSIS_RATE, flux / max(float(np.percentile(flux, 99)), 1e-12)


def _onsets(t: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Böck's peak picking: the maximum within ±30 ms, at least 0.2 above the mean of the
    surrounding 170 ms, and 30 ms after the previous onset."""
    fps = ANALYSIS_RATE / _ONSET_HOP
    peak = maximum_filter1d(flux, size=int(0.06 * fps) | 1, mode="nearest")
    mean = uniform_filter1d(flux, size=int(0.17 * fps) | 1, mode="nearest")
    out: list[float] = []
    for k in np.flatnonzero((flux == peak) & (flux >= mean + 0.2)):
        if not out or t[k] - out[-1] > 0.03:
            out.append(float(t[k]))
    return np.array(out)


def _lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """How many frames `b` lags behind `a`: the cross-correlation peak, refined by a parabola."""
    n = 1 << (len(a) + len(b)).bit_length()
    xc = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = xc[lags % n]
    k = int(np.argmax(vals))
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = vals[k - 1 : k + 2]
        if (curve := y0 - 2.0 * y1 + y2) < 0:
            return float(lags[k] + 0.5 * (y0 - y2) / curve)
    return float(lags[k])


def _shift(x: np.ndarray, n: int, length: int) -> np.ndarray:
    """`x` moved `n` samples earlier (later if negative), cut or padded to `length`."""
    x = x[:, n:] if n >= 0 else np.pad(x, ((0, 0), (-n, 0)))
    return x[:, :length] if x.shape[1] >= length else np.pad(x, ((0, 0), (0, length - x.shape[1])))


def _match(ref: np.ndarray, est: np.ndarray) -> dict[str, Any]:
    """Pair onsets greedily, closest first, within the tolerance."""
    candidates = []
    for i, r in enumerate(ref):
        lo, hi = np.searchsorted(est, [r - _TOLERANCE, r + _TOLERANCE])
        candidates += [(abs(est[j] - r), i, j) for j in range(lo, hi)]
    seen_ref, seen_est, errors = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in seen_ref and j not in seen_est:
            seen_ref.add(i)
            seen_est.add(j)
            errors.append(est[j] - ref[i])
    hits = len(errors)
    precision, recall = hits / max(len(est), 1), hits / max(len(ref), 1)
    ms = 1000.0 * np.abs(errors) if hits else np.zeros(1)
    return {
        "reference": len(ref), "render": len(est), "matched": hits,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if hits else 0.0,
        "median_ms": round(float(np.median(ms)), 1), "p90_ms": round(float(np.percentile(ms, 90)), 1),
        "mean_signed_ms": round(1000.0 * float(np.mean(errors)), 1) if hits else 0.0,
    }


def _slope(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float | None:
    sel = (t >= lo) & (t <= hi)
    return float(np.polyfit(t[sel], y[sel], 1)[0]) if sel.sum() >= 5 else None


def _decays(t: np.ndarray, ref: np.ndarray, est: np.ndarray, onsets: np.ndarray) -> dict[str, Any]:
    """Decay rates (dB/s) of both envelopes between the reference's onsets (gaps of 0.25 s or
    more, medians), and over the tail after its last onset (up to 3 s, above its floor)."""
    pairs = []
    for a, b in zip(onsets[:-1], onsets[1:], strict=True):
        if b - a >= 0.25:
            r, e = _slope(t, ref, a + 0.05, b - 0.02), _slope(t, est, a + 0.05, b - 0.02)
            if r is not None and e is not None:
                pairs.append((r, e))
    out: dict[str, Any] = {"segments": len(pairs)}
    if pairs:
        out["reference_db_per_s"], out["render_db_per_s"] = (round(float(v), 1) for v in np.median(pairs, axis=0))
    if len(onsets):
        last = float(onsets[-1])
        floor = ref.min() + 10.0
        quiet = t[(t > last + 0.05) & (ref <= floor)]
        end = min(last + 3.0, float(quiet[0]) if len(quiet) else float(t[-1]))
        r, e = _slope(t, ref, last + 0.05, end), _slope(t, est, last + 0.05, end)
        if r is not None and e is not None:
            out["tail_reference_db_per_s"], out["tail_render_db_per_s"] = round(r, 1), round(e, 1)
    return out


def _mrstft(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral convergence plus log-magnitude L1, averaged over FFT sizes 512, 1024, 2048."""
    total = 0.0
    for n_fft in (512, 1024, 2048):
        _, _, ma = _spec(a, n_fft, n_fft // 4)
        _, _, mb = _spec(b, n_fft, n_fft // 4)
        eps = 1e-4 * float(ma.max())
        total += float(np.linalg.norm(ma - mb) / max(float(np.linalg.norm(ma)), 1e-12))
        total += float(np.mean(np.abs(np.log(ma + eps) - np.log(mb + eps))))
    return total / 3.0


def _chroma(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pitch-class profile per frame (60 Hz–5 kHz, C = 0) and the frame energies."""
    f, t, mag = _spec(x, _CHROMA_FFT, _CHROMA_HOP)
    sel = (f >= 60.0) & (f <= 5000.0)
    classes = np.mod(np.round(12.0 * np.log2(f[sel] / 440.0)).astype(int) + 9, 12)
    chroma = np.zeros((12, mag.shape[1]))
    np.add.at(chroma, classes, mag[sel])
    return t, chroma, (mag[sel] ** 2).sum(axis=0)


def _spans(t: np.ndarray, flag: np.ndarray, value: np.ndarray, min_frames: int = 2) -> list[list[float]]:
    """Runs of `flag` as [start, end, lowest value]."""
    out, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1]:
                j += 1
            if j - i + 1 >= min_frames:
                out.append([round(float(t[i]), 2), round(float(t[j]), 2), round(float(value[i : j + 1].min()), 2)])
            i = j + 1
        else:
            i += 1
    return out


def _width(x: np.ndarray) -> dict[str, float]:
    """Side over mid energy per band, dB (-60 for mono)."""
    f, mid = welch((x[0] + x[1]) / 2.0, fs=ANALYSIS_RATE, nperseg=min(4096, x.shape[1]))
    _, side = welch((x[0] - x[1]) / 2.0, fs=ANALYSIS_RATE, nperseg=min(4096, x.shape[1]))
    out = {}
    for name, (lo, hi) in _BANDS.items():
        sel = (f >= lo) & (f < hi)
        out[name] = round(max(-60.0, float(_db(side[sel].sum()) - _db(mid[sel].sum()))), 1)
    return out


def _crest(x: np.ndarray) -> float:
    return round(float(20.0 * np.log10(np.max(np.abs(x)) / max(float(np.sqrt(np.mean(x**2))), 1e-12))), 1)


def _attack(a: np.ndarray, b: np.ndarray, onsets: np.ndarray) -> dict[str, Any]:
    """The start of every reference note (a 23 ms Hann window centred 10 ms after the onset)
    against the same moments of the render: their tone curve, and how much more the render's
    attacks stand out from its sound as a whole (`level_db`; negative: softer attacks)."""
    n = _ATTACK_FFT
    starts = [s for s in (int(round((o + 0.01) * ANALYSIS_RATE)) - n // 2 for o in onsets) if 0 <= s <= len(a) - n]
    if not starts:
        return {}
    window = np.hanning(n)
    att_a = np.mean([np.abs(np.fft.rfft(a[s : s + n] * window)) ** 2 for s in starts], axis=0)
    att_b = np.mean([np.abs(np.fft.rfft(b[s : s + n] * window)) ** 2 for s in starts], axis=0)
    level = _db(att_b.sum() / max(float(att_a.sum()), 1e-30)) - _db(np.mean(b**2) / max(float(np.mean(a**2)), 1e-30))
    freqs = np.fft.rfftfreq(n, 1.0 / ANALYSIS_RATE)
    return {"notes": len(starts), "level_db": round(float(level), 2),
            **_curve(_band_levels(freqs, att_a), _band_levels(freqs, att_b))}


def _flutter(mel: np.ndarray, steady: np.ndarray) -> float:
    """Mean change (dB) of the mel spectrum from one frame to the next away from note starts:
    the shimmer of chorus, vibrato and tremolo, and the grain of noise."""
    return round(float(np.abs(np.diff(mel, axis=1))[steady].mean()), 2) if steady.any() else 0.0


def compare(reference: np.ndarray, render: np.ndarray, sr: int = ANALYSIS_RATE, *, max_lag: float = 1.0) -> Comparison:
    """Align `render` to `reference` (both ``(channels, samples)`` at `sr`; the render may be
    shifted by up to `max_lag` seconds either way), match its loudness, and measure both."""
    ref = _stereo(_resample(np.atleast_2d(reference), sr, ANALYSIS_RATE))
    est = _stereo(_resample(np.atleast_2d(render), sr, ANALYSIS_RATE))
    ref_lufs, est_lufs = lufs(ref, ANALYSIS_RATE), lufs(est, ANALYSIS_RATE)
    if not (np.isfinite(ref_lufs) and np.isfinite(est_lufs)):
        raise ValueError("Both signals need sound to be compared")
    t_on, flux_ref = _onset_strength(ref.mean(axis=0))
    t_est, flux_est = _onset_strength(est.mean(axis=0) * 10.0 ** ((ref_lufs - est_lufs) / 20.0))
    # The jump from nothing at the very start is the cut, not a note: keep it out of the alignment.
    flux_ref, flux_est = np.where(t_on < 0.05, 0.0, flux_ref), np.where(t_est < 0.05, 0.0, flux_est)
    shift = int(round(_lag(flux_ref, flux_est, int(max_lag * ANALYSIS_RATE / _ONSET_HOP)) * _ONSET_HOP))
    est = _shift(est, shift, ref.shape[1])
    gain = ref_lufs - lufs(est, ANALYSIS_RATE)
    est = est * 10.0 ** (gain / 20.0)
    a, b = ref.mean(axis=0), est.mean(axis=0)

    tone_ref, tone_est = _third_octaves(a), _third_octaves(b)

    f, t, mag_a = _spec(a, _MEL_FFT, _MEL_HOP)
    _, _, mag_b = _spec(b, _MEL_FFT, _MEL_HOP)
    pow_a, pow_b = mag_a**2, mag_b**2
    filters = _mel_filters(f)
    mel_a, mel_b = _db(filters @ pow_a), _db(filters @ pow_b)
    floor = float(mel_a.max()) - _FLOOR_DB
    mel_a, mel_b = np.maximum(mel_a, floor), np.maximum(mel_b, floor)
    lit = (mel_a > floor) | (mel_b > floor)

    bands, envelopes = {}, {}
    for name, (lo, hi) in _BANDS.items():
        sel = (f >= lo) & (f < hi)
        ea, eb = _db(pow_a[sel].sum(axis=0)), _db(pow_b[sel].sum(axis=0))
        low = float(ea.max()) - _ENVELOPE_RANGE_DB
        ea, eb = np.maximum(ea, low), np.maximum(eb, low)
        act = (ea > low) | (eb > low)
        corr = float(np.corrcoef(ea[act], eb[act])[0, 1]) if act.sum() > 2 and np.std(eb[act]) > 0 else 0.0
        bands[name] = {"l1_db": round(float(np.mean(np.abs(ea - eb)[act])), 2) if act.any() else 0.0,
                       "corr": round(corr, 3)}
        envelopes[name] = (ea, eb)
    env_a, env_b = _db(pow_a.sum(axis=0)), _db(pow_b.sum(axis=0))
    env_floor = float(env_a.max()) - _ENVELOPE_RANGE_DB
    env_a, env_b = np.maximum(env_a, env_floor), np.maximum(env_b, env_floor)

    onsets_ref, onsets_est = _onsets(*_onset_strength(a)), _onsets(*_onset_strength(b))
    attack = _attack(a, b, onsets_ref)
    near = np.zeros(len(t), dtype=bool)
    for onset in onsets_ref:
        near |= (t > onset - 0.01) & (t < onset + 0.08)
    steady = lit[:, 1:] & lit[:, :-1] & ~near[None, 1:]

    tc, chroma_a, energy_a = _chroma(a)
    _, chroma_b, _ = _chroma(b)
    den = np.linalg.norm(chroma_a, axis=0) * np.linalg.norm(chroma_b, axis=0)
    sim = np.where(den > 0, (chroma_a * chroma_b).sum(axis=0) / np.maximum(den, 1e-30), 0.0)
    sounding = energy_a > energy_a.max() * 1e-4

    report = {
        "seconds": round(ref.shape[1] / ANALYSIS_RATE, 3),
        "alignment": {"lag_ms": round(1000.0 * shift / ANALYSIS_RATE, 1), "gain_db": round(gain, 2)},
        "loudness": {"reference_lufs": round(ref_lufs, 2), "render_lufs": round(est_lufs, 2),
                     "reference_crest_db": _crest(ref), "render_crest_db": _crest(est)},
        "tone": {
            "bands_hz": [round(float(c), 1) for c in _THIRDS],
            "reference_db": [round(float(v), 2) for v in tone_ref],
            "render_db": [round(float(v), 2) for v in tone_est],
            **_curve(tone_ref, tone_est),
        },
        "attack": attack,
        "spectrogram": {"mel_l1_db": round(float(np.mean(np.abs(mel_a - mel_b)[lit])), 2) if lit.any() else 0.0,
                        "mrstft": round(_mrstft(a, b), 4),
                        "reference_flutter_db": _flutter(mel_a, steady), "render_flutter_db": _flutter(mel_b, steady)},
        "envelope": {**bands, "decay": _decays(t, env_a, env_b, onsets_ref)},
        "onsets": _match(onsets_ref, onsets_est),
        "harmony": {"chroma_similarity": round(float(sim[sounding].mean()), 3) if sounding.any() else 0.0,
                    "dips": _spans(tc, sounding & (sim < _CHROMA_DIP), sim)},
        "space": {"reference_width_db": _width(ref), "render_width_db": _width(est)},
    }
    traces = {"t": t, "mel": (mel_a, mel_b), "floor": floor, "envelopes": envelopes,
              "onsets": (onsets_ref, onsets_est), "chroma_t": tc, "chroma_similarity": sim}
    return Comparison(report, ref, est, traces)


def listening(comp: Comparison, *, block: float = 2.0, gap: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Two files for the ear, at the same loudness: the reference, a pause and the render; and
    both at once toggling every `block` seconds (reference first) at the same point of the
    music, with 10 ms crossfades."""
    ref, est = comp.reference, comp.render
    scale = 0.98 / max(float(np.max(np.abs(ref))), float(np.max(np.abs(est))), 1e-9)
    ref, est = ref * min(scale, 1.0), est * min(scale, 1.0)
    pause = np.zeros((2, int(gap * ANALYSIS_RATE)))
    n = ref.shape[1]
    which = (np.arange(n) // int(block * ANALYSIS_RATE)) % 2
    ramp = int(0.01 * ANALYSIS_RATE)
    window = np.hanning(2 * ramp + 1)
    mix = np.clip(fftconvolve(which.astype(float), window / window.sum(), mode="same"), 0.0, 1.0)
    return np.concatenate((ref, pause, est), axis=1), (1.0 - mix) * ref + mix * est


def format_comparison(report: dict[str, Any]) -> str:
    """A digest of `compare`'s report, one line per axis."""
    al, loud, tone = report["alignment"], report["loudness"], report["tone"]
    env, ons, harm = report["envelope"], report["onsets"], report["harmony"]
    moved = f"moved {abs(al['lag_ms']):g} ms {'earlier' if al['lag_ms'] >= 0 else 'later'}"
    brighter = "brighter" if tone["tilt_db_per_octave"] > 0 else "darker"
    largest = ", ".join(f"{d:+.1f} dB at {f:g} Hz" for f, d in tone["largest"])
    decay, spec, att = env["decay"], report["spectrogram"], report.get("attack") or {}
    lines = [
        f"render vs reference over {report['seconds']} s: render {moved}, scaled {al['gain_db']:+.1f} dB "
        f"to the reference's {loud['reference_lufs']} LUFS",
        f"tone      mean {tone['mean_db']:+.1f} dB, rms {tone['rms_db']:.1f} dB, "
        f"tilt {tone['tilt_db_per_octave']:+.2f} dB/oct (render {brighter}); largest {largest}",
        f"spectrum  log-mel L1 {spec['mel_l1_db']} dB, MR-STFT {spec['mrstft']}; flutter reference "
        f"{spec['reference_flutter_db']} dB, render {spec['render_flutter_db']} dB",
        "envelope  " + ", ".join(f"{k} L1 {env[k]['l1_db']} dB (r {env[k]['corr']})" for k in _BANDS),
    ]
    if att:
        harder = "harder" if att["level_db"] > 0 else "softer"
        lines.insert(2, f"attack    first 20 ms of {att['notes']} notes: {att['level_db']:+.1f} dB against the whole "
                        f"(render {harder}); tone mean {att['mean_db']:+.1f} dB, rms {att['rms_db']:.1f} dB, "
                        f"tilt {att['tilt_db_per_octave']:+.2f} dB/oct")
    if "reference_db_per_s" in decay:
        lines.append(f"decay     between onsets: reference {decay['reference_db_per_s']} dB/s, render "
                     f"{decay['render_db_per_s']} dB/s ({decay['segments']} gaps)")
    if "tail_reference_db_per_s" in decay:
        lines.append(f"tail      reference {decay['tail_reference_db_per_s']} dB/s, "
                     f"render {decay['tail_render_db_per_s']} dB/s")
    lines.append(f"onsets    F1 {ons['f1']} (reference {ons['reference']}, render {ons['render']}, matched "
                 f"{ons['matched']}); timing median {ons['median_ms']} ms, p90 {ons['p90_ms']} ms, "
                 f"render {ons['mean_signed_ms']:+g} ms on average")
    dips = ", ".join(f"{s:g}-{e:g} s ({v})" for s, e, v in harm["dips"][:8]) or "none"
    lines.append(f"harmony   chroma similarity {harm['chroma_similarity']}; dips: {dips}")
    rw, ew = report["space"]["reference_width_db"], report["space"]["render_width_db"]
    lines.append("space     side/mid " + ", ".join(f"{k} {rw[k]:g}/{ew[k]:g} dB" for k in _BANDS)
                 + " (reference/render)")
    lines.append(f"crest     reference {loud['reference_crest_db']} dB, render {loud['render_crest_db']} dB")
    if "clap" in report:
        lines.append(f"CLAP      cosine {report['clap']['cosine']}")
    return "\n".join(lines)
