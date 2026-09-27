"""WAV input and output with the standard library (16-bit dithered or 24-bit PCM)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = ["read_wav", "write_wav"]


def write_wav(path: str | Path, audio: np.ndarray, sr: int, bits: int = 16, seed: int = 0) -> Path:
    """Write a ``(channels, samples)`` float signal in [-1, 1]; 16-bit output gets TPDF dither."""
    frames = np.clip(np.atleast_2d(audio), -1.0, 1.0).T
    if bits == 16:
        rng = np.random.default_rng(seed)
        dither = rng.random(frames.shape) - rng.random(frames.shape)
        data = np.clip(np.round(frames * 32767.0 + dither), -32768, 32767).astype("<i2").tobytes()
    elif bits == 24:
        ints = np.clip(np.round(frames * 8388607.0), -8388608, 8388607).astype("<i4")
        data = np.ascontiguousarray(ints).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        raise ValueError("bits must be 16 or 24")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(frames.shape[1])
        w.setsampwidth(bits // 8)
        w.setframerate(sr)
        w.writeframes(data)
    return out


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read 16- or 24-bit PCM into a ``(channels, samples)`` float array."""
    with wave.open(str(path), "rb") as w:
        channels, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        x = ints.astype(float) / 8388608.0
    else:
        raise ValueError(f"Unsupported sample width: {width} bytes")
    return x.reshape(-1, channels).T, sr
