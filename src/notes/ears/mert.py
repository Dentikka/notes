"""Optional music-specific reference embeddings; not a perceptual-realism verdict.

The pinned 95M MERT configuration uses the native HuBERT architecture. No model
repository Python is executed. Weights and configuration are checked before load.
"""
from __future__ import annotations

import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from notes.ears.clap import windows

__all__ = ["MERT_MODEL", "MERT_RATE", "MERT_REVISION", "MertEar", "to_mert_rate"]

MERT_MODEL = "m-a-p/MERT-v1-95M"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
MERT_RATE = 24000
_HASHES = {
    "config.json": "ea2627c4c7825cd66f3c944b6b966331604c35928174e0100cd4a82829424e32",
    "preprocessor_config.json": "cc5a5e4a5d3b1a758a5ed984b2eaa15bb0522d811d44a9eed82bfca4baa0dc8f",
    "pytorch_model.bin": "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd",
}


def to_mert_rate(audio: np.ndarray, sr: int) -> np.ndarray:
    """Finite mono audio at 24 kHz; reject silence, DC and clips below 100 ms."""
    x = np.asarray(audio, dtype=float)
    if (not x.size or x.ndim not in (1, 2) or not np.isfinite(sr) or sr < 1000 or int(sr) != sr
            or x.shape[-1] < sr*.1 or not np.isfinite(x).all()):
        raise ValueError("MERT needs finite mono/channels-first audio of at least 100 ms and a valid rate")
    mono = np.atleast_2d(x).mean(axis=0)
    if float(np.std(mono)) < 1e-8:
        raise ValueError("MERT cannot compare silent or DC-only audio")
    divisor = gcd(MERT_RATE, int(sr))
    if sr != MERT_RATE:
        mono = resample_poly(mono, MERT_RATE//divisor, int(sr)//divisor)
    return mono.astype(np.float32)


class MertEar:
    """Audio-only MERT-95M embeddings, last layer with mean pooling.

    Input is processed in overlapping windows of at most 10 seconds to bound
    attention memory. Only the pinned configuration is supported. ``local_path``
    selects an existing checkpoint; otherwise fetch only its three data files.
    Loading is explicit, CPU by default, and does not change global thread settings.
    """

    model_name = MERT_MODEL
    revision = MERT_REVISION

    def __init__(self, local_path: str | Path | None = None, *, device: str = "cpu") -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor
        except ImportError as error:
            raise ImportError("MERT needs the optional notes[ears] dependencies") from error
        path = Path(local_path) if local_path is not None else Path(snapshot_download(
            MERT_MODEL, revision=MERT_REVISION, allow_patterns=list(_HASHES)))
        for filename, expected in _HASHES.items():
            with (path/filename).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"MERT checkpoint does not match the pinned revision: {filename}")
        config = json.loads((path/"config.json").read_text(encoding="utf-8"))
        for key in ("auto_map", "architectures", "model_type"):
            config.pop(key, None)
        model, info = HubertModel.from_pretrained(
            path, config=HubertConfig(**config), local_files_only=True, use_safetensors=False,
            attn_implementation="eager", output_loading_info=True)
        if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
            raise ValueError("MERT weights do not match the native HuBERT architecture")
        self._model = model.to(device).eval()
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
        self._torch = torch
        self.device = device

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Unit vector for reference comparison; no text tags or realism label."""
        mono = to_mert_rate(audio, sr)
        vectors = []
        for window in windows(mono, sr=MERT_RATE):
            if float(np.std(window)) < 1e-8:
                continue
            inputs = self._processor(window, sampling_rate=MERT_RATE, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                vector = self._model(**inputs).last_hidden_state.mean(dim=1)[0].cpu().numpy()
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector).all() or norm < 1e-12:
                raise ValueError("MERT produced an invalid embedding")
            vectors.append(vector/norm)
        if not vectors:
            raise ValueError("MERT input contains no audible windows")
        result = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(result))
        if norm < 1e-12:
            raise ValueError("MERT window embeddings cancel")
        return result/norm
