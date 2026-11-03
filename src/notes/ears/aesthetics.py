"""Optional Audiobox aesthetic diagnostics, not an instrument-realism verdict.

The pinned official Python backend is fetched explicitly and hash-checked before
execution. No extra packages or checkpoints are loaded when this module imports.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from notes.engine.loudness import lufs

__all__ = ["AestheticsEar", "fetch_aesthetics", "to_aesthetics_rate"]

_SOURCE = "2618e9d451b456e9328b39495b5e6234678aa550"
_MODEL = "9b1dd8e5df9af7216e836a98974fe3b82c56ded6"
_RATE = 16000
_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_SOURCE_FILES = {
    "LICENSE": "9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94",
    "src/audiobox_aesthetics/__init__.py": _EMPTY,
    "src/audiobox_aesthetics/model/__init__.py": _EMPTY,
    "src/audiobox_aesthetics/model/aes.py": "2f339d5d2526ccee45f1daf7722d2b9d36bd9c99212134be12439057d1196238",
    "src/audiobox_aesthetics/model/utils.py": "6af4d2acb03b9dad5ee34698f2ce96d423d4adb6bbdc6f3c591015c4da22ff3b",
    "src/audiobox_aesthetics/model/wavlm.py": "ed010d90278d5a8414049041a0c8ff60826fbd53045f32f0551c66dee8cac29f",
}
_MODEL_FILES = {
    "config.json": "0b8eabc5ced92cefed116a3aca8f1e59d2a33dc4d376b9e86dfb9c072e1d280d",
    "model.safetensors": "a5a3c2412649cc2384ec525ffd5180ce6c4778f43bed6108e0a1303de04d014e",
}


def _cache(path: str | Path | None) -> Path:
    return Path(path) if path is not None else Path.home()/".cache/notes-audiobox-aesthetics"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_aesthetics(local_path: str | Path | None = None) -> Path:
    """Explicitly fetch the pinned official source/license and safe tensor weights.

    The cache is outside the repository by default. Existing verified files are
    reused; each replacement is checked before an atomic rename. Instantiating
    AestheticsEar itself never downloads or installs anything.
    """
    root = _cache(local_path)
    sources = [(root/"source"/name,
                f"https://raw.githubusercontent.com/facebookresearch/audiobox-aesthetics/{_SOURCE}/{name}", sha)
               for name, sha in _SOURCE_FILES.items()]
    models = [(root/"model"/name,
               f"https://huggingface.co/facebook/audiobox-aesthetics/resolve/{_MODEL}/{name}", sha)
              for name, sha in _MODEL_FILES.items()]
    for path, url, expected in (*sources, *models):
        if path.is_file() and _digest(path) == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+".part")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=45) as response, temporary.open("wb") as output:
                    while block := response.read(1024*1024):
                        output.write(block)
                if _digest(temporary) != expected:
                    raise ValueError(f"Audiobox file does not match pinned revision: {path.name}")
                temporary.replace(path)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt+1)
    return root


def to_aesthetics_rate(audio: np.ndarray, sr: int, *, normalize_lufs: float | None = -24.) -> np.ndarray:
    """Copy/downmix/resample to 16 kHz; reject invalid, silent or DC-only input.

    Input must last at least 100 ms. Normalization preserves transient peaks in
    floating point, including peaks above PCM full scale; no limiter is applied.
    None preserves the input level. This operation never mutates caller arrays.
    """
    x = np.asarray(audio, dtype=float)
    if (not x.size or x.ndim not in (1, 2) or not np.isfinite(x).all()
            or not np.isfinite(sr) or sr < 1000 or int(sr) != sr or x.shape[-1] < sr*.1):
        raise ValueError("aesthetics needs finite mono/channels-first audio, at least 100 ms, and a valid rate")
    mono = np.atleast_2d(x).mean(axis=0)
    if not np.isfinite(np.std(mono)) or np.std(mono) < 1e-8:
        raise ValueError("aesthetics cannot score silent or DC-only audio")
    if normalize_lufs is not None:
        if not np.isfinite(normalize_lufs) or not -80 <= normalize_lufs <= 0:
            raise ValueError("normalize_lufs must be None or between -80 and 0")
        # BS.1770 needs a 400 ms block; zero padding is for level measurement
        # only, not for changing the valid duration passed to the encoder.
        level_audio = np.pad(mono, (0, max(0, round(sr*.4)-len(mono))))
        level = lufs(level_audio[None, :], int(sr))
        if not np.isfinite(level):
            raise ValueError("aesthetics input has no measurable loudness")
        mono = mono*10**((normalize_lufs-level)/20)
    divisor = gcd(_RATE, int(sr))
    if sr != _RATE:
        mono = resample_poly(mono, _RATE//divisor, int(sr)//divisor)
    return mono.astype(np.float32)


class AestheticsEar:
    """Four official CE/CU/PC/PQ regression scores, CPU FP32 by default.

    PQ describes production quality, not authenticity. Even a sine can score
    highly; noise controls passed across our instruments, while quantization
    controls failed on violin and one distorted guitar. Use reference errors and
    calibrated comparisons alongside these scores. CE measures enjoyment; when
    matching a recording, its distance to the reference is more relevant than
    maximizing it. Short isolated notes are outside much of the training data.
    """

    model_name = "facebook/audiobox-aesthetics"
    revision = _MODEL
    source_revision = _SOURCE

    def __init__(self, local_path: str | Path | None = None, *, device: str = "cpu") -> None:
        root = _cache(local_path)
        for folder, files in (("source", _SOURCE_FILES), ("model", _MODEL_FILES)):
            for name, expected in files.items():
                path = root/folder/name
                if not path.is_file():
                    raise FileNotFoundError("Audiobox cache is incomplete; call fetch_aesthetics explicitly first")
                if _digest(path) != expected:
                    raise ValueError(f"Audiobox cache does not match pinned revision: {name}")
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError("Audiobox needs optional notes[quality] dependencies") from error
        package = root/"source/src/audiobox_aesthetics"
        identifier = hashlib.sha256(str(package.resolve()).encode()).hexdigest()[:12]
        namespace = f"_notes_audiobox_{identifier}"
        if namespace not in sys.modules:
            spec = importlib.util.spec_from_file_location(namespace, package/"__init__.py",
                                                         submodule_search_locations=[str(package)])
            if spec is None or spec.loader is None:
                raise ImportError("cannot load the verified Audiobox package")
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            spec.loader.exec_module(module)
        try:
            backend = importlib.import_module(namespace+".model.aes")
        except ImportError as error:
            raise ImportError("Audiobox needs optional notes[quality] dependencies") from error
        config = json.loads((root/"model/config.json").read_text(encoding="utf-8"))
        model = backend.AesMultiOutput(**config)
        state = load_file(str(root/"model/model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=True)
        self.weight_keys = len(state)
        self.parameter_count = sum(p.numel() for p in model.parameters())
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._transforms = config["target_transform"]

    def score_audio(self, audio: np.ndarray, sr: int, *, normalize_lufs: float | None = -24.) -> dict[str, float]:
        """Official 10-second chunks, valid-sample masks and duration-weighted scores."""
        wave = to_aesthetics_rate(audio, sr, normalize_lufs=normalize_lufs)
        torch = self._torch
        rows, lengths = [], []
        for start in range(0, len(wave), 10*_RATE):
            chunk = wave[start:start+10*_RATE]
            tensor = torch.zeros((1, 1, 10*_RATE), dtype=torch.float32, device=self._device)
            mask = torch.zeros_like(tensor, dtype=torch.bool)
            tensor[0, 0, :len(chunk)] = torch.from_numpy(chunk).to(self._device)
            mask[0, 0, :len(chunk)] = True
            with torch.inference_mode():
                raw = self._model({"wav": tensor, "mask": mask})
            rows.append({axis: float(value.item()*self._transforms[axis]["std"]+self._transforms[axis]["mean"])
                         for axis, value in raw.items()})
            lengths.append(len(chunk))
        result = {axis: float(np.average([row[axis] for row in rows], weights=lengths))
                  for axis in ("CE", "CU", "PC", "PQ")}
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Audiobox produced nonfinite diagnostic scores")
        return result
