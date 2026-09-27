"""Hearing for agents: CLAP embeddings of rendered audio, zero-shot tags and text similarity.

CLAP maps audio and text into one space. The ear embeds a render in 10-second windows —
the model's input size; cutting them ourselves keeps the result deterministic, since the
processor crops longer input at random — averages the windows per section of the score,
and compares the result with label prompts (genre, instrument, mood, production) or with
free-text descriptions. Scores are zero-shot and relative: compare versions and sections
with them, do not read them as absolute truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from notes.ears.vocab import TEMPLATES, VOCABULARY
from notes.ir.score import Score

__all__ = ["CLAP_SAMPLE_RATE", "DEFAULT_MODEL", "Ear", "Tag", "cosine", "format_report", "to_clap_rate", "windows"]

#: laion/larger_clap_music would be the natural choice for music, but the text tower of its
#: Hugging Face conversion is collapsed: every prompt embeds to nearly the same vector
#: (pairwise cosine 0.999 with transformers 4.57 and 5.17). clap-htsat-unfused is sound.
DEFAULT_MODEL = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000
_WINDOW_S = 10.0
_HOP_S = 5.0
_MIN_TAIL_S = 3.0
_MIN_SECTION_S = 0.5
#: Softmax temperature for per-category probabilities (the checkpoint's own logit scale is ~1).
_TEMPERATURE = 0.01


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(_normalize(a) @ _normalize(b))


def to_clap_rate(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mono mix resampled to CLAP's 48 kHz."""
    mono = np.atleast_2d(np.asarray(audio, dtype=float)).mean(axis=0)
    if sr == CLAP_SAMPLE_RATE:
        return mono
    g = gcd(CLAP_SAMPLE_RATE, sr)
    return resample_poly(mono, CLAP_SAMPLE_RATE // g, sr // g)


def windows(
    x: np.ndarray, sr: int = CLAP_SAMPLE_RATE, size_s: float = _WINDOW_S, hop_s: float = _HOP_S
) -> list[np.ndarray]:
    """Full-length windows with a hop; a final window aligned to the end covers a long tail.

    Input no longer than one window is returned whole (the processor repeat-pads it).
    """
    size, hop = int(size_s * sr), int(hop_s * sr)
    if len(x) <= size:
        return [x]
    starts = list(range(0, len(x) - size + 1, hop))
    if len(x) - (starts[-1] + size) >= int(_MIN_TAIL_S * sr):
        starts.append(len(x) - size)
    return [x[s : s + size] for s in starts]


@dataclass(frozen=True)
class Tag:
    label: str
    similarity: float
    prob: float


class Ear:
    """A CLAP model loaded once, embedding audio and text into a shared space."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as err:
            raise ImportError("Listening needs torch and transformers: pip install 'notes[ears]'") from err
        self._torch = torch
        self.model_name = model
        self.device = device
        self._processor = ClapProcessor.from_pretrained(model)
        self._model = ClapModel.from_pretrained(model).to(device).eval()
        self._labels: dict[tuple[str, ...], np.ndarray] = {}

    @staticmethod
    def _features(out: Any) -> Any:
        # transformers 4 returns a tensor, transformers 5 a model output with pooler_output
        return out if hasattr(out, "shape") else out.pooler_output

    def embed_windows(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """One unit vector per 10-second window."""
        clips = windows(to_clap_rate(audio, sr))
        inputs = self._processor(audio=clips, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
        with self._torch.no_grad():
            e = self._features(self._model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()}))
        return _normalize(e.cpu().numpy())

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Unit vector for a whole clip: the normalised mean of its window embeddings."""
        return _normalize(self.embed_windows(audio, sr).mean(axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True)
        with self._torch.no_grad():
            e = self._features(self._model.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()}))
        return _normalize(e.cpu().numpy())

    def embed_labels(self, labels: Sequence[str], templates: Sequence[str]) -> np.ndarray:
        """Prompt ensemble: each label is the normalised mean of its templated prompts (cached)."""
        key = (*templates, "\0", *labels)
        if key not in self._labels:
            prompts = [t.format(label) for label in labels for t in templates]
            e = self.embed_texts(prompts).reshape(len(labels), len(templates), -1).mean(axis=1)
            self._labels[key] = _normalize(e)
        return self._labels[key]

    def tags(
        self, embedding: np.ndarray, vocabulary: Mapping[str, Sequence[str]] | None = None, top: int = 3
    ) -> dict[str, list[Tag]]:
        """Top labels per category with cosine similarity and a within-category softmax."""
        out = {}
        for category, labels in (vocabulary or VOCABULARY).items():
            sims = self.embed_labels(labels, TEMPLATES.get(category, ("{}",))) @ embedding
            probs = _softmax(sims / _TEMPERATURE)
            out[category] = [
                Tag(labels[i], round(float(sims[i]), 4), round(float(probs[i]), 3)) for i in np.argsort(-sims)[:top]
            ]
        return out

    def similarity(self, embedding: np.ndarray, texts: Sequence[str]) -> dict[str, float]:
        """Cosine similarity of a clip embedding to free-text descriptions."""
        return {t: round(float(s), 4) for t, s in zip(texts, self.embed_texts(texts) @ embedding, strict=True)}

    def listen(
        self, audio: np.ndarray, sr: int, score: Score | None = None, prompts: Sequence[str] = (), top: int = 3
    ) -> dict[str, Any]:
        """Tags (and prompt similarities) for the whole piece and for every section of `score`."""
        spans = [("whole", 0.0, np.atleast_2d(audio).shape[1] / sr)]
        if score is not None:
            tempo = score.tempo_map()
            spans += [(m.name, tempo.seconds(m.start), tempo.seconds(m.end)) for m in score.markers]
        sections = []
        for name, start, end in spans:
            clip = np.atleast_2d(audio)[:, int(start * sr) : int(end * sr)]
            if clip.shape[1] < int(_MIN_SECTION_S * sr):
                continue
            emb = self.embed_audio(clip, sr)
            entry: dict[str, Any] = {
                "name": name,
                "start_s": round(start, 2),
                "end_s": round(end, 2),
                "tags": {c: [asdict(t) for t in ts] for c, ts in self.tags(emb, top=top).items()},
            }
            if prompts:
                entry["prompts"] = self.similarity(emb, prompts)
            sections.append(entry)
        return {"model": self.model_name, "temperature": _TEMPERATURE, "sections": sections}


def format_report(report: Mapping[str, Any]) -> str:
    """Human-readable digest of `Ear.listen` output."""
    lines = [f"CLAP ({report['model']}): cosine similarity (softmax share within the category)"]
    for s in report["sections"]:
        lines.append(f"[{s['name']} {s['start_s']:.1f}-{s['end_s']:.1f} s]")
        for category, tags in s["tags"].items():
            ranked = ", ".join(f"{t['label']} {t['similarity']:.3f} ({t['prob']:.2f})" for t in tags)
            lines.append(f"  {category:>10}: {ranked}")
        for text, sim in s.get("prompts", {}).items():
            lines.append(f"  {'prompt':>10}: {sim:.3f}  {text}")
    if "reference" in report:
        lines.append(f"similarity to reference {report['reference']['path']}: {report['reference']['similarity']:.3f}")
    return "\n".join(lines)
