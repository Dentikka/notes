"""Compare note events to an explicit reference transcription, rather than a spectrum."""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from notes.ears.transcribe import TrackedNote


def compare_notes(
    reference: Sequence[TrackedNote], estimated: Sequence[TrackedNote], *,
    onset_tolerance: float = 0.05, pitch_tolerance: float = 0.5, offset_ratio: float = 0.2,
) -> dict:
    """One-to-one onset+pitch and onset+pitch+offset event matches, in seconds/MIDI.

    Reference and estimate must share the cut's origin and tuning. An end matches
    within max(onset_tolerance, offset_ratio * reference duration). Matching maximises
    the number of valid pairs before minimising timing/pitch error. This measures
    agreement with the supplied notes; it cannot verify those reference notes.
    """
    if onset_tolerance <= 0 or pitch_tolerance <= 0 or offset_ratio < 0:
        raise ValueError("positive onset/pitch tolerances and nonnegative offset ratio required")
    for note in (*reference, *estimated):
        if not np.isfinite([note.start, note.end, note.pitch]).all() or note.start < 0 or note.end <= note.start:
            raise ValueError("notes need finite pitches and nonnegative, increasing start/end times")
    result = {"reference": len(reference), "estimated": len(estimated)}
    if reference and estimated:
        onset = np.abs(np.subtract.outer([n.start for n in reference], [n.start for n in estimated]))
        pitch = np.abs(np.subtract.outer([n.pitch for n in reference], [n.pitch for n in estimated]))
        offset = np.abs(np.subtract.outer([n.end for n in reference], [n.end for n in estimated]))
        tolerance = np.maximum(onset_tolerance, [offset_ratio * (n.end - n.start) for n in reference])
        valid = (onset <= onset_tolerance) & (pitch <= pitch_tolerance)
        ends_valid = valid & (offset <= tolerance[:, None])
        error = (onset / onset_tolerance + pitch / pitch_tolerance) / 3
        for name, mask in (("onset_pitch", valid), ("onset_pitch_offset", ends_valid)):
            cost = np.where(mask, error, len(reference) + len(estimated) + 1)
            rows, columns = linear_sum_assignment(cost)
            matched = int(mask[rows, columns].sum())
            result[name] = _counts(matched, len(reference), len(estimated))
    else:
        result.update({name: _counts(0, len(reference), len(estimated))
                       for name in ("onset_pitch", "onset_pitch_offset")})
    return result


def _counts(matched: int, reference: int, estimated: int) -> dict:
    precision, recall = matched / max(estimated, 1), matched / max(reference, 1)
    return {"matched": matched, "precision": precision, "recall": recall,
            "f1": 2 * matched / max(reference + estimated, 1)}
