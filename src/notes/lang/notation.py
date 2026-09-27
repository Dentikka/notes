"""Mini-notation: compact strings for melodies, step grids and harmony.

    melody("A4 C5 E5 _ D5 . C5@2")          # '_' holds, '.' rests, '@n' lasts n steps
    melody("[A3,C4,E4]@4 E4> D4!2")         # '[..]' stacks a chord, '>' accents, '!n' repeats
    melody("D5^2~@4 E5~@2 G5^")             # '^n' bends n semitones (2 if bare), '~' vibrato
    steps(kick="x...x...", hat="x.x.x.x.")  # a cell per step: x hit, X accent, o soft
    chords("Am F C G")                      # chord symbols, one per 4/4 bar by default
    euclid(3, 8)                            # 'x..x..x.' — a maximally even rhythm

Bar lines '|' are ignored everywhere and can be used freely for readability.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from fractions import Fraction

from notes.ir.gm import drum_number
from notes.lang.music import BeatsLike, E, Music, Note, Rest, S, W, beats, cat, place, stack
from notes.lang.theory import chord_pitches, parse_pitch, voice_lead

__all__ = ["chord", "chords", "euclid", "hit", "melody", "parse_hits", "steps"]

_HIT_VELOCITY = {"x": 0.8, "X": 1.0, "o": 0.5}
_RESTS = frozenset(".-~")
_HOLD = "_"
_BEND_RE = re.compile(r"\^(-?\d+(?:\.\d+)?)?$")
_VIBRATO_DEPTH = 0.25
_BEND_SEMITONES = 2.0


def parse_hits(pattern: str) -> tuple[list[tuple[int, int, float]], int]:
    """Parse a step grid into hits (start step, length in steps, velocity) and its size in steps.

    Cells: 'x' hit, 'X' accent, 'o' soft, '1'..'9' velocity n/9, '_' holds the previous hit
    one more step, '.', '-' or '~' rest. Spaces and '|' are ignored.
    """
    cells = [c for c in pattern if not c.isspace() and c != "|"]
    hits: list[list] = []
    for i, c in enumerate(cells):
        if c in _HIT_VELOCITY:
            hits.append([i, 1, _HIT_VELOCITY[c]])
        elif c in "123456789":
            hits.append([i, 1, int(c) / 9])
        elif c == _HOLD:
            if hits and hits[-1][0] + hits[-1][1] == i:
                hits[-1][1] += 1
        elif c not in _RESTS:
            raise ValueError(f"Bad step {c!r} in {pattern!r}: use x X o 1-9 for hits, . - ~ for rests, _ to hold")
    return [(s, n, v) for s, n, v in hits], len(cells)


def _split_token(token: str) -> tuple[str, Fraction, bool, int]:
    """'C5>@2!3' -> ('C5', 2, True, 3): body, length in steps, accent, repeat count."""
    body, length, repeat = token, Fraction(1), 1
    if m := re.search(r"!(\d+)$", body):
        repeat, body = int(m.group(1)), body[: m.start()]
    if m := re.search(r"@([\d./]+)$", body):
        length, body = beats(m.group(1)), body[: m.start()]
    accent = body.endswith(">")
    body = body[:-1] if accent else body
    if not body or length <= 0 or repeat < 1:
        raise ValueError(f"Bad token {token!r}")
    return body, length, accent, repeat


def _articulation(body: str) -> tuple[str, dict[str, float]]:
    """'D5^2~' -> ('D5', {'vibrato': 0.25, 'bend': 2.0}): strip the vibrato and bend marks."""
    params: dict[str, float] = {}
    if body.endswith("~"):
        params["vibrato"] = _VIBRATO_DEPTH
        body = body[:-1]
    if m := _BEND_RE.search(body):
        params["bend"] = float(m.group(1)) if m.group(1) else _BEND_SEMITONES
        body = body[: m.start()]
    return body, params


def melody(
    text: str,
    step: BeatsLike = E,
    *,
    vel: float = 0.8,
    octave: int = 4,
    resolve: Callable[[str], float] | None = None,
) -> Music:
    """A line of notes, one token per `step` (an eighth note by default).

    Tokens: a pitch ('A4', 'F#3'; 'Bb' falls in `octave`), a chord '[C4,E4,G4]', '.' rest,
    '_' hold. Suffixes, in this order: '^n' bend by n semitones ('^' alone: 2), '~' vibrato,
    '>' accent, '@n' length in steps, '!n' repeat — e.g. 'D5^2~>@4'.
    """
    st = beats(step)
    to_pitch = resolve or (lambda tok: parse_pitch(tok, octave))
    items: list[list] = []  # [pitches or None, length in steps, velocity, params]
    for token in text.split():
        if token == "|":
            continue
        body, length, accent, repeat = _split_token(token)
        for _ in range(repeat):
            if body == _HOLD:
                if not items:
                    raise ValueError(f"'_' must follow a note or rest in {text!r}")
                items[-1][1] += length
            elif body in _RESTS:
                items.append([None, length, 0.0, {}])
            else:
                body_, params = _articulation(body)
                names = body_[1:-1].split(",") if body_.startswith("[") and body_.endswith("]") else [body_]
                pitches = [to_pitch(n.strip()) for n in names]
                items.append([pitches, length, min(1.0, vel * 1.25) if accent else vel, params])
    if not items:
        raise ValueError("melody() got no notes")
    return cat(
        Rest(n * st) if ps is None else stack(*(Note(p, n * st, v, kw) for p in ps)) for ps, n, v, kw in items
    )


def _lane_target(name: str) -> tuple[float, tuple[tuple[str, str], ...]]:
    try:
        sound, number = drum_number(name)
    except ValueError as drum_error:
        try:
            return parse_pitch(name), ()
        except ValueError:
            raise drum_error from None
    return float(number), (("sound", sound),)


def steps(
    lanes: Mapping[str, str] | None = None,
    /,
    *,
    step: BeatsLike = S,
    length: BeatsLike | None = None,
    **named: str,
) -> Music:
    """Step-sequencer grid, one lane per sound; each cell lasts `step` (a sixteenth by default).

    Lane names are drum sounds ('kick', 'snare', 'hat', 'oh', 'clap', 'crash', ... or the
    aliases 'bd', 'sd', 'hh') or pitches via a dict: steps({'C2': 'x..x'}). With `length`
    every lane loops to fill it, so lanes of different sizes form a polyrhythm.
    """
    grid = {**(lanes or {}), **named}
    if not grid:
        raise ValueError("steps() needs at least one lane, e.g. steps(kick='x...x...')")
    st = beats(step)
    layers = []
    for name, pattern in grid.items():
        pitch, params = _lane_target(name)
        hits, cells = parse_hits(pattern)
        if cells == 0:
            continue
        lane = place([(i * st, Note(pitch, n * st, v, params)) for i, n, v in hits], cells * st)
        layers.append(lane if length is None else lane.loop(length))
    return stack(*layers)


def hit(sound: str, dur: BeatsLike = S, vel: float = 0.9) -> Music:
    """A single drum hit lasting `dur` beats (the sound itself rings as long as it rings)."""
    name, number = drum_number(sound)
    return Note(float(number), dur, vel, (("sound", name),))


def euclid(k: int, n: int, rotate: int = 0) -> str:
    """Maximally even placement of `k` onsets in `n` steps, as a step string.

    euclid(3, 8) == 'x..x..x.' (tresillo). Equals Bjorklund's algorithm up to rotation.
    """
    if not 0 <= k <= n:
        raise ValueError("euclid() needs 0 <= k <= n")
    cells = ["x" if (i * k) % n < k else "." for i in range(n)]
    r = rotate % n if n else 0
    return "".join(cells[r:] + cells[:r])


def chord(symbol: str, dur: BeatsLike = W, *, octave: int = 3, vel: float = 0.7) -> Music:
    """One chord symbol ('Am7', 'G/B', 'E5') held for `dur` beats."""
    return stack(*(Note(p, dur, vel) for p in chord_pitches(symbol, octave)))


def chords(
    text: str,
    dur: BeatsLike = W,
    *,
    octave: int = 3,
    vel: float = 0.7,
    voicing: str = "close",
    resolve: Callable[[str], list[float]] | None = None,
) -> Music:
    """A progression, one chord per `dur` beats: 'Am F C G'. '.' rests, '_' or '@n' extend.

    voicing='smooth' re-voices each chord to move as little as possible from the previous one.
    """
    d = beats(dur)
    to_pitches = resolve or (lambda sym: chord_pitches(sym, octave))
    items: list[list] = []  # [pitches or None, length in chords]
    for token in text.split():
        if token == "|":
            continue
        body, length, _, repeat = _split_token(token)
        for _ in range(repeat):
            if body == _HOLD:
                if not items:
                    raise ValueError(f"'_' must follow a chord in {text!r}")
                items[-1][1] += length
            elif body in _RESTS:
                items.append([None, length])
            else:
                items.append([to_pitches(body), length])
    if voicing == "smooth":
        voiced = iter(voice_lead([p for p, _ in items if p is not None]))
        items = [[None if p is None else next(voiced), n] for p, n in items]
    elif voicing != "close":
        raise ValueError(f"voicing must be 'close' or 'smooth', got {voicing!r}")
    return cat(*(Rest(n * d) if ps is None else stack(*(Note(p, n * d, vel) for p in ps)) for ps, n in items))
