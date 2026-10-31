"""The wooden body of an acoustic guitar, designed rather than measured, and miked.

The strings barely move air; the top plate does, driven through the bridge, and with the air
in the box it rings at its own modes: the Helmholtz resonance of the sound hole near 100 Hz,
the top's first bending mode near 200 Hz, the back near 230 Hz, then a ladder of plate modes
up to ~1 kHz, each a damped sinusoid; above them the modes crowd into a diffuse decay that a
short burst of coloured noise stands in for. `size` scales the box (1: a dreadnought; a
smaller body rings higher), and the impulse response is normalised to unity in the third
octave around 1 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import oaconvolve

from notes.engine.base import Effect, RenderContext
from notes.engine.dsp import filt
from notes.engine.registry import register_effect

__all__ = ["Body", "body_ir"]

#: Body modes of a dreadnought: (Hz, Q, level dB, against the bridge's direct share at 0 dB).
_MODES = (
    (98.0, 14.0, -4.0), (198.0, 22.0, -5.0), (232.0, 28.0, -10.0), (296.0, 30.0, -12.0), (384.0, 34.0, -10.0),
    (466.0, 38.0, -13.0), (556.0, 40.0, -14.0), (640.0, 44.0, -15.0), (788.0, 48.0, -16.0), (962.0, 50.0, -17.0),
)
_LENGTH = 0.25  # s
_DIFFUSE_T60 = 0.07  # s
_SEED = 1969


def _minimum_phase_ir(ir: np.ndarray) -> np.ndarray:
    from notes.engine.effects.speaker import _minimum_phase

    n_fft = 1 << (max(32768, 8*len(ir))-1).bit_length()
    return _minimum_phase(np.abs(np.fft.rfft(ir, n_fft)))[:len(ir)]


@lru_cache(maxsize=16)
def body_ir(sr: int, size: float = 1.0, diffuse: float = 0.8, phase: str = "designed") -> np.ndarray:
    """Body impulse response at `sr`: the modal ladder plus a diffuse noise decay at `diffuse`
    (relative level), unity gain at 1 kHz. Optional minimum phase concentrates
    energy earlier while approximately preserving the magnitude spectrum.
    """
    if phase not in ("designed", "minimum"):
        raise ValueError("body phase must be designed or minimum")
    n = int(_LENGTH * sr)
    t = np.arange(n) / sr
    ir = np.zeros(n)
    for f, q, db in _MODES:  # each mode a damped sine whose resonance peak is `db` over the direct share
        f /= size
        decay = np.pi * f / q
        ir += 10.0 ** (db / 20.0) * 2.0 * decay / sr * np.exp(-decay * t) * np.sin(2.0 * np.pi * f * t) * q
    noise = np.random.default_rng(_SEED).standard_normal(n) * 10.0 ** (-3.0 * t / _DIFFUSE_T60)
    noise = filt(filt(noise, "highpass", 400.0 / size, sr, 0.7), "lowpass", 10000.0, sr, 0.7)
    noise *= diffuse * 4.0 / max(float(np.sqrt(np.sum(noise**2))), 1e-12)  # a few dB under the direct share
    ir += noise
    ir[0] += 1.0  # the bridge's own direct share: the attack is not all ringing
    spec = np.abs(np.fft.rfft(ir, 1 << 15))
    f = np.fft.rfftfreq(1 << 15, 1.0 / sr)
    around_1k = float(np.sqrt(np.mean(spec[(f > 890.0) & (f < 1120.0)] ** 2)))  # a third octave, not one bin
    ir = ir / max(around_1k, 1e-12)
    if phase == "minimum":
        # Oversampling the log spectrum limits cepstral wrap-around. Retain the
        # original tail length, so the renderer's allocation contract is unchanged.
        ir = _minimum_phase_ir(ir)
    return ir


@register_effect("body")
@dataclass(frozen=True, kw_only=True)
class Body(Effect):
    """An acoustic guitar body (`body_ir`) and a mic in front of it: `size` of the box (1 a
    dreadnought), `diffuse` the share of the dense upper modes, `mix` how much of the sound is
    the body (0: the bare string, as a piezo pickup hears it). ``phase="minimum"``
    concentrates the complete dry/wet response earlier while retaining its magnitude;
    ``phase="designed"`` preserves the original response.
    """

    size: float = 1.0
    diffuse: float = 0.8
    mix: float = 1.0
    phase: str = "designed"

    def __post_init__(self) -> None:
        if self.phase not in ("designed", "minimum"):
            raise ValueError("body phase must be designed or minimum")

    def tail(self, ctx: RenderContext) -> float:
        return _LENGTH

    def process(self, x: np.ndarray, ctx: RenderContext) -> np.ndarray:
        size, diffuse = round(float(self.size), 4), round(float(self.diffuse), 4)
        if self.phase == "minimum":
            if self.mix == 1.0:
                ir = body_ir(ctx.sr, size, diffuse, "minimum")
            else:
                # Factor the entire dry/wet transfer. Blending a minimum-phase
                # wet response with dry audio would change the magnitude again.
                ir = body_ir(ctx.sr, size, diffuse) * self.mix
                ir[0] += 1.0-self.mix
                ir = _minimum_phase_ir(ir)
            return oaconvolve(x, ir.reshape((1,) * (x.ndim - 1) + (-1,)), axes=-1)[..., :x.shape[-1]]
        ir = body_ir(ctx.sr, size, diffuse)
        wet = oaconvolve(x, ir.reshape((1,) * (x.ndim - 1) + (-1,)), axes=-1)[..., : x.shape[-1]]
        return (1.0 - self.mix) * x + self.mix * wet
