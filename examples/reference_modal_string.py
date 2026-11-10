"""Two procedural A2 reconstructions with separately inferred performance controls.

The modal source preserves initial mechanical energy when release_motion changes.
Both sources have their own attack-fitted pluck and level controls. The cabinet
correction is shared. These two A2 cases do not establish realism across a guitar.
Time-varying pitch curves are not supported by ModalString yet.
No audio files, reference phases or measured impulse responses are loaded.
"""
from math import log2

from notes import Cabinet, ElectricGuitar, Filter, GuitarAmp, ModalString, Note, Rest, Song, Track, cat

_BANDS = ((125., .8, 6.1286038253905115), (350., .8, -2.708125410023477),
          (1000., .8, -6.569733522175824), (2800., .8, 2.4927786736061686),
          (6500., .8, -2.3913015328234892), (1250., 3., -17.415297003686614))
_BRIDGE_MOBILITY = .02026740039932687


def build_song(*, baseline: bool = False, amplified: bool = True, calibrated_decay: bool = False) -> Song:
    correction = tuple(Filter(kind="peak", cutoff=f, q=q, gain_db=g) for f, q, g in _BANDS)
    if baseline:
        instrument = ElectricGuitar(amplified=amplified, drive=.65, level=.5 if amplified else 1.,
                                    fretting=True, pickup_position=.06, pickup_width=.04, pickup_q=.7)
        effects = correction if amplified else ()
    else:
        # Shared loss fit from early partial decays of three other A2 takes.
        # Keep bridge mass/stiffness fixed when reducing its damping through Q.
        losses = ({"sustain": 15.14768380129981, "treble_decay": 1.5138166789202858,
                   "bridge_q": 58.06255708239231} if calibrated_decay else {})
        mobility_scale = losses.get("bridge_q", 20.)/20.
        instrument = ModalString(stiffness=.00012398226537801852, bridge_hz=115.,
                                  bridge_q=losses.get("bridge_q", 20.), sustain=losses.get("sustain", 9.),
                                  treble_decay=losses.get("treble_decay", 2.5),
                                  bridge_mobility=_BRIDGE_MOBILITY*mobility_scale, pluck_force=3.33609279766794,
                                  level=.5 if amplified else 1.)
        effects = (GuitarAmp(drive=.65), Cabinet(), *correction) if amplified else ()
    track = Track("guitar", instrument, fx=effects)
    parts = []
    cases = (
        (109.93146876656337, .7, .01692447936581963, .5, .15, .9, -4.528077027402601,
         .2, .3, -7.422744206429046),
        (108.86907025804253, .35, .013798072972829138, -1., .2, .9, 10.64611788000134,
         .15, .1, 5.478269486850724),
    )
    for hz, vel, mobility, motion, pick, hardness, gain, old_pick, old_hardness, old_gain in cases:
        params = {"string": 5, "pick_position": old_pick if baseline else pick,
                  "pick_hardness": old_hardness if baseline else hardness,
                  "source_gain_db": old_gain if baseline else gain,
                  "release_motion": motion, "bridge_scale": mobility/_BRIDGE_MOBILITY}
        if amplified and not baseline:
            # Track effects follow constant-power panning: retain the mono amp input level.
            params["source_gain_db"] += 10*log2(2)/log2(10)
        parts.append(cat(Note(69+12*log2(hz/440), dur=1.5, vel=vel).param(**params), Rest(.35)))
    return Song(track(cat(*parts)), bpm=60, sample_rate=44100, seed=17, loudness=-22,
                title="A2 reconstruction: " + ("waveguide" if baseline else
                      "modal release motion" + (" with calibrated losses" if calibrated_decay else "")))


song = build_song()
