# Experimental numeric timbre profiles

`SpectralTone` synthesizes up to 32 sinusoidal components and eight independently
generated noise bands. `SpectralProfile` stores at most 16 magnitude knots per
component. Neither stores audio samples, measured initial phases, or an impulse
response. The default profile is a designed sine pluck, not an instrument clone.

```python
from notes import Note, Song, SpectralTone, Track
from notes.ears import spectral_profile

# mono is a known steady reference note, already decoded at sample rate sr.
# Analysis is separate from rendering; only numeric magnitudes are retained.
profile = spectral_profile(mono, sr, expected_hz=220., pitch=57.)
song = Song(Track("profile", SpectralTone(profile=profile))(
    Note("A3", dur=2, vel=.7) | Note("E4", dur=2, vel=.5).shift(2)
), bpm=60)
song.render().write("profile.wav")
```

The analyzer requires at least 0.4 seconds of finite mono audio at 8 kHz or more,
with an audible 0.1–0.3 s anchor. The supplied pitch must be correct. Default
normalization sets that anchor's RMS to 0.02; `normalize_rms=None` retains the
recording's amplitude. Analysis uses tracked spectral peaks, complex demodulation
for magnitudes, and coarse residual noise bands. It does not infer note onset,
velocity, physical material, or a transferable instrument model.

Profiles are immutable and validated. `profile.to_dict()` and
`SpectralProfile.from_dict(data)` support plain JSON. The instrument's profile
also round-trips through normal score JSON and the instrument registry. Unknown
profile fields and unsupported schema versions are rejected.

Rendering uses the score's pitch and A4 tuning; `reference_hz`/`pitch` describe
the analysis anchor. Ordinary bends, slides, vibrato and `pitch_points` drive the
oscillators. Components whose maximum frequency reaches 45% of the output sample
rate are omitted; noise bands stay at absolute frequencies and are clipped to
that band limit. Each band has a separate seeded RNG stream, so extending a gate
does not change the note's beginning. The source is mono; the mixer supplies pan.

Envelope knots use seconds from note onset. A two-millisecond gate attack and
`release` (default 0.12 s) control note boundaries. Beyond the profile's final knot,
the envelope fades exponentially with the release time. This prevents an arbitrary
unmeasured sustain; it is not an inferred long-note decay. `level`,
`velocity_reference` (0.7) and `velocity_power` (1.7) control amplitude. Velocity
does not automatically change the spectrum. These controls are not an identified
law for a real instrument.

On a soft E3 guitar take excluded from profile estimation, a fixed soft-note
profile gave dry STFT convergence about 0.17 early and 0.14 at 2–4 seconds after
fundamental-only adaptation. A normal-strength profile did not consistently
transfer to another normal-strength take. Strongly distorted spectra, late tails,
other pitches, dynamics and continuous articulation remain unproven. This source
is an optional reconstruction tool; accepted instrument presets are unchanged.
