# notes — music as code

`notes` is a framework for writing music as programs, by hand or by an LLM agent. A
composition is a short Python script. The library compiles it into an exact, serialisable
intermediate representation (IR) and renders that to audio with a built-in DSP engine: no
samples, no external synthesizers, bit-reproducible output.

Instrument synthesis must remain fully procedural: no recordings, multisamples,
or recorded impulse responses as sound sources. Reference audio is for evaluation only.

```python
from notes import *

drums  = Track("drums", DrumKit())
guitar = Track("guitar", ElectricGuitar(drive=0.65, pickup="neck"), fx=[Delay(time=0.75), Reverb(mix=0.3)])

beat   = steps(kick="x.....x...x.....", snare="....x.......x...", hat="x.x.x.x.x.x.x.x.")
hook   = melody("A4 C5 E5 _ D5 C5 D5 E5 | F5 _ E5 D5 C5 _ A4 _", step=E)

intro  = drums((beat * 4).mute(bars=4))          # a steady beat, silent on bar 4
chorus = drums(beat * 2) | guitar(hook)          # a melody on a driven guitar over it
song   = Song(intro, chorus * 2, bpm=112, key="A minor")   # the chorus, twice

song.render("song.wav")
```

## Architecture

```
 script.py ──► Music tree ──perform──► Score (IR) ──render──► stereo master ──► WAV
 notes.lang     + | * and ops           notes.ir               notes.engine         + MIDI, JSON,
                (exact rationals)       (JSON, renderer-free)  (numpy / scipy)        piano roll
```

- **Language** (`notes.lang`): an algebra of immutable trees in the spirit of Haskore and
  Euterpea. `a + b` plays in sequence, `a | b` together, `a * n` repeats; operations such
  as `transpose`, `stretch`, `reverse`, `mute`, `swing` and `named` attach to a subtree
  without flattening it, so `chorus * 2` stays a repetition until the end. Time is in beats
  (a quarter note is 1) and kept as exact `Fraction`s: triplets and polyrhythms never drift.
- **IR** (`notes.ir`): a flat score of note events, track specs, section markers and a
  tempo map. Every event carries its structural path (`chorus/…`). Instruments and effects
  are plain `{"type": …, …params}` specs, so other back-ends (MIDI, SoundFonts, DAWs) can
  consume the same file. `Score.to_json()` / `from_json()` round-trip exactly.
  The optional `seed_pitch_int` event marker preserves a builtin integer pitch's
  RNG representation; old unmarked JSON keeps its previous float interpretation.
- **Engine** (`notes.engine`): each note becomes a voice, the voices are summed per track,
  run through the instrument's inserts, panned, run through the track's effects and mixed.
  The master is normalised to a loudness target (BS.1770) and peak-limited. Each voice's
  noise is seeded from its content, so a render is reproducible bit for bit and editing
  one phrase does not re-roll the others.

## The language at a glance

| | |
|---|---|
| sequence, stack, repeat | `a + b`, `a \| b`, `a * 4`, `cat(...)`, `stack(...)`, `place([(t, m), ...])` |
| structure | `.named("chorus")`, `.loop(beats)`, `(a * 8).every(4, fill)` |
| time | `.stretch(k)`, `.fast(k)`, `.slow(k)`, `.shift(b)`, `.reverse()`, `.slice(a, b)`, `.swing(2/3)` |
| pitch | `.transpose(n)`, `.octave(n)`, `.invert("E4")` |
| dynamics | `.velocity(k)`, `.legato(k)`, `.humanize(seed=…)`, `.param(palm_mute=True)` |
| silence | `.mute(bars=4)`, `.mute(bars=[2, 6])`, `.mute(window=(0, 2))` |
| expression | `.vibrato(0.3)`, `.bend(2)`, `.slide(-2)` — pitch curves for guitars, synths, FM |
| production | `guitar.doubled(part)` — two takes panned apart, as rhythm guitars are recorded |
| harmony | `.rhythm("x..x..x.")` re-strikes held chords, `.arp("updown")` arpeggiates |

**Mini-notation** (strings, `|` bar lines ignored):

```python
melody("A4 C5 E5 _ D5 . C5@2 [A3,C4,E4] G4> B4!2", step=E)   # _ hold, . rest, @n length,
                                                              # [..] chord, > accent, !n repeat
melody("E5~@4 D5^2~@2 G5^")                                   # ~ vibrato, ^n bend n semitones
steps(kick="x...x...", snare="....x...", hat="x.x.x.x.")      # x hit, X accent, o soft, 1-9
steps(kick=euclid(3, 8), hat="x..", length=bars(2))           # Euclidean rhythms, polymeter
chords("Am F C G", voicing="smooth")                          # chord symbols, voice-led
Key.parse("A minor").chords("i VI III VII")                   # roman numerals
Key.parse("A minor").melody("1 3 5 8 b7")                     # scale degrees
```

**Instruments**: `Synth` (subtractive) with `Bass`, `Pad`, `Lead`; `FM` with `EPiano`,
`Bell`; `Pluck` and `ElectricGuitar` (Karplus–Strong strings that follow bends and
vibrato, pick click, pickup comb and resonance, a two-stage amp clipped at 4x the sample
rate, a 4x12 cabinet; `drive`, `tone`, `pickup`, palm mutes); `DrumKit` (808/909-style synthesis addressed by GM
names: `kick`, `snare`, `hat`, `oh`, `clap`, `crash`, `ride`, toms, …). Presets are
calibrated to equal loudness, so a track's `gain` is dB relative to the others.

**Effects**: `Reverb` (Freeverb), `Delay` (tempo-synced), `Chorus`, `Compressor`, `EQ`,
`Filter`, `Drive`, `GuitarAmp`, `Cabinet`.

## Command line

### Transcribe an isolated instrument stem

```bash
notes transcribe bass.wav -o out/bass --low 23 --high 55
notes transcribe solo.wav -o out/solo --method melody --instrument electric --low 52 --high 88
notes transcribe acoustic.wav -o out/acoustic --method templates --instrument twelve --low 40 --high 80 --max-polyphony 6
```

`mono` uses YIN and carries a known pitch through audible unvoiced tails until silence
or a new onset. `melody` estimates the dominant pitch over accompaniment; choose its
register carefully and inspect the result for octave errors. It retains measured
bends/vibrato as per-note `pitch_points` (seconds, semitone offsets). This additive
renderer parameter also works through IR JSON; existing vibrato/bend/slide remain available.
For plucked strings, `part.param(pitch_interpolation="sinc")` opts into a
Blackman-windowed sinc readout for those curves. It preserves upper partials more
accurately than linear interpolation and filters components that would alias when
pitch rises. Acoustic/nylon/electric guitars, bass, Pluck and SynthGuitar support
it. The original linear readout remains the default; unmodulated notes are unchanged.
The optional mode uses filter lookahead without moving note events, processes
bounded chunks, and supports upward ratios up to 16x. This fixes a resampling
approximation; it does not by itself establish instrument realism. See
[Windowed Sinc Interpolation](https://www.dsprelated.com/freebooks/pasp/Windowed_Sinc_Interpolation.html).
The legacy melody path splits notes at full-signal attacks, which can re-pick a held
lead when accompaniment changes. Experimental `--melody-onsets pitch` instead uses
energy at the estimated voice's harmonics. It can miss compressed attacks or still
respond to overlapping partials; it has not been validated as a general solution.

`templates` jointly fits spectra rendered for each candidate note, with nonnegative
sparse activation and independent note endings. It can follow chords and arpeggios
without imposing a chord vocabulary or borrowing the bass rhythm. Include octave
courses inside each twelve-string template. `--settings params.json` accepts a flat
instrument-parameter object; `--a4` sets recording tuning. `--start/--end` cut the
reference; `--channel mid|side|left|right` selects the signal to analyse. Outputs are
local to `-o`: `notes.json` (times relative to the cut, pitches and contours), MIDI,
IR, reconstruction WAV, piano roll, spectrogram, comparison and A/B audio.

These are inspectable baselines, not guaranteed transcription of arbitrary songs.
Template mismatch, distortion, separation noise and bends can all create false notes.
`--relative-db`, `--sparsity`, `--min-duration` and `--max-polyphony` expose the assumptions.
Residual reconstruction error is not note accuracy. The reconstruction omits effects
outside the chosen instrument. Supply isolated stems; separation is a separate step.

**Current real-recording limitation:** the Humanity guitar reconstructions were rejected
in listening despite improved spectral/onset metrics. Treat all extracted notes as
candidates. `notes.json` records `status: unverified` unless `--reference-notes gold.json`
is supplied. That file contains `notes` with `start`, `end`, `pitch` in the cut's seconds
and tuning. Reference scoring reports one-to-one onset+pitch and onset+pitch+offset
precision/recall/F1; it measures agreement with those supplied notes, not their validity.
Tests cover a held note wrongly split into four plucks, shifted rhythms and wrong octaves.

An additional **experimental** `--method basic-pitch --model-path /path/to/nmp.onnx`
uses Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) waveform model on
CPU, with our own pitch-specific onset/frame decoder. Install `notes[transcription]`
for ONNX Runtime; obtain the official model separately. No model is downloaded by the
CLI. The report stores its SHA-256. This avoids matching against our synthesizer's
templates, but can still hallucinate notes and does not reliably separate lead/rhythm
guitars. It is an independent baseline, not a verified replacement. The original
methods and defaults remain available.

For Basic Pitch, `--note-decoder onsets` starts notes only at their own onset peaks
(default onset threshold 0.7), instead of treating frame-confidence runs as new notes.
This avoids false re-picks after confidence dropouts. `--release-threshold 0.15`
extends tails of already accepted notes without admitting new weak candidates.
`--harmonic-ratio 0.85` optionally drops a weaker coincident upper harmonic; it can
also drop a genuine quiet octave and is disabled by default. `--no-contours` renders
fixed pitches when model-estimated bends are inappropriate. All settings are recorded.

On the existing locally annotated clean Still Loving You control, this decoder with
harmonic pruning matched all 23 onset/pitch events in the first eight seconds (legacy:
22 matches among 42 candidates). On a separate 8.0–10.35 s segment it found 5 of 6 notes
with no extras (legacy: 5 matches among 11). These are narrow checks against an existing
manual part, not a general accuracy benchmark; note endings and distorted guitars
remain unresolved. Legacy `--note-decoder frames` remains the default for compatibility.

`--velocity-source audio` measures dynamics from the increase in harmonic energy at
each detected attack. Model confidence is not loudness: the legacy tracker-level
mapping can flatten the accents even with correct note timestamps. Coincident higher
notes are excluded from a lower note's shared partials when measuring its energy.
Measured dynamics use peak normalization, keeping relative levels without clipping
strong accents at a percentile reference. Original tracker levels are retained in
`notes.json`. Legacy `--velocity-source tracker` remains available and is the default.
Python: `measure_note_levels(audio, sr, notes, a4)` followed by
`tracked_music(notes, peak_normalize=True)`. This estimates pluck energy, not exact
instrument velocity; timing, articulation and note endings still need separate checks.

For an isolated monophonic channel, `measure_note_levels(audio, sr, notes,
method="rms")` measures sounding power during 10–120 ms after onset, capped at
the annotated note end (very short notes use their whole span). It preserves
quieter repeated notes that positive spectral growth can mistake for silence.
Global DC is removed. This mode includes any previous ringing and cannot separate
simultaneous instruments; use it on isolated voices or individual hex-pickup
strings, not a full mix. Both modes estimate relative levels, not calibrated MIDI
velocity. The default remains `method="growth"`.

`note.param(source_gain_db=...)` changes a note's signal level after synthesis
and before instrument inserts and track effects, without changing its velocity,
pitch or timing. This separates reference-level calibration from the performance
controls that can also change hammer/pick spectra. The range is -120..120 dB;
the default/explicit zero preserves the original waveform exactly. With a guitar
amplifier, this gain changes the operating point and therefore distortion.

Python APIs: `template_track`, `notes_from_templates`, `instrument_templates` in
`notes.ears.polyphonic`; `notes_from_track(..., sustain_unvoiced=True)` and
`tracked_music(notes, contours=True)` in `notes.ears.transcribe`. The latter returns
Music in seconds, to play at BPM 60. Existing transcription defaults are unchanged.

### Fit an instrument to a recording

`notes fit reference.wav examples/reference_fit.py -o out/fit --max-evaluations 30`
searches instrument/effect settings for a **fixed, known part** against an isolated
reference (or an externally separated stem). Use `--start m:ss --end m:ss` to cut the
reference; the script's part starts at zero. The Python script defines `initial`,
`grid` and `build_song(settings) -> Song` at 44100 Hz. Keep notes, tuning, timing and
seed fixed, and use `loudness=None` to avoid mastering masking the instrument's response.

The search uses cached coordinate descent, stops when a pass makes no improvement,
and never exceeds `--max-evaluations` renders (default 100). `--passes` defaults to 2.
Each trial measures time-dependent spectrum, tone, attacks, envelopes and stereo width.
Overall gain and up to `--max-lag` seconds of offset (default 0.05) are matched separately.
The Python API is `notes.ears.fit_reference(reference, render, initial, grid)`; `render`
returns audio at the specified sample rate. The result includes settings, comparison
and all unique trials; `on_trial` supports incremental logging.

Outputs: `fitted.json`, flushed `trials.jsonl`, the model script snapshot, `fitted.png`,
and `fitted.wav`/`fitted.ab.wav`/`fitted.toggle.wav`. Audio is aligned and loudness-matched;
the gain and offset are recorded in the report. Use a separate output folder for each run.
The grid search finds a local solution among the supplied values, not necessarily a
unique parameter set. A wrong part can still receive a lower loss by changing timbre;
check transcription separately. Spectral peaks include overtones and are not note labels.
Exact reproduction of a real instrument is limited by the synthesis model and stem quality.

```bash
notes render examples/first_groove.py -o out/      # WAV + IR JSON + MIDI + piano roll + meta.json
notes render song.py --spectrogram --bits 24
notes render song.py --listen --prompt "energetic synthwave"   # ... and hear it (see Ears)
notes listen out/first_groove.wav --reference other.wav         # tags per section, similarity
notes info song.py                                 # sections, tracks, ranges — no rendering
```

`meta.json` records loudness, peaks and per-track levels in the mix, the script's hash
and the engine's git revision; the script itself is copied next to its outputs.

## Ears

An agent cannot listen, so `notes.ears` gives it a model's hearing. CLAP
([`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused)) embeds the
render in 10-second windows — cut deterministically, averaged per section of the score —
and ranks genre, instrument, mood and production labels (prompt-ensembled) and any
free-text prompts; `--reference` compares two renders. Scores are zero-shot and relative:
use them to compare versions and sections, not as absolute judgments. (`larger_clap_music`
is not used: in its Hugging Face conversion every prompt embeds to nearly the same vector.)

## Install

```bash
pip install -e ".[dev]"      # numpy + scipy; matplotlib for pictures, pytest for tests
pip install -e ".[ears]"     # torch + transformers for listening
python -m pytest
```

## Status

Early but working: the full path from script to audio and back to a model's ears,
87 tests, one demo (`examples/first_groove.py`). Next: an MCP server so agents can compose,
render and listen in a loop; send buses; automation curves; a declarative text front-end.
CLAP recognises every synth preset and, in a produced context, every guitar variant.

## Piano and bowed strings

Known performed parts can be read without transcription or extra dependencies:
`from notes.performance import read_midi_performance`. The returned `notes` keep
`onset`, `key_offset` and effective binary-pedal `offset` in seconds, raw MIDI
velocity/release velocity, pitch and channel. SMF 0/1 PPQN, global tempo changes,
running status, channel-local sustain (CC64) and sostenuto (CC66) are supported.
All controls are retained; soft pedal (CC67) is not modeled. Pedal positions use
the threshold 64, not continuous half-pedaling. A note still held by a pedal at
file end has `damper_truncated=True`. Unmatched keys, SMPTE/format 2 files,
noncentral pitch bends and channel-mode reset/termination are rejected.
Repeated same-key overlaps pair FIFO; equal-tick cross-track ties use file order.
These gates describe an evaluation performance, not measured acoustic decay.
They do not add resonance or re-pedaling to the synthesizer.

`Piano()` synthesizes decaying inharmonic string partials, unison beating and a
velocity-dependent hammer spectrum. `music.param(pedal=True)` preserves each note's
natural decay after key release (a per-note pedal, not continuous pedal automation).
`BowedString()`, `Violin()` and `Cello()` provide sustained harmonic bow sources,
body resonances, friction noise and controllable vibrato. Explicit per-note vibrato
parameters override the preset. These are procedural approximations, not recorded
instrument libraries or a claim of reference matching.

`Piano(model="hammer_damped")` and `Piano(model="hammer")` also accept optional
`partial_gain_points` on the instrument or note. Rows contain seconds and linear
gains for successive dispersive mode groups, including each mode's detuned partner.
The same limits of 512 rows, 80 gains per row and gains in [0,16] apply; only modes actually
present in the model (at most 64) are read out. Gains affect the integrated contact
head and free ring at their actual clocks, before key-release/pedal envelopes.
They do not alter hammer contact, damping or modal frequencies. Empty keeps the
accepted waveform; an empty note override disables a configured curve. Nonempty
curves on legacy/soundboard pianos are rejected rather than silently ignored.

Optional `unison_strings=1..3` on contact pianos or their notes selects an equally
weighted free-ring group. Two/three components span `-detune..+detune` cents;
one component keeps the central mode. Their phases evolve from the calculated
impact state, with a smooth transition after contact. `None` retains the original
unequal two-component readout, and a note-level `None` restores it. This does not
add separate coupled hammer/string states; it is a controllable readout model.
Nonempty settings on legacy/soundboard pianos and explicit unisons beyond the
renderable frequency band are rejected. No recorded phases are used.

Contact pianos also accept optional `hammer_speed_points=((0.,0.), (1.,4.))`:
monotone (normalized MIDI velocity, initial hammer speed in model m/s) points.
At most 64 points must cover velocity 0..1; speed is in 0..4 and interpolates
linearly. Empty preserves the original `0.6 + 3.4*velocity` mapping exactly.
This changes the nonlinear impact and its spectrum, rather than multiplying
the final waveform. A note can supply a different curve or an empty reset.
Per-note `hammer_speed` overrides the curve; `None` restores the original
mapping. Velocity zero is silent even with an explicit speed. These controls
are only supported by hammer/hammer_damped; they are calibration controls,
not a claim of a universally correct velocity response or measured hammer speed.

Contact pianos accept `felt_stiffness_scale=.25..16`, also per note, to multiply
the felt compression-force coefficient independently of string bending stiffness.
`contact_oversample=1..16` refines only contact integration; its default 1 keeps
the original >=176.4 kHz rate and output-dependent mode band. Increasing it does
not add modes or change the output sample rate. Check convergence for harder felt.
Both controls default to the old behavior and require contact piano models.

For `hammer_damped`, optional `radiation_compensation=0..1` restores a fraction
of power removed by the synthetic register readout. It uses modal amplitudes at
first hammer separation and compares the original versus filtered bridge tap.
The resulting single scalar multiplies the whole voice: relative partials,
frequency, damping and contact forces stay the same. It is not a physical
soundboard radiation model or an energy-conservation guarantee. The default 0
preserves the accepted source; notes may override all three controls.

`Piano(model="hammer_damped", radiation_loss_scale=1.0)` independently scales
the extra register/frequency-dependent damping, during contact and free ringing.
Values in 0..4 are supported, also per note. Zero removes only that extra loss;
intrinsic sustain losses and the existing modal readout remain. One preserves
the accepted source. This changes decay rates and may change the impact state;
it is not output EQ, an identified soundboard or a validated global preset.
Other piano models reject nondefault values instead of silently ignoring them.

Per-note `undamped=True` permits natural free ringing after key release without
setting a pedal event. It uses the existing finite natural tail, applies to all
piano models and does not simulate sympathetic resonance or half-pedal. The
default false preserves key-off damping; an actual `pedal=True` still rings.
There is no automatic assumption about a particular piano's last damped key.
The renderer's existing quiet-tail trimming still applies.

`CoupledPiano` is a separate experimental instrument: one nonlinear felt hammer
drives one to three explicit string coordinates. Set
`unison_cents=(-1.4, 0., 1.4)` or `(0.,)`; coincident offsets preserve the
aggregate single-string limit. Unlike the existing unison readout, each string
has its own contact state. No waveform, measured phase or impulse response is
accepted. `examples/coupled_piano.py` supplies a short DSL demonstration.

Optional `unison_weights=(.115, .77, .115)` assigns relative output contributions
to the three strings. Nonnegative weights are normalized to sum one, and each
offset/weight pair stays together when sorted. This leaves mechanical masses,
contact force and damping unchanged. `None` preserves the original equal output
exactly; per-note `unison_weights=None` resets an inherited weighting. Pitch
compensation follows the weighted fundamental observation. This is a designed
readout control, not measured soundboard radiation or an accepted piano preset.

The compression-only contact uses a discrete potential gradient and midpoint
linear mechanics. `modes=1..64` and `mechanical_band=17640` bound the uncoupled
string basis independently of output sample rate. Observation is oversampled
and decimated as needed; factors above 16 raise rather than alias. The fundamental
of every string must fit below .4 of the output rate. `contact_oversample=1..16`
refines contact integration while retaining the same physical modes.

`pitch_compensation=True` tunes a generated fundamental-coordinate centroid
weighted at .1 seconds, using a fixed requested-note region and a bracketed
solver. It uses no recording and is not a perceptual-pitch certificate. Missing
or nonconvergent configurations raise explicitly. `bridge_mass=None` clamps
the endpoint; optional positive mass plus `bridge_hz`/`bridge_q` enables a moving
endpoint. This is not calibrated soundboard radiation: a tested moving-bridge
candidate had worse timbre and excessive attack peaks, so no such preset is adopted.

Source controls, `release`, `ring_time` and `pitch_compensation` accept note
overrides. `pedal=True` or `undamped=True` permits a finite
`max(release, ring_time)` after the gate; note overrides also set the renderer's
tail allocation. There is no inter-key coupling, continuous pedal, recorded
phase transfer or support for changing pitch contours. The older Piano's
readout curves and impact-noise controls are not implemented here.
`hammer_speed_points=((0., 0.), (1., 4.))` optionally maps MIDI velocity to
initial hammer speed in model m/s before nonlinear contact. Per-note
`hammer_speed` in 0..4 overrides that curve; `hammer_speed=None` restores the
original `.6+3.4*velocity` law and an empty curve retains existing audio exactly.
Zero speed or zero velocity is silent. This controls contact energy and spectrum,
not output gain; a register-specific calibration still needs independent validation.

Contact calibration has a scale ambiguity in the current homogeneous force law.
Holding the note, masses, losses, brightness and readout fixed, changing initial
speed from `v` to `a*v` and felt scale from `k` to `k/a**1.3` scales the raw
waveform by `a`, including the contact head. Thus a free overall gain prevents
separate identification of speed and felt from one level-normalized note. Fix
one as a calibration convention instead of jointly fitting both. This is an
invariant of this simplified force law, before nonlinear effects or limiting;
it is not a claim about the identifiability of real piano felt.

`CoupledPiano` permits `felt_stiffness_scale=0.01..16`, also per note, so softer
register-dependent contacts can be modeled without output attenuation. The
independent physical-coordinate contact checks include the softer range. This
does not change the default or certify a general piano calibration; the existing
50 ms separation and numerical-resolution guards still apply.
The source requires separation within 50 ms and explicitly rejects unsupported
contact/band configurations. It does not replace Piano or the accepted palette.

Optional `impact_gain=0..1` on contact pianos adds a procedurally generated
nonharmonic attack. A positive synthetic 1 ms pulse excites 48 designed damped
modes below 1350 Hz (and below .42 of the sample rate). No recorded noise or
measured response is used. The component is added after string damping and does
not alter string contact. `impact_decay=.03..0.5` is T60 at 200 Hz;
`impact_color=-1..2`, `impact_velocity_power=0..2` and `impact_delay=0..0.06`
control spectral tilt, MIDI-velocity response and causal delay. All accept note
overrides. The default gain 0 leaves the original audio and RNG stream unchanged.
Enabled tails are declared by the instrument; a note override exceeding that
allocation raises rather than silently truncating. This is an empirical output
component, not an identified or energy-coupled model of the piano mechanism.

`BowedString`, `Violin` and `Cello` accept per-note `pressure` and `bow_noise`,
for example `music.param(pressure=.22, bow_noise=.018)`. Values must be finite
and nonnegative; zero is allowed. They override the corresponding preset fields,
independently of note velocity, vibrato, `body_eq` and amplitude expression.
Here `pressure` controls harmonic roll-off, not a measured physical bow force;
the parameters do not automatically infer pp/ff or certify a realistic dynamic.
The default source is unchanged. These controls belong to the harmonic bowed
source; `ModalBowedString` has its separate physical contact parameters.

Optional `bow_noise_seed` on these three instruments, or per note via
`music.param(bow_noise_seed=41)`, changes only the procedural noise realization.
It accepts an unsigned 64-bit integer. Pitch, amplitude modulation and the caller's
random stream stay fixed, so seeded texture probes can vary independently.
The default `None` preserves the original waveform and random draw order; an
explicit per-note `None` restores that behavior for a configured instrument.
This is a repeatability control, not an automatic improvement in realism.

`bow_noise_highpass` and `bow_noise_lowpass` optionally set noise-only filter
corners in Hz, on the instrument or individual note. For example,
`music.param(bow_noise_highpass=80., bow_noise_lowpass=2500.)` changes the noise
colour without filtering the harmonic source. These causal second-order filters
run before the voice envelope. The highpass default `None` retains the original
body-scaled corner; lowpass `None` adds no filter. A note-level `None` restores
each default. Explicit corners must be finite, at least 10 Hz, and no greater
than 0.45 of the rendering sample rate; they are rejected rather than clamped.
They do not identify a physical friction spectrum or replace the accepted presets.

`partial_gain_points` optionally controls harmonic amplitudes in `BowedString`,
`Violin` and `Cello`, including per-note overrides. Rows contain seconds followed
by linear gains for harmonics 1..H. For example,
`music.param(partial_gain_points=((0., 1., .5), (.4, 1., 1.5)))` changes only the
second harmonic over time. Up to 512 rows and 80 gains per row are accepted;
times must be nonnegative and increasing, gains finite in [0,16]. Gains
interpolate linearly, hold at endpoints, and precede the common voice envelope
and body EQ. Unspecified harmonics stay at gain 1; noise, pitch and random draws
are unchanged. The empty default keeps the original source, and an empty note
override disables an instrument curve. These scalar controls contain no recorded
carriers or phases and do not themselves establish a realistic reference match.

`notes.ears.partial_envelopes(audio, sr, fundamental_hz, partials=20)` estimates
local sinusoid magnitudes by joint Hann-weighted regression on a supplied known
frequency or per-sample trajectory. Explicit modal ratios can replace the integer
partial count for known inharmonic strings. Results include clip-clock times,
amplitudes, residual RMS, design condition and an iid-white-noise coefficient
floor; the floor flag is not a general tone detector. Phases are not returned.
Frequency errors, coloured noise, rapid amplitude changes and unresolved modes
require independent checks. This analysis does not change the existing band-energy
profiles or transcribe unknown notes.

Optional `partial_hz` is a `(samples, partials)` matrix for modes that move
independently: each column is integrated separately, while `partials` keeps the
nominal mode labels. With explicit tracks, all requested partials must be supplied;
they are not truncated by the nominal harmonic frequencies. Actual frequencies
must remain ordered, positive and below `.45*sr`; unresolved bases fail explicitly.
Returned `partial_hz` contains the used frame-centre frequencies, or `None` in the
original shared-fundamental mode. Input tracks are caller-supplied assumptions,
not certified physical modes. The default calculation is unchanged.

`Violin`, `Cello`, `BowedString` and `ModalBowedString` accept optional `body_eq`:
up to eight `(frequency_Hz, Q, gain_dB)` peaking filters, also overridden per note
with `music.param(body_eq=...)`. The empty default preserves the existing sound;
an empty note override disables instrument-level EQ. Frequencies are absolute
and do not scale with body size. The declared release must cover the estimated
filter decay; frequencies outside the output band are rejected. This is effective
output colouring before expression/track processing, not recovered body mechanics.
`examples/reference_violin_strings.py` demonstrates a reference-specific G/A-string
correction; its D/E notes keep the baseline. The general palette is unchanged.

`ModalBowedString()` is a separate experimental nonlinear bow/string source.

Its soft friction curve drives finite string modes through a passive midpoint
contact with a checked globally unique scalar root. A fixed physical modal band
preserves the underlying model across output rates; the minimum integration
factor is 16x and increases when needed. Open strings default to cello tuning;
per-note `open_pitch`, `bow_speed`, `force_ratio`, `bow_position`, `bow_width`
and `stiffness` select geometry and bow controls. Note velocity scales amplitude
independently of bow speed/force. Violin geometry can be specified with
`ModalBowedString(open_pitches=(55,62,69,76), length=.328, tension=55,
body_size=1, program=40)`.

This source supports steady pitches and constant pitch offsets. It has no
time-varying pitch, shared legato state, thermal rosin or body feedback. Its bare
fundamental can sound flat. Optional `pitch_compensation=True` measures its own
three-second synthesized probe and applies a constant sinc playback correction
before the body. Only a scalar correction is cached for each frequency/rate and
physical-control set, independently of requested note duration. Unreliable phase
measurements raise an error. This corrects the mean of a held note; its small
rate change also retimes the source's attack, while note-off stays on the score
clock. The default `False` preserves the original source. Comparisons on eight
cello notes remain mixed, so the existing
`Violin`/`Cello` presets and the accepted palette are unchanged. The body is a
designed minimum-phase filter, with no recorded waves or measured IRs.
The friction curve follows equation 2 of
[Efficient Simulation of the Bowed String in Modal Form](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_14.pdf);
the solver here is independently implemented and does not reproduce that paper's
non-iterative algorithm.

`notes.engine.bowed_body` provides reusable passive string/moving-bridge
mechanics for experimental contact solvers. Positive-mass body modes enter the
mass, stiffness and damping matrices, so bridge motion can feed back into bow
contact. Full non-proportional damping is retained. The midpoint operators use
coupled-mode prewarping and expose the collocated input mobility for independent
work/energy checks. This low-level kernel is separate from instrument presets;
passivity and numerical convergence do not establish acoustic realism.
Optional `bridge_weights` in `mechanics` and `thermal_bowed_string` specify
signed body-mode shapes at the selected string's bridge port. Endpoint lift,
mass coupling, stiffness and applied forces all use the same weights, preserving
the work identity. This permits different string ports to excite a fixed body
and radiation observer differently. Unit weights retain the original model;
these coefficients are not a measured or calibrated bridge geometry.

`notes.engine.thermal_bow.thermal_bowed_string` is a low-level experimental
source using tangential elastic/plastic contact and a positive thermal network.
Rosin strength depends on its temperature history, enabling sustained stick-slip
motion in regimes where the isothermal control decays. Contact strain energy,
plastic heat, conduction/transport losses, and numerical loss are tracked
separately. Optional traces expose each body's velocity and acceleration;
the returned endpoint-gradient signal is not calibrated acoustic pressure.
The heat ladder and finite contact compliance are independent approximations
inspired by
[Bowed String Simulation Using a Thermal Friction Model](https://euphonics.org/wp-content/uploads/2022/03/Thermal_bowing.pdf),
not an exact implementation of that paper. No recorded excitation or measured
response is used. This source does not change the existing instrument presets
and has not established realism across notes, dynamics, or instruments.

Optional `roughness`, `grain_length` (metres), and `grain_seed` add a designed
spatial variation of its yield strength. The material coordinate advances only
with plastic sliding, and revisiting it yields the same value. No noise is mixed
into the output. `roughness=0` preserves the previous wave. Diagnostics expose
the final material coordinate and largest step relative to a grain cell, so
numerical resolution can be checked separately from acoustic similarity.
This controls texture, not a calibrated realistic preset: a two-note MF fit
overstated the residual energy on another A-string note by 14–15 dB even while
MERT increased. The accepted `Violin` and `Cello` are unchanged.

Run `notes render examples/instrument_palette.py -o palette --spectrogram` for a
palette including electric/acoustic guitars, bass and rock drums. The example's
`build_song(name)` also exposes each instrument separately.

`Piano(model="soundboard")` adds a hammer-position comb and two-stage partial
losses with a longer upper-harmonic tail; the default `model="legacy"` remains
available unchanged. `ElectricGuitar(amplified=False)` exposes the direct pickup
signal without amp/cabinet filtering. `AcousticGuitar(prompt_decay=0.45,
velocity_brightness=0.4)` enables two-stage string decay and velocity-dependent
picking hardness. These variants are experimental: listening rejected the brighter piano and guitar
settings. The instrument palette uses the baseline settings; neither piano nor clean
guitars have established realism.

`Piano(model="hammer")` is an experimental procedural alternative: a compression-only
felt contact couples a moving hammer to damped string modes. The contact runs at
at least 176.4 kHz; free ringing is evaluated analytically. Velocity sets the impact
speed rather than merely scaling a predefined spectrum. It is a simplified model,
not a calibrated acoustic piano. This does not change the default palette.

`Piano(model="hammer_damped")` tests a register-dependent modal readout and
frequency-dependent losses above A2, preserving the A2-and-below hammer source.
It addresses overtone dominance measured in the first hammer model; it is not a
full soundboard simulation or an established realism improvement.

The contact models optionally add a procedural nonharmonic attack with
`impact_gain` (default zero), `impact_decay` (T60 at 200 Hz), `impact_color`,
`impact_velocity_power` and `impact_delay` (seconds). Each accepts a per-note
override; the instrument's configured sustain/release must accommodate its tail.
The designed modal response is excited by a seeded synthetic positive pulse,
after string damping. No recorded noise or measured impulse response is loaded,
and this component is not an energy-coupled or identified piano-action model.
`examples/piano_impact.py` demonstrates the optional contact/impact profile.
Frozen controls improved several diagnostics on new rapid Disklavier sequences
at quiet and loud dynamics, but isolated-note regressions and incorrect quiet
levels remain. The listening-approved palette is not replaced.

`examples/quiet_piano_response.py` demonstrates a lower-velocity hammer-speed
taper using note controls, plus an energy-matched `source_gain_db` comparison.
Its `build_song(mode)` accepts `baseline`, `balanced`, or `gain_control`.
The gain is computed from procedural string previews, without reference audio.
On fresh shorter-gate P/M performances the balanced variant reduced register
contrast MAE from 6.50 to 2.07 dB, but quiet-fragment MERT improved only 4/8;
the calibration knee and near-bound taper remain experimental, not a new preset.

The listening-approved piano palette uses `Piano(model="hammer_damped")`.
`attack_presence=0.6` optionally lifts upper-register partials only during the onset
(time constant 60 ms); C4 and below and the late tail retain the original source.
The default is zero, preserving the accepted timbre. Listening found no noticeable
difference with 0.6: the raw E5 RMS change was only +0.27 dB in the first 60 ms.
This option is not an established improvement or a pedal control. A stronger local
attack experiment increased reference-spectrum errors on A4 and E5, so it was not
added to the instrument. The approved preset remains unchanged.

`Piano(model="hammer_damped", hammer_mass_scale=1.7)` changes the nonlinear
contact by multiplying the moving hammer's mass. The range is 0.25..4, default 1;
`note.param(hammer_mass_scale=...)` overrides individual impacts. Both hammer
models support this control; legacy and soundboard ignore it. It changes contact excitation
and therefore needs loudness matching when comparing timbres. It is not a pedal
or output-brightness control. A frozen 1.7x candidate improved the MERT diagnostic
on four of five new upper-register recordings, but harmonic error increased on G5
and the C6 embedding slightly regressed. It remains an optional experiment; the
listening-approved preset is unchanged.

Per-note `stiffness=...` overrides inharmonicity in the two contact models,
allowing known string properties to be set without altering every key. The
limited `examples/piano_contact_study.py` compares A4/D5/E5 contact settings;
a G5 regression prevents treating them as a universal treble preset.

Optional `notes[quality]` adds Audiobox aesthetic diagnostics. Setup is explicit:

```python
from notes.ears import AestheticsEar, fetch_aesthetics

cache = fetch_aesthetics()  # verified official source/license and safetensors, outside the repo
quality = AestheticsEar(cache)  # local-only load, CPU FP32 by default
scores = quality.score_audio(audio, sample_rate)  # CE, CU, PC, PQ
```

Source and weights are pinned and hash-checked; the backend is imported under a
private namespace. Loading never implicitly downloads or installs dependencies.
The default frontend downmixes to mono, normalizes to -24 LUFS without clipping,
resamples to 16 kHz and uses the official 10-second masks and weighted pooling.
It rejects silence/DC. These are aesthetic/technical scores, not a realism
certificate: the known piano preference passed 4/4, but quantization controls
failed on violin and one distorted guitar. A sine can score highly. Keep
reference similarity and physical errors alongside them; when copying a sound,
compare CE to the reference rather than maximizing enjoyment. Upstream:
[Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound](https://arxiv.org/pdf/2502.05139),
[official source and license](https://github.com/facebookresearch/audiobox-aesthetics).

Known-note reference diagnostics are available in `notes.ears.note_profile`:
`profile_note(mono, sr, expected_f0)` measures tuning, harmonic envelopes and attack;
`compare_profiles(reference, candidate)` separates harmonic, envelope, pitch and
attack errors while removing an overall gain difference. It requires an isolated
note with a known pitch and never produces a realism/approval label. Reference
recordings remain outside the synthesizer and the repository.

`notes.ears.harmonic_shape_distance(reference_profile, candidate_profile,
start=.4, end=1.8)` compares relative partial energies after removing each frame's
overall gain. Unlike the full profile loss, it separates harmonic shape from
crescendos and overall decay. The onset-relative window, audible union and
floor are explicit; added harmonics are included. Use it alongside envelope,
pitch and broadband measurements: it excludes phase/nonharmonic noise and does
not certify realism or numerical convergence by itself.

`notes.ears.known_note_tuning(audio, sr, expected_hz)` refines mean tuning of a
stable isolated note with a known nominal pitch. It fits demodulated phase in an
explicit window and checks agreement of detected harmonics. Check `reliable`
before using its `f0`: silence, noise, wrong priors, bends or missing fundamentals
can invalidate the estimate. The default window is 0.15–0.75 s and requires 100 ms
edge margins. This is a calibration tool, not an unknown-melody transcriber or a
replacement for the existing YIN path. It helps prevent a distorted reference's
pitch-estimator bias from being mistaken for a timbre mismatch.

For detuned high notes, `harmonic_bandwidth=True` scales the demodulation band
with pitch and harmonic number, while capping it below the neighboring partial.
The legacy fixed narrow band can suppress a real detuned upper partial and make
its phase estimate follow noise. The new mode still rejects wrong priors,
missing fundamentals and unstable phase; check `reliable`. Its default is False
to preserve existing calibrations. A moving tone can have a clear note label
while failing the stricter constant-frequency test.

`notes.ears.string_inharmonicity(mono, sr, expected_hz)` estimates a plucked
string's dimensionless stiffness coefficient from simultaneous partial phases.
It subtracts the shared fundamental phase before fitting, avoiding a common
pitch glide being mistaken for stiffness. The result includes per-partial
diagnostics, residual error and `reliable`; coupled modes, unisons, missing
fundamentals and departures from the simple string law can invalidate it.
This is an analysis tool; it never inserts reference audio into a render.

`NylonGuitar()` is an experimental procedural classical-guitar variant using a
rounded triangular-displacement waveguide readout. It supports the acoustic
instrument's body and twelve-string controls. `AcousticGuitar(fretting=True)` and
the nylon preset map a fixed physical pick location onto the shortened string;
per-note `string=1..6` selects the open string. These additions preserve the
original steel/electric/bass excitation defaults. Realism is not yet established.
The displacement construction follows the ideal-pluck formulation in
[Julius O. Smith, Physical Audio Signal Processing: The Ideal Plucked String](https://www.dsprelated.com/freebooks/pasp/Ideal_Plucked_String.html).

The waveguide instruments (`AcousticGuitar`, `NylonGuitar`, `ElectricGuitar`,
`BassGuitar`) accept `dispersion_fit="bracketed"`. This fixes an inner resonance
search that could abort coefficient fitting and silently disable string stiffness.
The original two-allpass approximation and its reference near 1.5 kHz remain;
this is a numerical correction, not a realism guarantee. Default `"legacy"`
preserves the original waveforms.

`AcousticGuitar(model="coupled")` is a separate experimental string/body model:
positive-mass designed body resonances absorb string energy as well as color its
output. Reactive bridge loading is compensated to retain the requested fundamental.
`coupling` controls mobility and `radiation=0..1` moves from bridge velocity toward
acceleration. `body_size`, `diffuse`, the decay and pick controls remain available;
`body` and `excitation` select behavior only in the original `model="waveguide"`.
Both models accept per-note `pick_position` and `pick_hardness` overrides.

`AcousticGuitar(model="modal")` renders analytic pluck-force modes with zero
initial velocity and a statically preloaded body. Pitch curves change the string
before body filtering, preserving the body's resonant frequencies. The body
tail is included in each voice; string losses are prescribed rather than
coupled to body mobility. Six/twelve strings, explicit string selection,
per-note pick controls and the existing pitch contours remain available.
`excitation` and `dispersion_fit` are unused in this mode. The example
`examples/reference_modal_acoustic.py` contains a fitted candidate; on two
annotated performances it reduced the physical objective in all six regions
and increased MERT in five, with one regression. Several isolated notes also
regressed, so the original waveguide remains the default. Excitation and body
responses are generated mathematically without recorded samples or measured IRs.

Optional `force_gains=(1.0, 0.6, 1.2)` changes the first modal force harmonics
in `AcousticGuitar(model="modal")`, including the summed static load used to
initialize the designed body. It scales both polarization branches; unspecified
gains remain one. Up to 80 gains in 0..16 may be set on the instrument or a note.
Other models reject nonempty values. This changes the prescribed initial force
shape, unlike output EQ or `ModalString.partial_gains` readout coefficients.
No recorded excitation or phase is loaded. Conditional gains improved four
isolated-note fits, but interpolating them by pitch did not transfer reliably;
they do not identify pluck geometry or replace the accepted guitar palette.

For calibration across output rates, opt into `body_sampling="fixed"` on
waveguide/modal acoustic guitars, or `Body(sampling="fixed")`. This converts one
44.1 kHz procedural response instead of drawing a different dense noise texture
at each rate. The complete dry/wet response and phase choice share a causal
132/44100-second delay (about 3 ms, even at mix=0) and a 257 ms tail allocation.
The conversion preserves filter gain rather than renormalizing each rate.
Rates below 4 kHz and the coupled mechanical model are unsupported in this mode.
The default `"native"` keeps previous renders. This fixes cross-rate consistency;
it does not establish a more realistic body or validate old fitted coefficients.

Modal acoustic guitars also accept `force_gain_points`, rows of
`(seconds, gain1, gain2, ...)`, on the instrument or individual notes. Up to 512
increasing nonnegative times and 80 gains per row in 0..16 are supported.
For example, `force_gain_points=((0., 0.6, 1.2), (0.2, 1., 0.8))` changes the
first two modal forces before body filtering. Endpoints hold and unspecified
modes stay one. These gains multiply `force_gains` and apply to both polarization
branches and the initial body load. Their clock remains in output seconds
through bends/slides; pitch interpolation reads force changes, preserving the
preloaded state. Curves stop driving the body when the note-off gate closes.
Empty/identity curves retain the previous source. Waveguide/coupled models
reject nonempty curves. These prescribed force envelopes are empirical controls,
not an identified passive contact or a universally realistic preset; neither
reference phases nor recorded carriers are loaded.

The waveguide/modal acoustic body can optionally use `body_phase="minimum"`
(`Body(phase="minimum")` as an effect). This moves energy toward the start of the
mathematically designed body response, approximately retaining its magnitude
spectrum and the original tail allocation. The default `"designed"` preserves
the previous signal exactly. On two GuitarSet performances, this is a controlled
phase experiment, not an established timbre improvement: auxiliary CLAP similarity
increased on two regions and decreased on the fitting region. It does not correct
an annotation-to-waveform onset offset. See
[Minimum-Phase Filters](https://www.dsprelated.com/freebooks/filters/Minimum_Phase_Filters.html).

The coupled example `examples/reference_coupled_acoustic.py` reduced the mean
diagnostic error on nine unseen Seagull recordings by about 19%, with no per-case
regression at either of two seeds. Low E2 still regresses on the development set.
CLAP similarity decreased on all nine recordings, so the evidence is conflicting
and the example is not adopted as a quality improvement. The frequency-domain model approximates
bends by resampling and omits mutual coupling between separate strings. Existing
acoustic and nylon defaults remain unchanged. The boundary equations follow
[Euphonics: Coupling a string to the instrument body](https://euphonics.org/5-1-1-coupling-a-string-to-the-instrument-body/).

The same known-note audit is exposed by the CLI:
`notes audit-note reference.wav render.wav --pitch A3 -o report.json`.
It does not infer a part from a mixture and does not label the instrument realistic.
For moving known notes, `profile_note(..., partial_mode="tracked")` or
`notes audit-note ... --partial-mode tracked` follows the fundamental within the
known-pitch prior and integrates wider, non-overlapping bands around the moving
partials. Fractional FFT-bin overlap keeps their width continuous. Unsupported
fundamental frames are flagged; this is not melody tracking or certified tuning.
These band energies include more nearby noise than the original measurement.
Both notes must use the same mode; tracked reports use metric version 3, while
the default fixed mode retains its original numerical behaviour and version 2.

For nearby resonances, `profile_note(..., analysis_window=.12)` or
`notes audit-note ... --analysis-window .12` explicitly selects a longer Hann
window instead of the original six-cycle window. It must cover at least three
nominal cycles and fit the post-onset audio, within 0.01..0.5 seconds. This improves
frequency resolution while smoothing spectral changes in time; RMS attack timing
is still measured separately. Both profiles must use the same window. Explicit
windows use metric versions 4 (fixed bands) or 5 (tracked bands); defaults are
numerically unchanged. The anchor uses the median of supported local peaks within
50 cents at complete-window centres in the early region. If fewer than two are
supported, `frequency_supported` is false: the supplied frequency is only a band
anchor, and comparisons return `pitch_cents=null` with `pitch_supported=false`.
The CLI also reports missing frequency estimates as null. Even supported peaks
are not a unique perceived-pitch certificate when several modes are present.
Explicit-window profile-set audits keep cases with unsupported frequencies from
passing the no-regression guard, listing them separately from measured regressions.

`notes audit-note ... --peak-window .02 .15 --peak-window .4 1` also reports
independent dominant peaks in a broader half-to-one-and-a-half-fundamental band.
These windows start after each automatically detected onset. The report includes
per-window frequency, contrast, relative level and support, exposing changes of
modal dominance. It does not identify a fundamental or prove that the same mode
was followed. Inspect these separately from the scalar profile pitch.

`notes.ears.interharmonic_energy` measures a separate residual-band axis for
sustained known-pitch notes. It includes FM leakage and inharmonic energy as
well as noise; quiet windows and inaudible upper bands are flagged. Optional
`notes audit-note ... --texture-window 1 2.5 --texture-band 1000 10000`
records it alongside the existing profile measurements.
`fit_roughness(reference, render, sr, f0)` renders a baseline, seeded probes and
one new-seed verification. The callback takes `(roughness, material_seed)` and
must keep the notes, timing and other controls fixed. A sufficient or excessive
baseline gets no added grain. Overshooting or regressing verification retains
the baseline, with the attempted result recorded in history. This fits one
conditional texture axis and does not certify the instrument's realism.

`notes compare reference.wav render.wav --mert` optionally adds audio-only
MERT-95M similarity alongside the existing measurements. `--mert-path PATH` uses
an existing pinned checkpoint offline; without it, the three checkpoint data files
are downloaded. `notes.ears.MertEar` exposes the same embeddings in Python.
It uses the last layer with temporal mean pooling and overlapping windows of at
most 10 seconds, and executes no model-repository Python. The regular comparison
path does not load MERT or require its weights.

Time alignment may otherwise cut an early attack or a late tail. MERT comparisons
therefore retain padded edge context while leaving the ordinary diagnostic interval
unchanged. In Python, use `compare(..., preserve_context=True)` followed by
`ear.similarity(comparison)`; custom fit objectives receive that context automatically.
`notes compare --preserve-context` also keeps these edges in A/B files without
loading an encoder; `--mert` enables it automatically.

For isolated short hits, `--mert-trim` or `trim_edges=True` uses a fixed -40 dB RMS
edge gate with 10/20 ms margins and at least 100 ms of input. Interior pauses remain.
This is opt-in: quiet tails can dominate mean pooling on percussion. The JSON records
the mode, edge-context duration and analysis revision; scores from different modes
are not interchangeable. Low C2 cello controls remain unresolved, so MERT must not
be used as an unrestricted judge across instruments.

The fixed representation preferred the listening-approved piano over the rejected
hammer version in six register/dynamic comparisons and detected seven +2-semitone
controls. This is a small calibration, not a realism certificate or validation of
other instrument families. The JSON explicitly marks the score diagnostic-only.
Model: [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training](https://arxiv.org/pdf/2306.00107).

`fit_reference(..., objective=callback)` optionally scores the complete `Comparison`
with aligned reference/render audio and the ordinary diagnostic report. The existing
objective remains the default. A CLI fit script may likewise define `objective(comp)`
and `objective_name`; the name is saved with its result. For MERT, use
`ear.similarity(comp)`: alignment-dependent padding means a cached reference
vector is not generally interchangeable between trials. This retains the independent timing,
spectrum and envelope measurements. A better embedding score alone is not an
acceptance criterion for the generated instrument.

For a reference with GuitarSet JAMS annotations, `notes.ears.guitarset.guitarset_music`
imports the performance as Music to be played at BPM 60. It preserves original
onsets/durations, maps string indices to `string=1..6`, and encodes fractional tuning
and voiced contours as `pitch_points` around the nominal fret pitch. Contour note
indices prevent adjacent notes from leaking into each other's pitch. An optional
`string=1..6` selects one string without shortening the timeline. Confidence is
never used as velocity; velocity is an explicit argument. Known erroneous files
are rejected, and voiced zero-frequency holes are reported and interpolated within
each note. Import the whole performance before cropping rendered audio, so held
notes are not retriggered at a crop boundary. Only musical control data enter the
renderer; recorded audio remains an evaluation input. Annotation fidelity does not
establish the realism of the resulting timbre.

`notes.ears.idmt_music("annotation.xml", velocity=.7)` imports the supported
picked/muted, normal-expression subset of [IDMT-SMT-Guitar](https://zenodo.org/records/7544110)
at BPM 60. Decimal XML times become exact rational beats; source strings 1..6
(low to high) become DSL strings 6..1. Pitch/fret/tuning inconsistencies and
unsupported articulations raise. Acoustic offsets become approximate gates;
they are not measured finger-release times, and no audio latency is corrected.
Only annotation data are read, never the recorded audio. For a direct-source
comparison, bind the result to `ElectricGuitar(amplified=False, fretting=True)`.

Muted electric-guitar notes accept per-note losses/excitation controls:
`Note("D3").param(palm_mute=True, mute_decay=1.2, mute_brightness=.6)`.
`mute_decay` is positive seconds and `mute_brightness` is 0..1; both require
`palm_mute=True`. They override the existing instrument controls for that note,
without changing the defaults or modeling a physical palm/string contact.
Varying decay does not fix a missing or mistimed attack.

Note-audit metric version 2 uses the union of audible reference and candidate
partials, so synthesized extra harmonics are penalized. A separate full-band
spectral error also catches energy outside the analysed harmonic range. Version 1
reference-only masked scores must not be used to claim a timbre improvement.
The finger-bass example passes the diagnostic no-regression guard on twelve
recorded E1-D#2 notes, including eight unseen notes and two synthesis seeds.
Other registers/techniques remain unverified. The nylon fit is not transferable
unchanged to steel-string guitars.

`audit_profile_set(references, baseline, candidate)` reports regressions for every
case, including when the mean improves. All case IDs must match. Its no-regression
flag concerns diagnostic features only, not perceptual realism.

`RockKit(kick_damping=..., kick_body=...)` exposes frequency-dependent head losses
and the second-head share of the kick model. Defaults preserve the existing kit;
these are procedural modal parameters, not recorded samples.

`RockKit` also accepts absolute per-note `tune` (semitones) and `decay` overrides,
for example `Note(51, 0.1).param(tune=-6, decay=2)`. One kit track can therefore
tune each drum/cymbal independently. Tune/decay are unsupported on rim and pedal
hat; clap supports decay only. Invalid or ignored controls raise an error.
Longer per-note decays extend the renderer's allocation through
`Instrument.release_time_for(params)`; note times and gates remain unchanged.
`sound="hat"/"open_hat"/"pedal_hat"` aliases now choke by their actual sound in
both kits, even when attached to another MIDI pitch. A kick on a hat-numbered
placeholder no longer chokes an open hat. Default voices retain their waveform.

Tom pitch sag is independently configurable with `tom_glide` (nonnegative
initial fractional lift), `tom_glide_time` (positive seconds), and
`tom_velocity_glide` (0..1). The effective lift is `tom_glide` times the blend
between fixed strength and nominal velocity. Zero velocity coupling keeps the
pitch gesture fixed while impact brightness and output level still follow
velocity. Both heads use the time constant; the secondary lift retains its
original proportion. `decay` still controls amplitude damping separately.
Constructor controls affect only toms; per-note overrides on other pieces raise.
Defaults .22/.08/1 preserve the old source. This is prescribed calibration,
not identified hit force, a coupled-head model, or a new accepted kit preset.

`tom_modes=7..512` optionally extends the legacy seven head modes with higher
ideal circular-membrane eigenfrequencies computed from Bessel roots.
The count applies to the batter head; the secondary head retains three modes.
`tom_damping>=0` controls the frequency-dependent loss exponent of both heads,
separately from fundamental decay and pitch sag. The old seven modes, stick and
noise keep their original random stream; extra modes use a separate generated,
count-independent pool. Modes above .44 of the output rate at maximum glide
are discarded without moving the other frequencies. Constructor/per-note
controls affect toms only. Defaults 7/1.6 preserve the original kit. The added
amplitudes, spatial readout and losses are empirical; this is not a coupled-head
or measured physical model and does not replace the accepted kit palette.

`notes.ears.band_peak_trajectory(audio, sr, band=(60, 220), windows=...)` measures
dominant low-band peaks across specified clip windows, with contrast/relative
level support and no phase output. `compare_peak_trajectories(reference, candidate)`
rejects missing or mismatched supported target peaks, including moving ones.
This can prevent a tom fit from matching an upper mode while losing the lower
one despite a lower broadband loss. Supply isolated-onset mono audio covering
all windows; no automatic alignment occurs. The descriptor identifies neither
the fundamental nor a physical mode, and zero-padding is not extra resolution.
A passing peak guard does not certify the rest of the spectrum or realism.

`examples/reference_rock_kit.py` combines the reference-calibrated kick and a
procedurally generated room with the unchanged baseline snare and hats. The
alternative snare fit remains experimental rather than replacing the baseline.

With `ElectricGuitar(fretting=True)`, an explicit per-note `string=1..6` now
selects the string used to compute the fret length, pick and pickup geometry.
Previously it only affected wound/plain damping and the automatic string still
set the geometry. The default unfretted renderer remains unchanged.

Electric-guitar notes also accept `pick_position` and `pick_hardness` through
`Music.param(...)`. These physical controls override the instrument defaults
before fretting, mute/legato and velocity-brightness modifiers. They store scalar
performance parameters; no reference waveform is used by the synthesizer.

`ElectricGuitar(excitation_gains=(1., .5, 1.5))` optionally shapes successive
non-DC DFT bins of the generated pluck period before the feedback loop. Gains
are in 0..16, at most 80; unlisted bins remain unchanged and unavailable bins
are ignored. Generated bin phases are retained and no peak normalization
follows shaping. Dispersion/fractional delay means these are not exact
resonant-partial gains. The same control is available on `steel_string` and on
individual electric-guitar notes; an empty note tuple resets inherited gains.
Empty/unity settings preserve the previous samples and RNG stream exactly.

`excitation_noise=0..1` independently controls the initial spatial noise,
default .1, on the electric guitar and its notes. It is separate from the short
`pick_noise` click. Zero still consumes the same RNG draws, so subsequent note
humanization is not reseeded. These controls permit conditional reference fits
with mathematical excitation; no recorded period, phase or impulse response
is loaded by the synthesizer. They do not establish a general realistic preset.

`ElectricGuitar(pluck_sampling="spectral")` evaluates the continuous triangular
pluck (or its velocity profile) as a finite Fourier series, rather than rounding
the pick coordinate to a sample. This avoids accidental grid notches and abrupt
changes when the pick crosses a sample boundary. The physical coordinate can
still create spectral zeros. Existing lowpass rounding, peak normalization and
noise draws remain; the optional method does not certify a realistic timbre.
It is also available per note and on `steel_string`; `"grid"` is the unchanged
default and can reset an inherited spectral setting on an individual note.
The spectral method requires at least three samples in the string delay.

`ElectricGuitar(partial_t60_scales=(1., .5, .2))` independently scales the
waveguide's modal decay times: here the fundamental keeps its lifetime, while
the second and third modes decay twice and five times faster. Positive factors
retain each mode's frequency and mathematically generated state at the end of
the excitation period. The period itself is unchanged before subsequent
pickup/filter/pitch readout; the new losses apply after that initialization.
The factor applies separately to both polarisations, preserving their lifetime
ratio. Unlisted modes stay unchanged; at most 80 factors are accepted and modes
above the available oscillatory band are ignored. This is also a note override
and a `steel_string` option. Empty/unity factors preserve old samples and RNG
exactly; an empty note tuple resets inherited factors. The optional path needs
a contractive loop and resolved distinct modes. Times are in the unbent source
clock, before pitch readout/release. It controls stable modal damping, without
certifying real-string mechanics, an unchanged finite-window attack, or realism.

`ElectricGuitar(partial_pitch_cents=(0., 3., -1.))` retunes resolved modes by
small static offsets after the generated excitation period: here the first
mode stays put, the second moves up three cents and the third down one cent.
It retains the synthesizer's generated modal states and native lifetimes;
`partial_t60_scales` can change lifetimes simultaneously. At most 80 finite
offsets in [-100, 100] are accepted. Both polarisations receive the same cents
offset; crossing neighbouring modes or Nyquist is rejected. Unlisted or
unavailable modes stay unchanged. Constructor, note override and empty reset
are supported; empty/zero offsets retain previous samples/RNG exactly,
including the decay-only path. This is a source-clock modal parameter, not a
certificate that a measured phase-frequency trajectory is one physical mode.

`notes.ears.fit_partial_pitch_cents(paired_measurements, current=existing_offsets)`
pools paired `(reference, model)` results of `string_inharmonicity` for the same
known anchor note and matching windows. It uses individually supported partial
ratios even when a global stiffness law does not fit. By default at least two
observations, at most 1.5 cents total span, strong/covered components and low phase
residual are required. `numbers=(5,)` restricts correction to the fifth component.
The result's `offsets` are candidate `partial_pitch_cents` settings; `deltas`,
support counts, spans, acceptance flags and rejection reasons remain visible.
The fundamental and unsupported settings are preserved. Large deltas and final
offsets outside [-100,100] are rejected rather than clipped. This is conditional
effective-frequency calibration, not pitch transcription, recorded-phase transfer
or a cross-register material table. Validate native root ordering/Nyquist and the
complete sound on held windows separately.

`ElectricGuitar` and `BassGuitar` also accept per-note `stiffness`, for example
`Note("E4").param(stiffness=0.0001)`. This is the effective dispersion B of the
vibrating string; it is not scaled again by fret geometry. Values must be finite
and nonnegative. Omitting it preserves the instrument's existing global B.
This lets known-note inharmonicity measurements reach individual notes without
creating a separate track for every stiffness value; it does not establish realism.
For measured-B calibration, select `dispersion_fit="bracketed"` explicitly: the
legacy search can disable dispersion and make different requested B values yield
the same waveform. Both modes retain the existing approximate two-allpass model.

`ElectricGuitar(fretting=True, stiffness_reference="open")` interprets its global
`stiffness` as the open-string B and scales it by the inverse squared vibrating
length ratio for each fret. Default `stiffness_reference="vibrating"` keeps the
original fixed coefficient. Per-note `stiffness` always remains an absolute B of
the sounding length and wins over this scaling. Open mode requires fretting and
a finite nonnegative constructor B. Choose a coefficient appropriate to the
string and use the existing `dispersion_fit="bracketed"` for calibrated values.
This is a fixed-tension approximation; it does not infer string material or
simulate tension changes during bends. The length dependence follows the
[stiff-string relation described by Daniel Russell](https://www.acs.psu.edu/drussell/Demos/Stiffness-Inharmonicity/Stiffness-B.html).

For amplified guitars, instrument `level` is an excitation gain before the amp;
changing it changes saturation even after loudness matching. Use `Track.gain` or
master loudness for output-volume changes that preserve the amp operating point.

For crescendo/diminuendo inside a held note, use
`Note("A3", dur=4).param(expression_points=((0, .2), (.6, 1), (2.5, .35)))`.
Points are seconds from onset and linear gains from 0 to 1; values interpolate
linearly and hold before/after the endpoints, including release. The audio renderer
applies this to every instrument before its inserts and amp, independently of
velocity and pitch. It controls expression gain, not bow pressure or excitation
brightness. Omitting it preserves old renders; MIDI export does not emit this curve.
`notes.ears.infer_expression_points(reference_profile, candidate_profile)` estimates
ten coarse gain points from RMS envelopes for a known-note reconstruction. It keeps
the original onset clock, normalizes overall gain and returns only scalar controls.
For short notes, pass a shorter `times=(0, ..., end)` schedule. This fits performance
amplitude, not timbre or a general instrument preset.

`ElectricGuitar(pickup_position=.06, pickup_width=.04)` opts into a spatial
magnetic pickup with fractional delays and a finite uniform sensing aperture.
`coil_spacing` adds two equal in-phase coils. These dimensions are fractions of
the open string when `fretting=True`, otherwise of its vibrating length; the
whole aperture must fit inside the vibrating string. Fundamental gain is
normalized, so these controls primarily change harmonic balance. They are distinct
from `pick_position`, which describes where the plectrum strikes.

`pickup_cutoff` and `pickup_q` independently override the electrical lowpass.
Their default None values retain the existing pickup filter; zero widths/spacings
and no explicit position retain the original integer-delay point pickup exactly.
All filters are generated mathematically, without measured IRs or sample playback.
The model assumes a uniform magnetic aperture and nearly harmonic propagation.
Spatial derivations: [pickup position and width](https://till.com/articles/PickupResponse/),
[equal pickup mixing](https://till.com/articles/PickupMixing/).

`examples/reference_direct_guitar.py` exposes a frozen DI candidate and an optional
baseline. The frozen candidate improved note-profile diagnostics on twelve other
note/dynamic cases with two synthesis seeds at both 22.05 and 44.1 kHz, covering
all six strings. MERT regressed on D3,
so this is an experimental reference fit, not a universal clean-guitar preset or
an established realism result. The default guitar remains unchanged.

`examples/reference_rock_guitar.py` runs the spatial source through the existing
amp and designed cabinet, then applies six optional mathematical peak filters.
Their scalar gains were fitted from three real DI/processed note pairs,
including a shared trough near 1.2 kHz. Only scalar filter settings enter the
example; it loads no recordings or measured impulse responses. `build_song(baseline=True)`
keeps the same source, part and amplifier without the additional filters.
The correction reduced profile error on other FSBS notes, but some notes regress
under MERT; it remains a reference-specific experiment. The stored DI files have
nearly equal full-scale peaks; their original physical input gains and exact
synchronous pairing remain unknown.
On A2, the correction improves MERT for the recorded-DI analysis chain
(0.846 to 0.914) but worsens it for synthetic input under two independently
measured tuning anchors. The synthetic source/amp interaction remains unresolved;
this is not a universal rock-guitar preset.
Source: [FreePats FSBS distorted electric guitar](https://freepats.zenvoid.org/ElectricGuitar/distorted-electric-guitar.html).

`ModalString()` is an experimental direct electromagnetic string source. A finite
mass/stiffness/damping system includes a moving bridge and exact non-proportional
damping. It uses no recorded excitation, measured IR, reference phases or wavetable.
`release_motion=-1..1` assigns a signed fraction of initial mechanical energy to
velocity instead of displacement, preserving total initial energy. Its default 0
releases a held pluck; +/-1 begins at zero deflection with opposite velocities.
Velocity scales energy independently. `pluck_force` sets the model's effective
pull; `level` converts mechanical velocity to DSP amplitude.

Optional `partial_gains=(1.0, 0.7, 1.3)` scales the readout of the first physical
string coordinates (up to 80 gains in 0..16); remaining coordinates keep gain one.
The same parameter is available per note and in the low-level `modal_string` and
`palm_string` functions. It leaves mechanical matrices, initial energy and the
moving-bridge readout unchanged. These are empirical static spectral controls,
not recorded carriers, identified pickup geometry or a new accepted preset.
Empty or identity gains preserve the previous output. Reference-conditioned
gains must be validated on other notes/strikes before claiming transfer.

Optional `ModalString(partial_t60s=(12.0, None, 2.0))` sets isolated string-coordinate
amplitude decay times in seconds, also per note. `None` or missing entries retain
the sustain/treble loss law; an empty note override restores that law for all modes.
Up to 80 finite positive times are supported. The control changes positive
mechanical damping before body/contact coupling, preserving passivity and the
initial state. Coupled poles can have different lifetimes, and the explicit note
gate/release still applies. This calibration option does not establish a general
realistic preset; no measured phases, carriers or impulse responses are loaded.
Explicit entries replace scalar muting losses too; viscous palm damping remains
an additional mechanical loss when that contact is enabled.

Optional `partial_t60_steps=((0.64, (8.0, None, 2.0)),)` on `ModalString` or a
note replaces the loss tuple at each strictly increasing onset-relative time.
Up to 64 rows are supported. Each row is a complete replacement: `None` or
missing modes use intrinsic sustain/treble losses, and `()` resets every mode.
Before the first step, `partial_t60s` applies; a row at zero sets initial damping.
The source propagates mechanical displacement/velocity and release-filter memory
through each boundary, changing positive damping only. M/K, initial pluck and
pickup stay fixed; changing loss does not inject mechanical energy or restart
the note. Steps use the internal observation grid, with no recorded carriers.
Nontrivial schedules use the physical-band contact solver, with zero contact
resistance when no palm contact was requested; specify that same solver for a
controlled static comparison. Empty/redundant schedules preserve the previous
path. These are effective piecewise losses, not an identified damping mechanism
or a general realistic preset. Pitch curves remain unsupported by `ModalString`.

`notes.ears.fit_partial_gains(reference_envelopes, candidate_envelopes)` estimates
static shape controls for corresponding mode indices. First call
`partial_envelopes` on each signal using its own supported frequencies/ratios;
forcing the recording onto the synthesizer's frequencies can bias magnitudes.
The measurements need equal mode counts and identical frame clocks. The result
contains a tuple of bounded `gains` plus support/clipping diagnostics, suitable
for `ModalString(partial_gains=fit.gains)` or a note's `partial_gains` parameter.
It fits normalized spectral shape, not loudness, tuning, recorded phases or a
transferable preset. Unsupported reference estimates use a white-noise bound,
which is not a general noise classifier.

For a dry voice that is affine in its excitation controls,
`notes.ears.excitation_basis(render, controls=12)` builds generated constant/unit
responses and checks a non-unity probe. The callback must use the same note,
losses, tuning, sample count and identically seeded fresh RNG every time. Return
floating mono audio before master normalization, limiting and amplification,
including the linear pickup/insert filters that the actual renderer will apply.
Nonlinear or nondeterministic callbacks fail the probe. For example, render an
`ElectricGuitar(amplified=False)` with the supplied `excitation_gains` tuple.

```python
from notes.ears import excitation_basis, fit_excitation_gains

basis = excitation_basis(render_gains, controls=12)
fit = fit_excitation_gains(
    reference_mono, basis, 44100,
    reference_hz=reference_partial_hz, model_hz=model_partial_hz,
    start=0.14, end=0.65,
)
# Apply numeric controls through the same instrument or note's excitation_gains.
```

The two frequency lists must label corresponding supported components on each
signal's own frequencies. Exact generated band-power quadratic forms avoid
re-rendering during fitting; the recording supplies magnitudes only. One early
level is retained, with explicit fit/normalization windows on the original clock.
The result exposes selected gains, cost, initial cost, optimizer convergence,
evaluation counts, support, active bounds and the affine-probe error. DFT-bin
excitation gains differ from independent modal readout gains. This conditional
shape fit does not identify losses, physical geometry, pitch or recorded phases;
validate later windows and new executions before adopting a preset. No existing
source default or fitting method is replaced.

`notes.ears.fit_modal_pluck(observations, pick_reference="auto")` fits three
shared effective controls: pluck position, pickup position and force bandwidth.
Each `ModalPluckObservation` supplies a known pitch/string, fundamental, stiffness,
consecutive modal envelopes and optional previously calibrated loss times and
trajectory support mask. At least two different pitches are required. The model
assumes uncoupled spatially rounded strings, zero bridge/contact coupling and a
specified pickup aperture/electrical lowpass; it does not infer those assumptions.
The result reports all local/coarse starts, convergence, active bounds and
Jacobian conditioning. Point excitation and readout positions may exchange roles:
these effective parameters are not uniquely identified physical geometry.
Use `fit.note_controls(pitch, string=1)` on notes with
`ModalString(fretting=True, pickup_position=fit.pickup_position, ...)` and the
same source assumptions. `model_sample_rate=44100` models the renderer's digital
filter even when envelopes were measured on resampled audio. The default uses
each observation's rate. This estimates normalized modal shape, returns no audio,
recorded phase or IR, and requires separate validation on other notes/strikes.
It does not change any instrument default or establish a realistic general preset.

For conditional reconstruction of one known stroke, pass `fixed_pickup=0.07`.
This holds the open-length pickup position exactly, overrides its optimization
bounds, and fits only pluck position/bandwidth; one observation is then sufficient.
The aperture must fit every observed vibrating string. `pickup_fitted=False`
distinguishes this result from joint geometry identification. Such a per-recording
fit is a reconstruction, not evidence of transfer to an unseen strike or note.

For muted notes, optional `palm_resistance` enables a passive distributed viscous
contact instead of scalar `mute_decay`. None retains the original source; zero
selects contact mechanics with no additional damping. `palm_position` and
`palm_width` describe its region along the open string (vibrating length when
fretting is disabled); resistance is an effective N*s/m. These three controls
also accept note overrides requiring `palm_mute=True`. Contact uses intrinsic
`sustain`/`treble_decay`; it does not combine with scalar `mute_decay`.
`mechanical_band` fixes the physical modal band independently of output rate,
bounded by `modes`; low rates use an oversampled observation and decimation.
Real overdamped poles are retained. `pick_cutoff` optionally specifies the force
bandwidth in Hz instead of deriving it from hardness; `pluck_rounding` selects
`"spatial"` or `"release"` (the default). `pick_cutoff` also accepts note overrides.

`examples/palm_contact_guitar.py` demonstrates the new procedural source with
scalar controls, without loading audio. It is an experimental DI profile, not a
replacement for the accepted guitar palette. The fixed-bridge contact prototype
improved several metrics on fresh G3 recordings, but some comparisons regress;
realism, other registers and amplified-riff transfer remain unproven.

Per-note `release_motion`, `bridge_scale=0..1`, `pick_position`, `pick_hardness`
and effective `stiffness` control individual strokes. `fretting=True` uses guitar
string/fret geometry. `modes=8..128` bounds the basis; only modes in the audio band
are emitted. The source currently supports steady pitches, including constant
pitch offsets; changing bends, slides and vibrato raise an explicit error.

`examples/reference_modal_string.py` reconstructs two additional A2 performances
with separately inferred stroke and input-level controls. The distorted-note MERT
diagnostic improves on both, but the ordinary take's spectral profile regresses
against the separately fitted waveguide. Validation is narrow and does not establish
realism across registers or a universal guitar preset. The original instruments
retain their behavior.

  `build_song(calibrated_decay=True)` in that example enables an optional shared
  loss calibration from early (0.2–0.6 s) partial decays of three other A2 takes.
  Bridge Q and mobility change together to preserve the bridge's mass/stiffness.
  The source's input gains are separately matched to the DI attack before the
  shared amp/cabinet chain. On eight other notes/strokes, spectral loss improves
  for 7/8 but MERT improves for only 4/8. This remains a conditional experiment,
  not the default guitar or a demonstrated solution across registers.

  `notes audit-note reference.wav candidate.wav --pitch A2 -o audit.json
  --decay-window 0.2 0.6 --decay-window 1 2` also reports each partial's decay rate,
  residual, audibility and local T60 in the requested windows. Growing, weak and
  strongly modulated partials are flagged unreliable and have no T60 estimate.
  `notes.ears.partial_decays(profile, start=..., end=...)` exposes the same analysis
  without a model dependency. A local exponential fit does not establish a mode's
  lifetime or perceptual realism.
