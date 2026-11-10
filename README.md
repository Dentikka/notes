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

`Piano()` synthesizes decaying inharmonic string partials, unison beating and a
velocity-dependent hammer spectrum. `music.param(pedal=True)` preserves each note's
natural decay after key release (a per-note pedal, not continuous pedal automation).
`BowedString()`, `Violin()` and `Cello()` provide sustained harmonic bow sources,
body resonances, friction noise and controllable vibrato. Explicit per-note vibrato
parameters override the preset. These are procedural approximations, not recorded
instrument libraries or a claim of reference matching.

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

`notes.ears.known_note_tuning(audio, sr, expected_hz)` refines mean tuning of a
stable isolated note with a known nominal pitch. It fits demodulated phase in an
explicit window and checks agreement of detected harmonics. Check `reliable`
before using its `f0`: silence, noise, wrong priors, bends or missing fundamentals
can invalidate the estimate. The default window is 0.15–0.75 s and requires 100 ms
edge margins. This is a calibration tool, not an unknown-melody transcriber or a
replacement for the existing YIN path. It helps prevent a distorted reference's
pitch-estimator bias from being mistaken for a timbre mismatch.

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

`ElectricGuitar` and `BassGuitar` also accept per-note `stiffness`, for example
`Note("E4").param(stiffness=0.0001)`. This is the effective dispersion B of the
vibrating string; it is not scaled again by fret geometry. Values must be finite
and nonnegative. Omitting it preserves the instrument's existing global B.
This lets known-note inharmonicity measurements reach individual notes without
creating a separate track for every stiffness value; it does not establish realism.
For measured-B calibration, select `dispersion_fit="bracketed"` explicitly: the
legacy search can disable dispersion and make different requested B values yield
the same waveform. Both modes retain the existing approximate two-allpass model.

For amplified guitars, instrument `level` is an excitation gain before the amp;
changing it changes saturation even after loudness matching. Use `Track.gain` or
master loudness for output-volume changes that preserve the amp operating point.

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
