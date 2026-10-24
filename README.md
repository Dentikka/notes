# notes — music as code

`notes` is a framework for writing music as programs, by hand or by an LLM agent. A
composition is a short Python script. The library compiles it into an exact, serialisable
intermediate representation (IR) and renders that to audio with a built-in DSP engine: no
samples, no external synthesizers, bit-reproducible output.

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
