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
| harmony | `.rhythm("x..x..x.")` re-strikes held chords, `.arp("updown")` arpeggiates |

**Mini-notation** (strings, `|` bar lines ignored):

```python
melody("A4 C5 E5 _ D5 . C5@2 [A3,C4,E4] G4> B4!2", step=E)   # _ hold, . rest, @n length,
                                                              # [..] chord, > accent, !n repeat
steps(kick="x...x...", snare="....x...", hat="x.x.x.x.")      # x hit, X accent, o soft, 1-9
steps(kick=euclid(3, 8), hat="x..", length=bars(2))           # Euclidean rhythms, polymeter
chords("Am F C G", voicing="smooth")                          # chord symbols, voice-led
Key.parse("A minor").chords("i VI III VII")                   # roman numerals
Key.parse("A minor").melody("1 3 5 8 b7")                     # scale degrees
```

**Instruments**: `Synth` (subtractive) with `Bass`, `Pad`, `Lead`; `FM` with `EPiano`,
`Bell`; `Pluck` and `ElectricGuitar` (Karplus–Strong strings, pickup model, amp and
cabinet; `drive`, `tone`, `pickup`); `DrumKit` (808/909-style synthesis addressed by GM
names: `kick`, `snare`, `hat`, `oh`, `clap`, `crash`, `ride`, toms, …). Presets are
calibrated to equal loudness, so a track's `gain` is dB relative to the others.

**Effects**: `Reverb` (Freeverb), `Delay` (tempo-synced), `Chorus`, `Compressor`, `EQ`,
`Filter`, `Drive`, `GuitarAmp`, `Cabinet`.

## Command line

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
79 tests, one demo (`examples/first_groove.py`). Next: an MCP server so agents can compose,
render and listen in a loop; a more convincing guitar model (CLAP recognises every synth
preset but not the guitars); send buses; automation curves; a declarative text front-end.
