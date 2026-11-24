"""Five independent instrumental miniatures, all voices generated procedurally.

Each piece has its own meter, harmonic language, lead roles and dramatic arc.
No reference recordings, sampled notes or measured impulse responses are used.
"""

from fractions import Fraction

from notes import (
    EQ,
    FM,
    AcousticGuitar,
    Bass,
    BassGuitar,
    Cello,
    Chorus,
    Compressor,
    Delay,
    DrumKit,
    ElectricGuitar,
    EPiano,
    Filter,
    Lead,
    Note,
    NylonGuitar,
    Pad,
    Piano,
    Rest,
    Reverb,
    RockKit,
    Room,
    Song,
    Synth,
    Track,
    Violin,
    cat,
    place,
    stack,
)

QUALITIES = {"m": (0, 3, 7), "M": (0, 4, 7), "m7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11), "7": (0, 4, 7, 10)}
CATALOG = {
    "chamber": ("Листопад", "Камерный вальс", 88, 3, 32),
    "bossa": ("Солнечный двор", "Босса-нова", 112, 4, 24),
    "funk": ("Зелёный свет", "Фанк", 104, 4, 24),
    "synthwave": ("Ночной маршрут", "Синтвейв", 118, 4, 32),
    "postrock": ("После дождя", "Мелодичный пост-рок, 6/8", 84, 3, 28),
}


class Piece:
    def __init__(self, kind):
        self.kind = kind
        self.title, self.genre, self.bpm, self.bar, self.bars = CATALOG[kind]
        self.length = self.bar * self.bars
        self.tracks = {}
        self.events = {}

    def track(self, name, instrument, gain=0.0, pan=0.0, fx=()):
        self.tracks[name] = Track(name, instrument, gain=gain, pan=pan, fx=fx)
        self.events[name] = []

    def note(self, name, at, pitch, duration, velocity=0.65, **params):
        at, duration = Fraction(str(at)), Fraction(str(duration))
        assert 0 <= at < self.length and 0 < duration <= self.length - at
        assert 0 < velocity <= 1
        self.events[name].append((at, Note(pitch, duration, velocity, params)))

    def chord(self, name, at, pitches, duration, velocity=0.6, spread=0.018, **params):
        for i, pitch in enumerate(pitches):
            self.note(name, at + i * spread, pitch, duration - i * spread, velocity * (1 - 0.08 * i), **params)

    def bow(self, name, at, pitches, durations, velocity=0.55, swell=1.0):
        # One continuous voice with short, smooth pitch transitions between notes.
        seconds = 60 / self.bpm
        points, cursor, previous = [(0.0, 0.0)], 0.0, pitches[0]
        for pitch, duration in zip(pitches, durations, strict=True):
            if cursor:
                points.extend(
                    [
                        (max(0.0, cursor * seconds - 0.035), previous - pitches[0]),
                        (cursor * seconds + 0.025, pitch - pitches[0]),
                    ]
                )
            cursor += duration
            previous = pitch
        gate = cursor - 0.05
        envelope = (
            (0.0, 0.12),
            (0.18, 0.78),
            (gate * seconds * 0.55, swell),
            (gate * seconds - 0.12, 0.72),
            (gate * seconds, 0.18),
        )
        self.note(
            name, at, pitches[0], gate, velocity, pitch_points=tuple(points), gain_points=envelope, vibrato_delay=0.28
        )

    def finish(self, sections, key, loudness, seed):
        markers = cat(*(Rest(bars * self.bar).named(label) for label, bars in sections))
        assert markers.dur == self.length
        music = stack(
            *(self.tracks[name](place(items, length=self.length)) for name, items in self.events.items() if items)
        )
        meter = (6, 8) if self.kind == "postrock" else ((3, 4) if self.bar == 3 else (4, 4))
        return Song(
            stack(music, markers),
            bpm=self.bpm,
            meter=meter,
            key=key,
            title=self.title,
            seed=seed,
            sample_rate=44100,
            loudness=loudness,
            ceiling=-1.0,
        )


def chamber():
    p = Piece("chamber")
    p.track(
        "piano",
        Piano(model="hammer_damped", release=0.17),
        gain=21,
        fx=(Filter(kind="highpass", cutoff=70), Reverb(size=0.4, damp=0.55, mix=0.13)),
    )
    p.track(
        "violin",
        Violin(pressure=0.42, vibrato=0.055, bow_noise=0.012, attack=0.10, release=0.25),
        gain=-8,
        pan=0.25,
        fx=(Filter(cutoff=5000), Reverb(size=0.47, damp=0.5, mix=0.18)),
    )
    p.track(
        "cello",
        Cello(pressure=0.4, vibrato=0.045, bow_noise=0.012, release=0.25),
        gain=-9,
        pan=-0.23,
        fx=(Filter(kind="highpass", cutoff=95), Reverb(size=0.43, mix=0.13)),
    )
    a = [(48, "m"), (44, "M"), (51, "M"), (46, "M"), (41, "m"), (48, "m"), (43, "7"), (48, "m")]
    b = [(44, "M"), (46, "M"), (51, "M"), (43, "7"), (41, "m"), (44, "M"), (43, "7"), (43, "7")]
    harmony = a[:4] + a + b + a + [(41, "m"), (43, "7"), (48, "m"), (48, "m")]
    theme = [
        [72, 75, 79, 77],
        [75, 72, 68],
        [70, 74, 79],
        [77, 74, 70],
        [72, 77, 80],
        [79, 75, 72],
        [71, 74, 77, 74],
        [75, 74, 72],
    ]
    for bar, (root, quality) in enumerate(harmony):
        at = bar * 3
        energy = 0.48 if bar < 4 else (0.6 if bar < 20 else 0.72)
        if bar >= 28:
            energy = [0.58, 0.52, 0.48, 0.36][bar - 28]
        intervals = QUALITIES[quality]
        if bar == 31:
            p.chord("piano", at, [48, 55, 60, 63, 67, 72], 2.85, 0.46, spread=0.04)
            continue
        left = [root, root + 7, root + 12, root + intervals[1] + 12, root + 7, root + 12]
        for i, pitch in enumerate(left):
            p.note("piano", at + i * 0.5, pitch, 0.62, energy * (0.76 if i == 0 else 0.52))
        if 4 <= bar < 12 or 20 <= bar < 28:
            melody = theme[(bar - 4) % 8]
            times = [0.0, 0.75, 1.5, 2.25] if len(melody) == 4 else [0.0, 1.0, 2.0]
            for i, (time, pitch) in enumerate(zip(times, melody, strict=True)):
                p.note(
                    "piano", at + time, pitch, 0.66 if len(melody) == 4 else 0.91, energy * (1.16 if i == 0 else 1.0)
                )
        elif bar < 4:
            p.note("piano", at + 1, root + 24 + intervals[1], 1.25, energy * 0.9)
        elif 12 <= bar < 20:
            p.chord("piano", at + 1, [root + 12 + i for i in intervals[:3]], 1.4, energy * 0.43)
        else:
            p.note("piano", at + 1, [77, 71, 72][min(bar - 28, 2)], 1.6, energy)
    for block, notes in enumerate([[80, 79, 77, 74], [79, 77, 75, 74], [77, 80, 79, 77], [74, 71, 74, 77]]):
        p.bow("violin", (12 + block * 2) * 3, notes, [1.5] * 4, 0.54 + 0.035 * block)
    for block in range(4):
        root1 = harmony[20 + 2 * block][0]
        root2 = harmony[21 + 2 * block][0]
        p.bow("cello", (20 + 2 * block) * 3, [root1, root1 + 7, root2, root2 + 7], [1.5] * 4, 0.48)
    p.bow("violin", 28 * 3, [77, 74, 71, 72], [3, 3, 3, 3], 0.43, 0.86)
    return p.finish(
        [("Отражение", 4), ("Тема", 8), ("Ответ скрипки", 8), ("Возвращение", 8), ("Тихая кода", 4)],
        "C minor",
        -19,
        241101,
    )


def bossa():
    p = Piece("bossa")
    p.track(
        "nylon",
        NylonGuitar(body_size=1.25, brightness=0.48),
        gain=3,
        pan=-0.26,
        fx=(Filter(kind="highpass", cutoff=105), Filter(cutoff=5700), Reverb(size=0.28, damp=0.6, mix=0.09)),
    )
    p.track(
        "steel",
        AcousticGuitar(fretting=True, brightness=0.55, pick_noise=0.045),
        gain=-4,
        pan=0.35,
        fx=(Filter(kind="highpass", cutoff=170), Filter(cutoff=5000), Reverb(size=0.25, mix=0.08)),
    )
    p.track("rhodes", EPiano(), gain=1, pan=0.1, fx=(Filter(cutoff=5300), Reverb(size=0.32, mix=0.13)))
    p.track(
        "bass",
        BassGuitar(clank=0.035, drive=0.05, sustain=8, release=0.05),
        gain=-3,
        fx=(Compressor(threshold=-20, ratio=2, attack=0.02, release=0.12),),
    )
    p.track(
        "percussion", DrumKit(decay=0.55, punch=0.15), gain=-10, pan=0.1, fx=(Room(size=0.2, decay=0.22, mix=0.06),)
    )
    progression = [(48, "maj7"), (45, "m7"), (50, "m7"), (43, "7"), (52, "m7"), (45, "7"), (50, "m7"), (43, "7")]
    harmony = (
        progression[:2]
        + progression
        + [(53, "maj7"), (46, "7"), (52, "m7"), (45, "7")]
        + progression[2:]
        + progression[:4]
    )
    harmony[-1] = (48, "maj7")
    for bar, (root, quality) in enumerate(harmony):
        at = bar * 4
        iv = QUALITIES[quality]
        energy = 0.52 if bar < 2 else (0.68 if bar < 20 else 0.58)
        last = bar == 23
        voicing = [root + 12 + i for i in iv]
        for time, vel, duration in [(0.0, 0.8, 0.8), (1.5, 0.65, 0.7), (2.5, 0.88, 0.9)]:
            p.chord("nylon", at + time, voicing, duration, energy * vel, spread=0.022)
        for time, pitch in [
            (0.0, root - 12),
            (1.75, root - 5),
            (2.0, root - 12),
            (3.5, root - 10 if quality == "7" else root - 5),
        ]:
            p.note("bass", at + time, pitch, 0.42 if time % 1 else 1.35, 0.68 if time % 1 == 0 else 0.46)
        if bar >= 2:
            for t in [0.0, 2.0]:
                p.note("percussion", at + t, 36, 0.1, 0.47)
            for t in [1.0, 3.0]:
                p.note("percussion", at + t, 37, 0.1, 0.57)
            for i in range(8):
                p.note("percussion", at + i * 0.5, 42, 0.1, 0.26 if i % 2 == 0 else 0.4)
        if 10 <= bar < 14:
            for i, offset in enumerate([0, 4, 7, 12, 7, 4]):
                p.note("steel", at + i * 0.5, root + 12 + offset, 0.38, 0.52 if i in [0, 3] else 0.4)
        elif bar >= 2:
            pattern = [iv[-1], 12, iv[1] + 12, iv[2] + 12] if bar % 2 == 0 else [iv[2] + 12, iv[1] + 12, 12, iv[-1]]
            for i, (time, duration) in enumerate([(0.0, 0.65), (0.75, 0.65), (1.75, 0.85), (3.0, 0.75)]):
                p.note("rhodes", at + time, root + pattern[i], duration, energy * (0.95 if i == 2 else 0.8))
        if last:
            p.chord("nylon", at + 3.25, [48, 55, 59, 64], 0.7, 0.45, spread=0.025)
    return p.finish(
        [
            ("Солнечное вступление", 2),
            ("Прогулка", 8),
            ("Акустический ответ", 4),
            ("Разговор", 6),
            ("Возвращение домой", 4),
        ],
        "C major",
        -18,
        241102,
    )


def funk():
    p = Piece("funk")
    p.track(
        "bass",
        BassGuitar(sustain=9, drive=0.22, clank=0.16, release=0.025),
        gain=-2,
        fx=(Compressor(threshold=-19, ratio=2.6, attack=0.018, release=0.11),),
    )
    p.track(
        "guitar",
        ElectricGuitar(
            drive=0.16, tone=0.4, pickup="neck", pickup_q=0.7, fretting=True, treble_decay=0.75, pick_noise=0.04
        ),
        gain=-2,
        pan=-0.36,
        fx=(
            Filter(kind="highpass", cutoff=170),
            Filter(cutoff=4700),
            Chorus(side_only=True, mix=0.11, depth=0.0005),
            Room(mix=0.04),
        ),
    )
    p.track("rhodes", EPiano(), gain=1, pan=0.28, fx=(Chorus(mix=0.13, depth=0.001), Reverb(size=0.25, mix=0.08)))
    p.track(
        "drums",
        RockKit(decay=0.65, rimshot=0.18, wires=0.62),
        gain=-1,
        fx=(Room(size=0.22, decay=0.25, damp=0.6, mix=0.06),),
    )
    p.track("hats", RockKit(decay=0.5), gain=-7, pan=0.32)
    roots = [40, 40, 45, 45, 38, 47, 40, 47]
    for bar in range(24):
        at = bar * 4
        root = roots[bar % 8]
        breakdown = 12 <= bar < 16
        energy = 0.58 if bar < 4 else (0.54 if breakdown else 0.76)
        cells = [
            (0, 0, 0.62),
            (0.75, 0, 0.22),
            (1.5, 7, 0.28),
            (2.0, 10, 0.43),
            (2.75, 12, 0.22),
            (3.25, 7, 0.22),
            (3.75, 10, 0.18),
        ]
        if bar % 4 == 3:
            cells = [(0, 0, 0.7), (1.0, 7, 0.35), (1.75, 10, 0.2), (2.5, 12, 0.2), (3.0, 11, 0.2), (3.5, 10, 0.3)]
        for i, (time, offset, duration) in enumerate(cells):
            if breakdown and i not in [0, 2, 4]:
                continue
            p.note(
                "bass", at + time, root - 12 + offset, duration, energy * (1.15 if i == 0 else 0.92), palm_mute=i == 1
            )
        for time, vel in [(0.0, 0.9), (1.5, 0.62), (2.75, 0.76)]:
            p.note("drums", at + time, 36, 0.1, energy * vel)
        if not breakdown:
            for t in [1.0, 3.0]:
                p.note("drums", at + t, 38, 0.1, 0.77)
            if bar % 2:
                p.note("drums", at + 2.75, 38, 0.1, 0.23)
        for i in range(8 if breakdown else 16):
            step = 0.5 if breakdown else 0.25
            p.note(
                "hats",
                at + i * step + (0.012 if i % 2 else 0),
                42,
                0.07,
                0.5 if i % 4 == 2 else (0.33 if i % 2 else 0.4),
            )
        quality = "7" if root in [45, 47] else "m7"
        iv = QUALITIES[quality]
        chord = [root + 24 + iv[1], root + 24 + iv[2], root + 24 + iv[3]]
        if not breakdown:
            for t in [0.5, 1.75, 2.5, 3.5]:
                p.chord(
                    "guitar",
                    at + t,
                    chord,
                    0.16,
                    0.55 if t == 2.5 else 0.44,
                    spread=0.006,
                    palm_mute=True,
                    mute_decay=0.13,
                )
        if 4 <= bar < 12 or bar >= 16:
            if bar % 2 == 0:
                for t, offset, dur in [
                    (0.0, 12, 0.5),
                    (0.75, iv[1] + 12, 0.5),
                    (1.5, iv[2] + 12, 0.8),
                    (2.75, iv[3] + 12, 0.9),
                ]:
                    p.note("rhodes", at + t, root + 12 + offset, dur, 0.7 if t == 1.5 else 0.58)
            else:
                p.chord("rhodes", at + 0.5, [root + 12 + i for i in iv], 1.2, 0.48)
        if bar in [7, 11, 19, 23]:
            for i, pitch in enumerate([38, 45, 47, 50]):
                p.note("drums", at + 3 + i * 0.25, pitch, 0.1, 0.4 + 0.1 * i)
    return p.finish(
        [("Басовый крючок", 4), ("Полный грув", 8), ("Карман", 4), ("Соло клавиш", 4), ("Последний круг", 4)],
        "E minor",
        -17,
        241103,
    )


def synthwave():
    p = Piece("synthwave")
    p.track(
        "pad",
        Pad(unison=4, detune=16, cutoff=1500, attack=0.28, release=0.65),
        gain=-10,
        pan=-0.15,
        fx=(Chorus(side_only=True, mix=0.22, depth=0.001), Reverb(size=0.45, damp=0.5, mix=0.16)),
    )
    p.track("bass", Bass(cutoff=250, env_amount=2.2, sub=0.35), gain=-4)
    p.track(
        "arp",
        Synth(wave="triangle", cutoff=2800, attack=0.003, decay=0.14, sustain=0.12, release=0.13),
        gain=-6,
        pan=0.35,
        fx=(Delay(time=0.75, feedback=0.28, mix=0.3, tone=3200), Reverb(size=0.35, mix=0.09)),
    )
    p.track(
        "lead",
        Lead(wave="saw", unison=2, detune=5, cutoff=2900, resonance=0.8, env_amount=0.7, release=0.18),
        gain=-5,
        pan=-0.08,
        fx=(Delay(time=0.75, feedback=0.28, mix=0.2, tone=3300), Reverb(size=0.42, mix=0.14)),
    )
    p.track(
        "glass",
        FM(ratio=3, index=1.1),
        gain=-10,
        pan=-0.4,
        fx=(Delay(time=1.5, feedback=0.32, mix=0.3), Reverb(size=0.5, mix=0.2)),
    )
    p.track("drums", DrumKit(decay=0.82, punch=0.5), gain=-2, fx=(Room(size=0.35, decay=0.48, mix=0.1),))
    chords = [(42, "m"), (38, "M"), (45, "M"), (40, "M")]
    for bar in range(32):
        at = bar * 4
        root, quality = chords[(bar // 2) % 4]
        iv = QUALITIES[quality]
        quiet = 20 <= bar < 24
        if 12 <= bar < 20:
            root, quality = [(47, "m"), (38, "M"), (40, "M"), (37, "7")][((bar - 12) // 2) % 4]
            iv = QUALITIES[quality]
        if bar >= 30:
            root, quality = 42, "m"
            iv = QUALITIES[quality]
        end = bar >= 28
        energy = 0.44 if bar < 4 or quiet else (0.56 if end else 0.72)
        if bar % 2 == 0:
            p.chord(
                "pad", at, [root + 12 + i for i in iv] + [root + 24], 7.8 if bar < 30 else 7.6, energy * 0.8, spread=0.0
            )
        for i in range(8):
            if quiet and i % 2:
                continue
            p.note(
                "arp",
                at + i * 0.5,
                root + 24 + [0, 7, 12, iv[1], 7, 12, 19, iv[1] + 12][i],
                0.28,
                energy * (1.0 if i % 4 == 0 else 0.7),
            )
        if bar >= 4 and not quiet and bar < 30:
            for i in range(8):
                p.note("bass", at + i * 0.5, root - 12 + (12 if i in [3, 7] else 0), 0.34, 0.7 if i % 2 == 0 else 0.51)
            for time in [0.0, 2.0]:
                p.note("drums", at + time, 36, 0.1, 0.85)
            for time in [1.0, 3.0]:
                p.note("drums", at + time, 39, 0.1, 0.69)
            for i in range(8):
                p.note("drums", at + i * 0.5, 42, 0.1, 0.33 if i % 2 == 0 else 0.48)
        if 4 <= bar < 20 or 24 <= bar < 28:
            motif = [0, 7, 12, 10, 7, 3] if quality == "m" else [0, 7, 12, 11, 7, 4]
            if quality == "7":
                motif = [0, 4, 7, 10, 7, 4]
            if bar % 2:
                motif = list(reversed(motif))
            for i, (time, dur) in enumerate(
                [(0.0, 0.7), (0.75, 0.65), (1.5, 0.4), (2.0, 0.65), (2.75, 0.4), (3.25, 0.6)]
            ):
                p.note(
                    "lead",
                    at + time,
                    root + 24 + motif[i],
                    dur,
                    energy * (1.05 if i in [0, 3] else 0.9),
                    vibrato=0.035,
                    vibrato_delay=0.18,
                    slide=-1 if i == 0 and bar % 4 == 0 else 0,
                    slide_time=0.08,
                )
        if quiet or end:
            p.note("glass", at + 0.5, root + 36 + iv[1], 1.25, energy * 0.65)
        if bar in [11, 19, 27]:
            for i in range(4):
                p.note("drums", at + 3 + i * 0.25, 38, 0.1, 0.28 + i * 0.12)
    return p.finish(
        [
            ("Неоновая сетка", 4),
            ("Маршрут", 8),
            ("Новый горизонт", 8),
            ("Невесомость", 4),
            ("Возвращение темы", 4),
            ("Огни вдали", 4),
        ],
        "F# minor",
        -17,
        241104,
    )


def postrock():
    p = Piece("postrock")
    p.track(
        "clean",
        ElectricGuitar(drive=0.09, tone=0.38, pickup="neck", fretting=True, pick_noise=0.045, pickup_q=0.7),
        gain=-1,
        pan=-0.3,
        fx=(
            Filter(kind="highpass", cutoff=130),
            Chorus(side_only=True, mix=0.16, depth=0.0005),
            Delay(time=1.5, feedback=0.25, mix=0.18),
            Reverb(size=0.5, mix=0.15),
        ),
    )
    for name, pan, position in [("rhythm-L", -0.65, 0.065), ("rhythm-R", 0.65, 0.075)]:
        p.track(
            name,
            ElectricGuitar(
                drive=0.49,
                tone=0.36,
                fretting=True,
                pickup_position=position,
                pickup_width=0.04,
                pickup_q=0.65,
                pickup_cutoff=3500,
                treble_decay=1.5,
                pick_noise=0.07,
            ),
            gain=-4,
            pan=pan,
            fx=(
                Filter(kind="highpass", cutoff=105),
                EQ(high=-3, high_freq=3600, mid=-2, mid_freq=1500),
                Filter(cutoff=4700),
                Reverb(size=0.3, mix=0.06),
            ),
        )
    p.track(
        "lead",
        ElectricGuitar(
            drive=0.52, tone=0.4, sustain=9, fretting=True, pickup_q=0.7, pick_noise=0.055, polarization=0.16
        ),
        gain=-1,
        fx=(
            Filter(kind="highpass", cutoff=180),
            Filter(cutoff=4800),
            Delay(time=0.75, feedback=0.25, mix=0.17, tone=3300),
            Reverb(size=0.4, mix=0.15),
        ),
    )
    p.track(
        "bass",
        BassGuitar(drive=0.17, clank=0.09, sustain=9, release=0.04),
        gain=-4,
        fx=(Compressor(threshold=-21, ratio=2.2, attack=0.02, release=0.13),),
    )
    p.track(
        "drums",
        RockKit(decay=0.8, wires=0.7, rimshot=0.24),
        gain=-1,
        fx=(Room(size=0.38, decay=0.4, damp=0.6, mix=0.1),),
    )
    p.track(
        "cello", Cello(pressure=0.4, vibrato=0.05, bow_noise=0.012), gain=-11, pan=0.2, fx=(Reverb(size=0.5, mix=0.17),)
    )
    progression = [(50, "m"), (46, "M"), (53, "M"), (48, "M"), (43, "m"), (46, "M"), (45, "7"), (50, "m")]
    harmony = (
        progression[:4]
        + progression
        + [(43, "m"), (46, "M"), (45, "7"), (45, "7")]
        + progression
        + [(43, "m"), (45, "7"), (50, "m"), (50, "m")]
    )
    for bar, (root, quality) in enumerate(harmony):
        at = bar * 3
        iv = QUALITIES[quality]
        full = 4 <= bar < 12 or 16 <= bar < 24
        energy = 0.56 if bar < 4 or 12 <= bar < 16 else 0.74
        if bar >= 24:
            energy = [0.62, 0.59, 0.52, 0.42][bar - 24]
        if (not full or bar < 8) and bar != 27:
            for i, offset in enumerate([0, 7, 12, iv[1] + 12, 7, 12]):
                p.note(
                    "clean",
                    at + i * 0.5,
                    root + offset,
                    0.45 if i == 5 else 0.7,
                    energy * (0.9 if i == 0 else 0.68),
                    pitch_interpolation="sinc",
                )
        if full:
            for side, name in enumerate(["rhythm-L", "rhythm-R"]):
                for time, duration, vel in [(0.0, 1.1, 0.82), (1.5, 0.28, 0.68), (2.0, 0.7, 0.8)]:
                    p.chord(
                        name,
                        at + time + side * 0.012,
                        [root, root + 7],
                        duration,
                        energy * vel,
                        spread=0.016,
                        palm_mute=time == 1.5,
                        pitch_interpolation="sinc",
                    )
            p.note("bass", at, root - 12, 1.32, 0.76)
            p.note("bass", at + 1.5, root - 12, 1.2, 0.66)
            for time in [0.0, 2.0]:
                p.note("drums", at + time, 36, 0.1, 0.82 if time == 0 else 0.64)
            p.note("drums", at + 1.5, 38, 0.1, 0.73)
            for i in range(6):
                p.note("drums", at + i * 0.5, 42, 0.1, 0.43 if i in [0, 3] else 0.31)
        if (8 <= bar < 12) or (16 <= bar < 24):
            phrase = (
                [root + 12, root + iv[1] + 12, root + 19, root + 12]
                if bar % 2 == 0
                else [root + 19, root + iv[1] + 12, root + 12]
            )
            times = [0.0, 0.75, 1.5, 2.25] if len(phrase) == 4 else [0.0, 1.0, 2.0]
            for i, (time, pitch) in enumerate(zip(times, phrase, strict=True)):
                p.note(
                    "lead",
                    at + time,
                    pitch,
                    0.66 if len(phrase) == 4 else 0.9,
                    energy * (1.0 if i == 0 else 0.87),
                    pitch_interpolation="sinc",
                    vibrato=0.07,
                    vibrato_delay=0.2,
                    slide=-2 if i == 0 and bar % 2 == 0 else 0,
                    slide_time=0.12,
                )
        if bar in [11, 23]:
            for i, pitch in enumerate([45, 47, 50]):
                p.note("drums", at + 2 + i / 3, pitch, 0.1, 0.58 + 0.08 * i)
        if bar == 27:
            p.chord("clean", at, [50, 57, 62, 65], 2.8, 0.5, spread=0.035)
            p.note(
                "lead",
                at,
                62,
                2.6,
                0.6,
                vibrato=0.045,
                pitch_interpolation="sinc",
                gain_points=((0.0, 0.75), (1.0, 0.7), (1.8, 0.25)),
            )
    p.bow("cello", 12 * 3, [55, 58, 57, 52], [3, 3, 3, 3], 0.5)
    p.bow("cello", 24 * 3, [55, 57, 50, 50], [3, 3, 3, 3], 0.43, 0.85)
    return p.finish(
        [("Капли", 4), ("Движение", 8), ("Просвет", 4), ("Подъём", 8), ("После дождя", 4)], "D minor", -18, 241105
    )


def build_song(kind="chamber"):
    return {"chamber": chamber, "bossa": bossa, "funk": funk, "synthwave": synthwave, "postrock": postrock}[kind]()


song = build_song()
