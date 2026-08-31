"""
audio_player.py
---------------
MIDI → Audio conversion and playback.

Three backends, tried in order of quality:

  Backend A — FluidSynth (best quality)
    Requires:  fluidsynth executable + a SoundFont (.sf2) file
    Produces:  high-quality WAV via synthesis
    Install:   https://www.fluidsynth.org/  (Windows: choco install fluidsynth)

  Backend B — pygame.mixer (instant, no extra install needed)
    Requires:  pygame  (`pip install pygame`)
    Produces:  direct MIDI playback through the OS MIDI device
    Note:      playback quality depends on the system's MIDI synthesiser

  Backend C — mido + numpy beep synthesiser (pure Python fallback)
    Requires:  mido, numpy, soundfile  (`pip install mido numpy soundfile`)
    Produces:  a simple sine-wave WAV — good enough to verify the melody
    Note:      no external binary required; always works

Usage:
    python src/audio_player.py output/midi/generated_001.mid
    python src/audio_player.py output/midi/generated_001.mid --backend pygame
    python src/audio_player.py output/midi/generated_001.mid --export-wav
    python src/audio_player.py output/midi/generated_001.mid --soundfont path/to/font.sf2
"""

import os
import sys
import time
import struct
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR    = Path(__file__).resolve().parent
BASE_DIR   = SRC_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
AUDIO_DIR  = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Common SoundFont locations (Windows + Linux + macOS)
DEFAULT_SOUNDFONTS = [
    # Windows (MuseScore, VirtualMIDISynth, etc.)
    r"C:\Program Files\MuseScore 4\sound\MuseScore_General.sf3",
    r"C:\soundfonts\GeneralUser.sf2",
    r"C:\Windows\System32\drivers\gm.dls",
    # Linux
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    # macOS (HomeBrew fluidsynth)
    "/usr/local/share/fluidsynth/GeneralUser.sf2",
    "/opt/homebrew/share/fluidsynth/GeneralUser.sf2",
]


def find_soundfont() -> str | None:
    """Return the first available SoundFont path, or None."""
    for sf in DEFAULT_SOUNDFONTS:
        if Path(sf).exists():
            return sf
    return None


# ---------------------------------------------------------------------------
# Backend A — FluidSynth
# ---------------------------------------------------------------------------

def _fluidsynth_available() -> bool:
    try:
        result = subprocess.run(
            ["fluidsynth", "--version"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def convert_midi_to_wav_fluidsynth(
    midi_path:  Path,
    wav_path:   Path,
    soundfont:  str = None,
    sample_rate: int = 44100,
) -> bool:
    """
    Use FluidSynth to render a MIDI file to a WAV.
    Returns True on success, False on failure.
    """
    if not _fluidsynth_available():
        return False

    sf = soundfont or find_soundfont()
    if not sf:
        print("[fluidsynth] No SoundFont found. "
              "Provide one with --soundfont or install MuseScore/FluidR3.")
        return False

    cmd = [
        "fluidsynth",
        "-ni",
        sf,
        str(midi_path),
        "-F", str(wav_path),
        "-r", str(sample_rate),
    ]
    print(f"[fluidsynth] Rendering {midi_path.name} …")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and wav_path.exists():
            print(f"[fluidsynth] ✓ WAV saved → {wav_path}")
            return True
        else:
            print(f"[fluidsynth] ✗ Failed: {result.stderr.decode()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("[fluidsynth] ✗ Timed out.")
        return False


def play_midi_fluidsynth(midi_path: Path, soundfont: str = None) -> bool:
    """Play MIDI through FluidSynth's built-in audio driver."""
    if not _fluidsynth_available():
        return False

    sf = soundfont or find_soundfont()
    if not sf:
        return False

    cmd = ["fluidsynth", "-a", "dsound", sf, str(midi_path)]
    if sys.platform != "win32":
        cmd[2] = "pulseaudio"

    print(f"[fluidsynth] Playing {midi_path.name} …  (press Ctrl+C to stop)")
    try:
        subprocess.run(cmd, timeout=600)
        return True
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Backend B — pygame.mixer
# ---------------------------------------------------------------------------

def play_midi_pygame(midi_path: Path, wait: bool = True) -> bool:
    """
    Play a MIDI file using pygame.mixer.music.
    Returns True on success.
    """
    try:
        import pygame
    except ImportError:
        return False

    try:
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.music.load(str(midi_path))
        pygame.mixer.music.play()

        print(f"[pygame] Playing {midi_path.name} …  (press Ctrl+C to stop)")
        if wait:
            try:
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except KeyboardInterrupt:
                pygame.mixer.music.stop()

        pygame.mixer.quit()
        pygame.quit()
        return True

    except Exception as exc:
        print(f"[pygame] ✗ {exc}")
        try:
            pygame.quit()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Backend C — Pure-Python sine-wave synthesiser
# ---------------------------------------------------------------------------

# MIDI note number → frequency in Hz
def _midi_note_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _parse_midi_events(midi_path: Path) -> list:
    """
    Minimal MIDI parser using only stdlib struct.
    Returns list of (time_seconds, note, velocity) tuples.
    Events with velocity==0 are note-offs.
    """
    data = midi_path.read_bytes()
    pos  = 0

    def read_bytes(n):
        nonlocal pos
        chunk = data[pos:pos + n]
        pos  += n
        return chunk

    def read_uint(n):
        return int.from_bytes(read_bytes(n), "big")

    def read_varlen():
        nonlocal pos
        result = 0
        while True:
            b = data[pos]; pos += 1
            result = (result << 7) | (b & 0x7F)
            if not (b & 0x80):
                break
        return result

    # Header chunk
    if read_bytes(4) != b"MThd":
        return []
    read_uint(4)                        # length (always 6)
    fmt          = read_uint(2)
    num_tracks   = read_uint(2)
    ticks_per_beat = read_uint(2)

    tempo        = 500000               # default 120 BPM
    events       = []
    abs_tick     = 0

    for _ in range(num_tracks):
        if data[pos:pos+4] != b"MTrk":
            break
        pos     += 4
        trk_len  = read_uint(4)
        end_pos  = pos + trk_len
        abs_tick = 0
        running_status = None

        while pos < end_pos:
            delta = read_varlen()
            abs_tick += delta

            b = data[pos]

            if b == 0xFF:               # meta event
                pos += 1
                meta_type = data[pos]; pos += 1
                meta_len  = read_varlen()
                meta_data = read_bytes(meta_len)
                if meta_type == 0x51 and meta_len == 3:
                    tempo = int.from_bytes(meta_data, "big")
                continue

            if b == 0xF0 or b == 0xF7:  # SysEx
                pos += 1
                sysex_len = read_varlen()
                read_bytes(sysex_len)
                continue

            if b & 0x80:
                running_status = b
                pos += 1
            else:
                b = running_status      # running status

            if b is None:
                pos += 1
                continue

            status  = b & 0xF0
            channel = b & 0x0F

            if status in (0x80, 0x90):  # note off / note on
                note     = data[pos]; pos += 1
                velocity = data[pos]; pos += 1
                time_s   = abs_tick * tempo / ticks_per_beat / 1_000_000
                events.append((time_s, note, velocity if status == 0x90 else 0))
            elif status in (0xA0, 0xB0, 0xE0):
                pos += 2
            elif status in (0xC0, 0xD0):
                pos += 1
            else:
                pos += 1

    return events


def convert_midi_to_wav_synth(
    midi_path:   Path,
    wav_path:    Path,
    sample_rate: int   = 22050,
    amplitude:   float = 0.25,
) -> bool:
    """
    Pure-Python MIDI → WAV converter using sine-wave synthesis.
    Requires numpy and soundfile.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("[synth] numpy or soundfile not available — install with pip.")
        return False

    print(f"[synth] Synthesising {midi_path.name} …")
    events = _parse_midi_events(midi_path)

    if not events:
        print("[synth] No MIDI events found — empty file?")
        return False

    # Build per-note (start, end, freq) from note-on/off pairs
    note_ons   = {}   # note → (time_s, velocity)
    segments   = []   # (start, end, freq, vel)

    for t, note, vel in sorted(events):
        if vel > 0:
            note_ons[note] = (t, vel)
        else:
            if note in note_ons:
                start, v = note_ons.pop(note)
                freq = _midi_note_to_hz(note)
                segments.append((start, t, freq, v))

    # Close any still-open notes at the last event time
    last_t = max(t for t, _, _ in events) if events else 1.0
    for note, (start, v) in note_ons.items():
        segments.append((start, last_t + 0.25, _midi_note_to_hz(note), v))

    if not segments:
        print("[synth] No complete note segments found.")
        return False

    total_duration = max(end for _, end, _, _ in segments) + 0.5
    num_samples    = int(total_duration * sample_rate)
    audio          = np.zeros(num_samples, dtype=np.float32)

    for start, end, freq, vel in segments:
        s0  = int(start * sample_rate)
        s1  = int(end   * sample_rate)
        dur = s1 - s0
        if dur <= 0:
            continue
        t_arr  = np.linspace(0, (end - start), dur, endpoint=False)
        wave   = np.sin(2 * np.pi * freq * t_arr).astype(np.float32)

        # Simple ADSR envelope (attack 5 ms, decay 10 ms, release 20 ms)
        attack_s  = min(int(0.005 * sample_rate), dur // 4)
        decay_s   = min(int(0.010 * sample_rate), dur // 4)
        release_s = min(int(0.020 * sample_rate), dur // 4)
        sustain_s = max(dur - attack_s - decay_s - release_s, 0)

        env = np.concatenate([
            np.linspace(0, 1, attack_s),
            np.linspace(1, 0.8, decay_s),
            np.full(sustain_s, 0.8),
            np.linspace(0.8, 0, release_s),
        ]).astype(np.float32)

        if len(env) < dur:
            env = np.pad(env, (0, dur - len(env)))
        wave *= env[:dur] * (vel / 127.0) * amplitude

        end_idx = min(s0 + dur, num_samples)
        audio[s0:end_idx] += wave[:end_idx - s0]

    # Normalise to prevent clipping
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    audio = np.clip(audio * 0.9, -1.0, 1.0)

    sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")
    print(f"[synth] ✓ WAV saved → {wav_path}")
    return True


def play_wav(wav_path: Path) -> bool:
    """Play a WAV file using pygame."""
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        sound = pygame.mixer.Sound(str(wav_path))
        print(f"[pygame] Playing WAV {wav_path.name} …  (press Ctrl+C to stop)")
        sound.play()
        try:
            time.sleep(sound.get_length())
        except KeyboardInterrupt:
            sound.stop()
        pygame.mixer.quit()
        return True
    except Exception as exc:
        print(f"[wav-play] ✗ {exc}")
        return False


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------

def convert_to_wav(
    midi_path:  Path,
    wav_path:   Path = None,
    soundfont:  str  = None,
    backend:    str  = "auto",
    sample_rate: int = 44100,
) -> Optional[Path]:
    """
    Convert a MIDI file to WAV using the best available backend.

    Args:
        midi_path   : path to source .mid file
        wav_path    : destination .wav path (auto-derived if None)
        soundfont   : path to a .sf2/.sf3 file (FluidSynth only)
        backend     : "auto" | "fluidsynth" | "synth"
        sample_rate : output sample rate in Hz

    Returns:
        Path to the WAV file, or None if all backends failed.
    """
    midi_path = Path(midi_path)
    if wav_path is None:
        wav_path = AUDIO_DIR / (midi_path.stem + ".wav")
    wav_path = Path(wav_path)

    if backend in ("auto", "fluidsynth"):
        if convert_midi_to_wav_fluidsynth(midi_path, wav_path, soundfont, sample_rate):
            return wav_path

    if backend in ("auto", "synth"):
        if convert_midi_to_wav_synth(midi_path, wav_path, sample_rate):
            return wav_path

    print(f"[convert] ✗ All backends failed for {midi_path.name}")
    return None


def play_midi(
    midi_path: Path,
    backend:   str  = "auto",
    soundfont: str  = None,
    wait:      bool = True,
) -> bool:
    """
    Play a MIDI file using the best available backend.

    Args:
        midi_path : path to .mid file
        backend   : "auto" | "fluidsynth" | "pygame"
        soundfont : path to SoundFont (FluidSynth only)
        wait      : block until playback finishes

    Returns:
        True if playback started successfully.
    """
    midi_path = Path(midi_path)

    if backend in ("auto", "fluidsynth"):
        if play_midi_fluidsynth(midi_path, soundfont):
            return True

    if backend in ("auto", "pygame"):
        if play_midi_pygame(midi_path, wait=wait):
            return True

    # Last resort: convert to WAV then play
    print("[play] No direct MIDI playback available — converting to WAV first …")
    wav_path = convert_to_wav(midi_path)
    if wav_path and wav_path.exists():
        return play_wav(wav_path)

    print("[play] ✗ Could not play MIDI. "
          "Install pygame (`pip install pygame`) or FluidSynth.")
    return False


def batch_convert(
    midi_dir:   Path,
    output_dir: Path = None,
    soundfont:  str  = None,
    backend:    str  = "auto",
) -> List[Path]:
    """
    Convert all .mid files in `midi_dir` to WAV.

    Returns:
        List of successfully created WAV Paths.
    """
    midi_dir   = Path(midi_dir)
    output_dir = Path(output_dir) if output_dir else AUDIO_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    if not midi_files:
        print(f"[batch] No MIDI files found in {midi_dir}")
        return []

    print(f"[batch] Converting {len(midi_files)} MIDI file(s) …")
    results = []
    for f in midi_files:
        wav_path = output_dir / (f.stem + ".wav")
        result   = convert_to_wav(f, wav_path=wav_path,
                                  soundfont=soundfont, backend=backend)
        if result:
            results.append(result)

    print(f"[batch] Converted {len(results)}/{len(midi_files)} file(s).")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Play or convert MIDI files to audio."
    )
    parser.add_argument(
        "midi_file", nargs="?",
        help="Path to a .mid file (default: first file in output/midi/)",
    )
    parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "fluidsynth", "pygame", "synth"],
        help="Audio backend (default: auto)",
    )
    parser.add_argument(
        "--export-wav", action="store_true",
        help="Convert the MIDI to a WAV file instead of playing it",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Convert all MIDI files in output/midi/ to WAV",
    )
    parser.add_argument(
        "--soundfont", type=str, default=None,
        help="Path to a SoundFont .sf2/.sf3 file (used by FluidSynth)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for WAV files (default: output/audio/)",
    )
    args = parser.parse_args()

    # ---- Batch convert ----------------------------------------------------
    if args.batch:
        midi_dir = BASE_DIR / "output" / "midi"
        out_dir  = Path(args.output_dir) if args.output_dir else AUDIO_DIR
        batch_convert(midi_dir, output_dir=out_dir,
                      soundfont=args.soundfont, backend=args.backend)
        return

    # ---- Resolve MIDI file ------------------------------------------------
    if args.midi_file:
        midi_path = Path(args.midi_file)
    else:
        # Try to find a generated file automatically
        candidates = sorted((BASE_DIR / "output" / "midi").glob("*.mid"))
        if not candidates:
            print("No MIDI file specified and none found in output/midi/.")
            print("Generate one first:  python src/generator.py")
            sys.exit(1)
        midi_path = candidates[0]
        print(f"[auto] Using {midi_path}")

    if not midi_path.exists():
        print(f"File not found: {midi_path}")
        sys.exit(1)

    # ---- Export or play ---------------------------------------------------
    if args.export_wav:
        out_dir  = Path(args.output_dir) if args.output_dir else AUDIO_DIR
        wav_path = out_dir / (midi_path.stem + ".wav")
        result   = convert_to_wav(
            midi_path, wav_path=wav_path,
            soundfont=args.soundfont, backend=args.backend,
        )
        if result:
            print(f"WAV saved to: {result}")
        else:
            print("Conversion failed.")
            sys.exit(1)
    else:
        success = play_midi(
            midi_path,
            backend=args.backend,
            soundfont=args.soundfont,
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
