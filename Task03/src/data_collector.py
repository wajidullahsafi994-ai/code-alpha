"""
data_collector.py
-----------------
Downloads / collects MIDI files for training the music generation model.

Sources used (all public-domain / freely licensed):
  1. music21 built-in corpus  — Bach chorales, classical pieces
  2. Curated public-domain MIDI URLs (Classic MIDI, BitMidi mirrors)
  3. Synthetic fallback generator — creates simple scale/chord MIDI files
     when no internet is available or downloads fail.

Usage:
    python src/data_collector.py
    python src/data_collector.py --source corpus   # only music21 corpus
    python src/data_collector.py --source download # only URL downloads
    python src/data_collector.py --source synthetic # only synthetic data
    python src/data_collector.py --source all --synthetic-count 20
"""

import os
import sys
import time
import random
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MIDI_DIR = BASE_DIR / "data" / "midi"
MIDI_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Public-domain MIDI download URLs
# Each tuple: (filename, url)
# Sources: BitMidi.com (public domain), musopen.org open MIDI links
# ---------------------------------------------------------------------------
PUBLIC_MIDI_URLS = [
    # Bach
    ("bach_invention_1.mid",
     "https://www.midiworld.com/download/4491"),
    ("bach_cminor_prelude.mid",
     "https://www.midiworld.com/download/2609"),
    # Mozart
    ("mozart_turkish_march.mid",
     "https://www.midiworld.com/download/571"),
    ("mozart_k331.mid",
     "https://www.midiworld.com/download/2480"),
    # Beethoven
    ("beethoven_fur_elise.mid",
     "https://www.midiworld.com/download/501"),
    ("beethoven_moonlight.mid",
     "https://www.midiworld.com/download/1"),
    # Chopin
    ("chopin_nocturne_op9_1.mid",
     "https://www.midiworld.com/download/1209"),
    ("chopin_waltz_op64_1.mid",
     "https://www.midiworld.com/download/2"),
    # Debussy
    ("debussy_clair_de_lune.mid",
     "https://www.midiworld.com/download/3"),
    # Jazz standards (public domain arrangements)
    ("jazz_blues_12bar.mid",
     "https://www.midiworld.com/download/4"),
]

# ---------------------------------------------------------------------------
# 1. music21 corpus collector
# ---------------------------------------------------------------------------

def collect_from_corpus(limit: int = 60) -> int:
    """Copy MIDI files from the music21 built-in corpus."""
    try:
        from music21 import corpus
    except ImportError:
        print("[corpus] music21 not installed — skipping corpus collection.")
        return 0

    print("\n[corpus] Scanning music21 built-in corpus …")

    # Composer-based searches that reliably return MIDI-compatible scores
    queries = [
        ("bach",     "bach"),
        ("beethoven","beethoven"),
        ("mozart",   "mozart"),
        ("schubert", "schubert"),
        ("handel",   "handel"),
    ]

    collected = 0
    for composer, tag in queries:
        if collected >= limit:
            break
        try:
            paths = corpus.getComposer(composer)
        except Exception as exc:
            print(f"  [corpus] Could not get composer '{composer}': {exc}")
            continue

        for p in paths:
            if collected >= limit:
                break
            p = Path(str(p))
            if p.suffix.lower() not in (".mid", ".midi", ".xml", ".mxl", ".krn"):
                continue
            dest = MIDI_DIR / f"{tag}_{p.name}"
            if dest.exists():
                collected += 1
                continue
            try:
                import shutil
                shutil.copy2(str(p), str(dest))
                print(f"  [corpus] Copied: {dest.name}")
                collected += 1
            except Exception as exc:
                print(f"  [corpus] Copy failed for {p.name}: {exc}")

    print(f"[corpus] Collected {collected} files from corpus.")
    return collected


# ---------------------------------------------------------------------------
# 2. URL downloader
# ---------------------------------------------------------------------------

def download_midi_files(urls: list = None, timeout: int = 15) -> int:
    """Download MIDI files from public-domain URLs."""
    if urls is None:
        urls = PUBLIC_MIDI_URLS

    print(f"\n[download] Attempting to download {len(urls)} MIDI files …")
    downloaded = 0

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for filename, url in urls:
        dest = MIDI_DIR / filename
        if dest.exists():
            print(f"  [download] Already exists: {filename}")
            downloaded += 1
            continue
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            # Basic MIDI header check (MThd magic bytes)
            if len(data) > 4 and data[:4] == b"MThd":
                dest.write_bytes(data)
                print(f"  [download] ✓ {filename}  ({len(data)//1024} KB)")
                downloaded += 1
            else:
                print(f"  [download] ✗ {filename} — response is not valid MIDI")
            time.sleep(0.5)          # polite delay
        except urllib.error.HTTPError as e:
            print(f"  [download] ✗ {filename} — HTTP {e.code}")
        except Exception as exc:
            print(f"  [download] ✗ {filename} — {exc}")

    print(f"[download] Downloaded {downloaded} files.")
    return downloaded


# ---------------------------------------------------------------------------
# 3. Synthetic MIDI generator (always works, no internet needed)
# ---------------------------------------------------------------------------

def _write_variable_length(value: int) -> bytes:
    """Encode an integer as MIDI variable-length quantity."""
    result = []
    result.append(value & 0x7F)
    value >>= 7
    while value:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.reverse()
    return bytes(result)


def _make_midi_bytes(
    note_sequence: list,
    tempo: int = 500000,   # microseconds per beat (120 BPM)
    ticks_per_beat: int = 480,
    channel: int = 0,
    velocity: int = 80,
    note_duration_ticks: int = 480,   # one beat per note
) -> bytes:
    """
    Build a minimal Type-0 MIDI file from a flat list of MIDI note numbers.
    Each note is played for `note_duration_ticks` ticks then released.
    """
    import struct

    def var_len(v):
        return _write_variable_length(v)

    events = bytearray()

    # Tempo meta-event: FF 51 03 tt tt tt
    tempo_bytes = struct.pack(">I", tempo)[1:]          # 3 bytes big-endian
    events += b"\x00\xff\x51\x03" + tempo_bytes

    # Note events
    for note in note_sequence:
        note = max(0, min(127, note))
        # delta time 0, Note On
        events += var_len(0)
        events += bytes([0x90 | channel, note, velocity])
        # delta time = note_duration_ticks, Note Off
        events += var_len(note_duration_ticks)
        events += bytes([0x80 | channel, note, 0])

    # End-of-track meta-event
    events += b"\x00\xff\x2f\x00"

    # Build track chunk
    track_data = bytes(events)
    track_chunk = b"MTrk" + struct.pack(">I", len(track_data)) + track_data

    # Build header chunk
    header_chunk = (
        b"MThd"
        + struct.pack(">I", 6)     # chunk length always 6
        + struct.pack(">H", 0)     # format 0 (single track)
        + struct.pack(">H", 1)     # number of tracks
        + struct.pack(">H", ticks_per_beat)
    )

    return header_chunk + track_chunk


# Common scale patterns (intervals from root)
SCALES = {
    "major":           [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":   [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":  [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_major":[0, 2, 4, 7, 9],
    "pentatonic_minor":[0, 3, 5, 7, 10],
    "blues":           [0, 3, 5, 6, 7, 10],
    "dorian":          [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":      [0, 2, 4, 5, 7, 9, 10],
}

GENRES = {
    "classical": {
        "scales":   ["major", "natural_minor", "harmonic_minor"],
        "tempos":   [400000, 500000, 600000],   # 100–150 BPM
        "octaves":  [4, 5],
        "lengths":  [32, 48, 64],
        "velocities": [70, 80, 90],
    },
    "jazz": {
        "scales":   ["dorian", "mixolydian", "blues", "pentatonic_minor"],
        "tempos":   [333333, 375000, 428571],   # 140–180 BPM
        "octaves":  [4, 5],
        "lengths":  [32, 48],
        "velocities": [75, 85, 95],
    },
    "folk": {
        "scales":   ["major", "pentatonic_major", "dorian"],
        "tempos":   [500000, 545454, 600000],
        "octaves":  [4, 5],
        "lengths":  [24, 32, 40],
        "velocities": [65, 75, 80],
    },
}


def generate_synthetic_midi(
    genre: str = "classical",
    root_note: int = None,
    count: int = 20,
    seed: int = None,
) -> int:
    """
    Generate `count` synthetic MIDI files for the given genre.
    Files are saved to data/midi/synthetic_<genre>_<n>.mid
    """
    if seed is not None:
        random.seed(seed)

    print(f"\n[synthetic] Generating {count} synthetic '{genre}' MIDI files …")
    params = GENRES.get(genre, GENRES["classical"])
    created = 0

    for i in range(count):
        scale_name = random.choice(params["scales"])
        scale      = SCALES[scale_name]
        tempo      = random.choice(params["tempos"])
        octave     = random.choice(params["octaves"])
        length     = random.choice(params["lengths"])
        velocity   = random.choice(params["velocities"])
        root       = root_note if root_note is not None else random.randint(48, 65)

        # Build note sequence with some musical variation
        notes = []
        for _ in range(length):
            # Choose a note from the scale (possibly in adjacent octave)
            degree   = random.choice(scale)
            oct_shift = random.choice([0, 0, 0, 12, -12])  # mostly stay, sometimes shift
            note = root + degree + oct_shift
            notes.append(note)

            # Occasionally add a short rest (represented by a silent note we skip)
            if random.random() < 0.1:
                notes.append(-1)        # sentinel for rest

        # Filter out rests for simplicity (just skip them in the MIDI builder)
        notes = [n for n in notes if n != -1]

        midi_bytes = _make_midi_bytes(
            note_sequence=notes,
            tempo=tempo,
            velocity=velocity,
        )

        filename = MIDI_DIR / f"synthetic_{genre}_{i:03d}_{scale_name}.mid"
        filename.write_bytes(midi_bytes)
        created += 1

    print(f"[synthetic] Created {created} synthetic MIDI files.")
    return created


# ---------------------------------------------------------------------------
# 4. Dataset summary
# ---------------------------------------------------------------------------

def print_dataset_summary():
    """Print a summary of MIDI files currently in data/midi."""
    midi_files = list(MIDI_DIR.glob("*.mid")) + list(MIDI_DIR.glob("*.midi"))
    xml_files  = list(MIDI_DIR.glob("*.xml")) + list(MIDI_DIR.glob("*.mxl"))

    print(f"\n{'='*50}")
    print(f"  Dataset Summary")
    print(f"{'='*50}")
    print(f"  MIDI files : {len(midi_files)}")
    print(f"  XML files  : {len(xml_files)}")
    print(f"  Total      : {len(midi_files) + len(xml_files)}")
    print(f"  Location   : {MIDI_DIR}")

    # Group by prefix
    prefixes = {}
    for f in midi_files + xml_files:
        prefix = f.stem.split("_")[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    if prefixes:
        print(f"\n  By type:")
        for p, c in sorted(prefixes.items()):
            print(f"    {p:<20} {c:>4} files")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect MIDI training data for the AI music generator."
    )
    parser.add_argument(
        "--source",
        choices=["corpus", "download", "synthetic", "all"],
        default="all",
        help="Which data source(s) to use (default: all)",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=60,
        help="Max number of files to copy from music21 corpus (default: 60)",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=30,
        help="Number of synthetic MIDI files to generate per genre (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic generation (default: 42)",
    )
    args = parser.parse_args()

    total = 0

    if args.source in ("corpus", "all"):
        total += collect_from_corpus(limit=args.corpus_limit)

    if args.source in ("download", "all"):
        total += download_midi_files()

    if args.source in ("synthetic", "all"):
        for genre in ["classical", "jazz", "folk"]:
            total += generate_synthetic_midi(
                genre=genre,
                count=args.synthetic_count,
                seed=args.seed,
            )

    print_dataset_summary()

    if total == 0:
        print("WARNING: No MIDI files were collected. Check your internet connection")
        print("         or run with --source synthetic to generate training data.")
        sys.exit(1)

    print(f"Data collection complete. {total} files ready for preprocessing.")


if __name__ == "__main__":
    main()
