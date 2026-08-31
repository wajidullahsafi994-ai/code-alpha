"""
preprocessor.py
---------------
Converts raw MIDI / MusicXML files into tokenised note sequences
suitable for LSTM training.

Pipeline:
  1. Parse each file with music21
  2. Extract (pitch, duration, offset) events per part
  3. Encode pitches as string tokens  e.g. "C4", "E4", "rest"
  4. Build a vocabulary (token → integer index)
  5. Slide a fixed-length window over every song to create
     (X, y) training pairs
  6. Persist sequences, vocab, and numpy arrays to data/processed/

Usage:
    python src/preprocessor.py
    python src/preprocessor.py --seq-len 64 --min-files 5
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
MIDI_DIR      = BASE_DIR / "data" / "midi"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REST_TOKEN      = "rest"
PAD_TOKEN       = "<PAD>"
UNK_TOKEN       = "<UNK>"
SPECIAL_TOKENS  = [PAD_TOKEN, UNK_TOKEN, REST_TOKEN]

# Duration quantisation buckets (in quarter-note units)
DURATION_BUCKETS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def quantise_duration(dur: float) -> float:
    """Snap a raw duration to the nearest bucket."""
    if dur <= 0:
        return 0.25
    return min(DURATION_BUCKETS, key=lambda b: abs(b - dur))


# ---------------------------------------------------------------------------
# 1.  Parse a single file with music21
# ---------------------------------------------------------------------------

def parse_file(filepath: Path) -> list:
    """
    Parse a MIDI or MusicXML file and return a flat list of token strings.

    Token format:
        pitch  →  "<note_name><octave>_<duration_bucket>"
                  e.g. "C4_1.0",  "F#5_0.5"
        chord  →  "<n1>.<n2>.<n3>_<duration>"
                  e.g. "C4.E4.G4_1.0"
        rest   →  "rest_<duration>"
                  e.g. "rest_0.5"
    """
    try:
        from music21 import converter, chord, note, stream
    except ImportError:
        raise RuntimeError(
            "music21 is not installed. Run: pip install music21"
        )

    try:
        score = converter.parse(str(filepath))
    except Exception as exc:
        print(f"  [parse] Could not parse {filepath.name}: {exc}")
        return []

    tokens = []

    # Flatten into a single stream of notes / chords / rests
    try:
        flat = score.flatten().notesAndRests
    except Exception:
        try:
            flat = score.flat.notesAndRests
        except Exception as exc:
            print(f"  [parse] Could not flatten {filepath.name}: {exc}")
            return []

    for element in flat:
        dur = quantise_duration(float(element.duration.quarterLength))

        if isinstance(element, note.Rest):
            tokens.append(f"{REST_TOKEN}_{dur}")

        elif isinstance(element, note.Note):
            pitch_str = element.pitch.nameWithOctave   # e.g. "C4"
            tokens.append(f"{pitch_str}_{dur}")

        elif isinstance(element, chord.Chord):
            # Sort pitches so "C4.E4.G4" is canonical regardless of voicing
            pitches = ".".join(
                sorted(p.nameWithOctave for p in element.pitches)
            )
            tokens.append(f"{pitches}_{dur}")

    return tokens


# ---------------------------------------------------------------------------
# 2.  Build vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(all_token_lists: list, min_freq: int = 2) -> tuple:
    """
    Build token ↔ integer mappings from a list of token sequences.

    Returns:
        token_to_int : dict[str, int]
        int_to_token : dict[int, str]
    """
    counter = Counter(tok for seq in all_token_lists for tok in seq)

    # Keep tokens that appear at least min_freq times
    vocab = SPECIAL_TOKENS + sorted(
        tok for tok, cnt in counter.items()
        if cnt >= min_freq and tok not in SPECIAL_TOKENS
    )

    token_to_int = {tok: idx for idx, tok in enumerate(vocab)}
    int_to_token = {idx: tok for tok, idx in token_to_int.items()}

    return token_to_int, int_to_token


# ---------------------------------------------------------------------------
# 3.  Sliding-window sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    token_lists: list,
    token_to_int: dict,
    seq_len: int = 64,
) -> tuple:
    """
    Slide a window of length `seq_len` over every song to create (X, y) pairs.

    X shape: (N, seq_len)
    y shape: (N,)           — next token index (classification target)
    """
    unk_idx = token_to_int[UNK_TOKEN]
    X, y = [], []

    for tokens in token_lists:
        if len(tokens) < seq_len + 1:
            continue                         # skip very short pieces

        indices = [token_to_int.get(t, unk_idx) for t in tokens]

        for i in range(len(indices) - seq_len):
            X.append(indices[i : i + seq_len])
            y.append(indices[i + seq_len])

    if not X:
        return np.array([]), np.array([])

    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# 4.  Normalised float sequences (alternative input format)
# ---------------------------------------------------------------------------

def normalise_sequences(X: np.ndarray, vocab_size: int) -> np.ndarray:
    """Return X scaled to [0, 1] for networks that prefer float input."""
    return X.astype(np.float32) / float(vocab_size - 1)


# ---------------------------------------------------------------------------
# 5.  Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(
    midi_dir: Path = MIDI_DIR,
    processed_dir: Path = PROCESSED_DIR,
    seq_len: int = 64,
    min_freq: int = 2,
    min_files: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
        X, y, X_norm, token_to_int, int_to_token,
        vocab_size, seq_len, all_tokens
    """
    # ---- Gather MIDI / XML files ----------------------------------------
    extensions = (".mid", ".midi", ".xml", ".mxl", ".krn")
    files = [
        f for f in midi_dir.iterdir()
        if f.suffix.lower() in extensions
    ]

    if not files:
        raise FileNotFoundError(
            f"No MIDI/XML files found in {midi_dir}.\n"
            "Run  python src/data_collector.py  first."
        )

    if verbose:
        print(f"\n[preprocess] Found {len(files)} files in {midi_dir}")

    # ---- Parse all files -------------------------------------------------
    all_token_lists = []
    parse_errors    = 0

    for i, f in enumerate(files, 1):
        if verbose:
            print(f"  [{i:>3}/{len(files)}] Parsing {f.name} …", end=" ")
        tokens = parse_file(f)
        if tokens:
            all_token_lists.append(tokens)
            if verbose:
                print(f"{len(tokens)} tokens")
        else:
            parse_errors += 1
            if verbose:
                print("skipped (no tokens)")

    successfully_parsed = len(all_token_lists)
    if verbose:
        print(
            f"\n[preprocess] Parsed {successfully_parsed} files "
            f"({parse_errors} skipped)"
        )

    if successfully_parsed < min_files:
        raise RuntimeError(
            f"Only {successfully_parsed} files parsed successfully "
            f"(minimum required: {min_files}).\n"
            "Add more MIDI files or lower --min-files."
        )

    # ---- Build vocabulary ------------------------------------------------
    token_to_int, int_to_token = build_vocabulary(all_token_lists, min_freq=min_freq)
    vocab_size = len(token_to_int)

    if verbose:
        print(f"[preprocess] Vocabulary size: {vocab_size} unique tokens")

    # ---- Build sequences -------------------------------------------------
    X, y = build_sequences(all_token_lists, token_to_int, seq_len=seq_len)

    if X.size == 0:
        raise RuntimeError(
            f"No training sequences generated. "
            f"Try lowering --seq-len (current: {seq_len})."
        )

    X_norm = normalise_sequences(X, vocab_size)

    if verbose:
        print(f"[preprocess] Training samples : {len(X):,}")
        print(f"[preprocess] Sequence length  : {seq_len}")
        print(f"[preprocess] X shape          : {X.shape}")
        print(f"[preprocess] y shape          : {y.shape}")

    # ---- Flatten all tokens for optional use ----------------------------
    all_tokens = [tok for seq in all_token_lists for tok in seq]

    # ---- Persist to disk ------------------------------------------------
    np.save(processed_dir / "X.npy",      X)
    np.save(processed_dir / "y.npy",      y)
    np.save(processed_dir / "X_norm.npy", X_norm)

    with open(processed_dir / "token_to_int.pkl", "wb") as f:
        pickle.dump(token_to_int, f)
    with open(processed_dir / "int_to_token.pkl", "wb") as f:
        pickle.dump(int_to_token, f)
    with open(processed_dir / "all_tokens.pkl", "wb") as f:
        pickle.dump(all_tokens, f)

    # Save metadata as a simple text summary
    meta_path = processed_dir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"seq_len={seq_len}\n")
        f.write(f"num_samples={len(X)}\n")
        f.write(f"num_files_parsed={successfully_parsed}\n")
        f.write(f"num_files_total={len(files)}\n")

    if verbose:
        print(f"\n[preprocess] Saved processed data to {processed_dir}")
        print(f"  X.npy          {X.shape}")
        print(f"  y.npy          {y.shape}")
        print(f"  X_norm.npy     {X_norm.shape}")
        print(f"  token_to_int.pkl  ({vocab_size} tokens)")
        print(f"  int_to_token.pkl")
        print(f"  all_tokens.pkl   ({len(all_tokens):,} total tokens)")
        print(f"  metadata.txt")

    return {
        "X":            X,
        "y":            y,
        "X_norm":       X_norm,
        "token_to_int": token_to_int,
        "int_to_token": int_to_token,
        "vocab_size":   vocab_size,
        "seq_len":      seq_len,
        "all_tokens":   all_tokens,
    }


# ---------------------------------------------------------------------------
# 6.  Load already-processed data
# ---------------------------------------------------------------------------

def load_processed(processed_dir: Path = PROCESSED_DIR) -> dict:
    """
    Load previously saved preprocessing outputs.
    Returns the same dict structure as preprocess().
    """
    required = ["X.npy", "y.npy", "X_norm.npy",
                "token_to_int.pkl", "int_to_token.pkl"]
    for fname in required:
        if not (processed_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing processed file: {processed_dir / fname}\n"
                "Run preprocessing first."
            )

    X      = np.load(processed_dir / "X.npy")
    y      = np.load(processed_dir / "y.npy")
    X_norm = np.load(processed_dir / "X_norm.npy")

    with open(processed_dir / "token_to_int.pkl", "rb") as f:
        token_to_int = pickle.load(f)
    with open(processed_dir / "int_to_token.pkl", "rb") as f:
        int_to_token = pickle.load(f)

    all_tokens = []
    all_tok_path = processed_dir / "all_tokens.pkl"
    if all_tok_path.exists():
        with open(all_tok_path, "rb") as f:
            all_tokens = pickle.load(f)

    # Read metadata
    vocab_size = len(token_to_int)
    seq_len    = X.shape[1] if X.ndim == 2 else 64
    meta_path  = processed_dir / "metadata.txt"
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "seq_len":
                seq_len = int(v.strip())

    return {
        "X":            X,
        "y":            y,
        "X_norm":       X_norm,
        "token_to_int": token_to_int,
        "int_to_token": int_to_token,
        "vocab_size":   vocab_size,
        "seq_len":      seq_len,
        "all_tokens":   all_tokens,
    }


# ---------------------------------------------------------------------------
# 7.  Visualisation helper
# ---------------------------------------------------------------------------

def plot_token_distribution(
    token_to_int: dict,
    all_tokens: list,
    top_n: int = 30,
    save_path: Path = None,
):
    """Bar chart of the top-N most frequent tokens."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping plot.")
        return

    counter  = Counter(all_tokens)
    top_toks = counter.most_common(top_n)
    labels   = [t for t, _ in top_toks]
    counts   = [c for _, c in top_toks]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(labels)), counts, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Top {top_n} Most Frequent Tokens")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150)
        print(f"[plot] Saved token distribution to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MIDI files into training sequences."
    )
    parser.add_argument(
        "--seq-len", type=int, default=64,
        help="Length of each input sequence window (default: 64)",
    )
    parser.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum token frequency to include in vocabulary (default: 2)",
    )
    parser.add_argument(
        "--min-files", type=int, default=3,
        help="Minimum number of successfully parsed files required (default: 3)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show / save a token frequency bar chart after processing",
    )
    args = parser.parse_args()

    try:
        result = preprocess(
            seq_len=args.seq_len,
            min_freq=args.min_freq,
            min_files=args.min_files,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    if args.plot:
        plot_token_distribution(
            result["token_to_int"],
            result["all_tokens"],
            save_path=PROCESSED_DIR / "token_distribution.png",
        )

    print("\nPreprocessing complete.")
    print(f"  Vocabulary : {result['vocab_size']} tokens")
    print(f"  Samples    : {len(result['X']):,}")
    print(f"  Seq length : {result['seq_len']}")


if __name__ == "__main__":
    main()
